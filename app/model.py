# app/model.py
import time
from typing import Any

import numpy as np
import torch
from torch import nn
from transformers import AutoModelForVideoClassification, AutoVideoProcessor

from app.schemas import DEFAULT_TOP_K, Prediction
from app.telemetry import get_meter, get_tracer


class TracingHooks:
    """Manages OTel spans for PyTorch module forward passes.

    Uses paired pre/post hooks to create spans that accurately measure
    submodule execution time, including GPU synchronization.
    """

    def __init__(self, device: str):
        self.device = device
        self._active_spans: dict[int, Any] = {}  # module id -> (span, token)
        self._tracer = get_tracer()

    def _sync_device(self) -> None:
        """Synchronize device for accurate timing."""
        if self.device == "cuda":
            torch.cuda.synchronize()
        elif self.device == "mps":
            torch.mps.synchronize()

    def _make_pre_hook(self, span_name: str):
        """Create a pre-forward hook that starts a span."""
        def hook(module: nn.Module, args: tuple) -> None:
            self._sync_device()
            span = self._tracer.start_span(span_name)
            self._active_spans[id(module)] = span
        return hook

    def _make_post_hook(self, span_name: str):
        """Create a post-forward hook that ends the span."""
        def hook(module: nn.Module, args: tuple, output: Any) -> None:
            self._sync_device()
            span = self._active_spans.pop(id(module), None)
            if span is not None:
                span.end()
        return hook

    def register(self, module: nn.Module, span_name: str) -> None:
        """Register tracing hooks on a module."""
        module.register_forward_pre_hook(self._make_pre_hook(span_name))
        module.register_forward_hook(self._make_post_hook(span_name))


def select_device(requested: str | None = None) -> str:
    """Auto-detect or validate the compute device."""
    if requested:
        device = requested.lower()
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        if device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available")
        return device

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class VJepa2Model:
    def __init__(self, model_path: str, device: str):
        self.device = device
        self.processor = AutoVideoProcessor.from_pretrained(model_path)
        self.model = AutoModelForVideoClassification.from_pretrained(model_path)
        self.model.to(device)
        self.model.train(False)  # Set to inference mode
        self.id2label: dict[int, str] = self.model.config.id2label

        # Register tracing hooks for JEPA submodules
        self._tracing_hooks = TracingHooks(device)
        self._register_tracing_hooks()

        meter = get_meter()
        self._clip_duration = meter.create_histogram(
            "vjepa2_clip_processing_seconds",
            description="Wall time to process one clip",
            unit="s",
        )
        self._clips_total = meter.create_counter(
            "vjepa2_clips_processed_total",
            description="Total clips processed",
        )
        self._rt_violations = meter.create_counter(
            "vjepa2_clip_realtime_violations_total",
            description="Clips where processing exceeded realtime threshold",
        )

    def _register_tracing_hooks(self) -> None:
        """Register OTel tracing hooks on JEPA model submodules."""
        # Map submodule paths to span names
        # These paths are specific to HuggingFace V-JEPA2 model structure
        hook_config = [
            ("vjepa2.encoder", "jepa_encode"),
            ("vjepa2.predictor", "jepa_predict"),
            ("pooler", "jepa_pool"),
        ]

        for module_path, span_name in hook_config:
            try:
                # Get submodule by traversing the path (e.g., "vjepa2.encoder")
                module = self.model
                for part in module_path.split("."):
                    module = getattr(module, part)
                self._tracing_hooks.register(module, span_name)
            except AttributeError:
                # Submodule not found - skip (model may have different structure)
                pass

    def predict(
        self,
        frames: np.ndarray,
        top_k: int = DEFAULT_TOP_K,
        stride: int | None = None,
        source_fps: float = 30.0,
    ) -> list[Prediction]:
        """Run inference on video frames.

        Args:
            frames: numpy array of shape (num_frames, H, W, 3), dtype uint8.
            top_k: number of top predictions to return.
            stride: stride used for clip extraction (for RT ratio calculation).
            source_fps: source video FPS (for RT ratio calculation).

        Returns:
            List of Prediction(label, score), sorted by score descending.
        """
        tracer = get_tracer()
        clip_start = time.monotonic()

        with tracer.start_as_current_span("input_preprocess"):
            inputs = self.processor(list(frames), return_tensors="pt")
            key = (
                "pixel_values_videos"
                if "pixel_values_videos" in inputs
                else "pixel_values"
            )
            pixel_values = inputs[key].to(self.device)

        # Forward hooks on submodules create jepa_encode, jepa_predict, jepa_pool spans
        with torch.no_grad():
            outputs = self.model(pixel_values)

        with tracer.start_as_current_span("output_postprocess"):
            probs = torch.softmax(outputs.logits, dim=-1)[0]
            top_scores, top_indices = torch.topk(probs, k=top_k)

            predictions = [
                Prediction(
                    label=self.id2label[idx.item()],
                    score=round(score.item(), 6),
                )
                for score, idx in zip(top_scores, top_indices)
            ]

        clip_duration = time.monotonic() - clip_start
        self._clip_duration.record(clip_duration)
        self._clips_total.add(1)

        effective_stride = stride if stride is not None else len(frames)
        if source_fps > 0:
            rt_threshold = effective_stride / source_fps
            if clip_duration > rt_threshold:
                self._rt_violations.add(1)

        return predictions
