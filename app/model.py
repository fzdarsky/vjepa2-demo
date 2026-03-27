# app/model.py
import time

import numpy as np
import torch
from transformers import AutoModelForVideoClassification, AutoVideoProcessor

from app.schemas import Prediction
from app.telemetry import get_meter, get_tracer


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
        self.model.eval()
        self.id2label: dict[int, str] = self.model.config.id2label

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

    def predict(
        self,
        frames: np.ndarray,
        top_k: int = 5,
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

        with tracer.start_as_current_span("preprocess"):
            inputs = self.processor(list(frames), return_tensors="pt")
            key = (
                "pixel_values_videos"
                if "pixel_values_videos" in inputs
                else "pixel_values"
            )
            pixel_values = inputs[key].to(self.device)

        with tracer.start_as_current_span("inference"):
            with torch.no_grad():
                outputs = self.model(pixel_values)

        with tracer.start_as_current_span("postprocess"):
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

        if stride is not None and source_fps > 0:
            rt_threshold = stride / source_fps
            if clip_duration > rt_threshold:
                self._rt_violations.add(1)

        return predictions
