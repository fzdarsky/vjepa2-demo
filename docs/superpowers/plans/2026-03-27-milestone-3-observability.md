# Milestone 3 — Observability & Metrics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add OpenTelemetry-based observability (metrics + tracing) to the V-JEPA2 inference service with a turnkey Grafana dashboard.

**Architecture:** The FastAPI app uses the OTel Python SDK to emit metrics and trace spans via OTLP to an OTel Collector. The Collector fans out to Prometheus (metrics) and Jaeger (traces). Grafana reads from both with a pre-provisioned dashboard. GPU metrics come from dcgm-exporter. The entire observability stack is optional via `--profile observability` in Podman Compose.

**Tech Stack:** opentelemetry-api, opentelemetry-sdk, opentelemetry-exporter-otlp, opentelemetry-instrumentation-fastapi, Prometheus, Jaeger, Grafana, OTel Collector

**Spec:** `docs/superpowers/specs/2026-03-27-milestone-3-design.md`

---

## File Structure

| File | Responsibility |
|------|---------------|
| `app/telemetry.py` | NEW — OTel initialization (meter, tracer, providers), no-op fallback |
| `app/main.py` | MODIFY — call `init_telemetry()`, add request-level metrics middleware |
| `app/model.py` | MODIFY — trace spans for preprocess/inference/postprocess, clip metrics |
| `app/video.py` | MODIFY — trace span for decode_video |
| `requirements.txt` | MODIFY — add OTel packages |
| `configs/otel-collector.yaml` | NEW — Collector pipeline config |
| `configs/prometheus.yaml` | NEW — Prometheus scrape config |
| `configs/grafana/datasources.yaml` | NEW — Grafana datasource provisioning |
| `configs/grafana/dashboards/dashboard.yaml` | NEW — Grafana dashboard provisioning config |
| `configs/grafana/dashboards/vjepa2.json` | NEW — Dashboard JSON |
| `compose.yaml` | MODIFY — add observability profile services |
| `tests/test_telemetry.py` | NEW — telemetry unit tests |
| `README.md` | MODIFY — observability usage docs |

---

### Task 1: Add OTel Dependencies

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Add OTel packages to requirements.txt**

Append to `requirements.txt`:

```text
opentelemetry-api>=1.20.0
opentelemetry-sdk>=1.20.0
opentelemetry-exporter-otlp>=1.20.0
opentelemetry-instrumentation-fastapi>=0.41b0
```

- [ ] **Step 2: Install and verify**

Run: `pip install -r requirements.txt`
Expected: All packages install successfully. No conflicts.

- [ ] **Step 3: Verify existing tests still pass**

Run: `pytest tests/ -v`
Expected: All 27 tests pass.

- [ ] **Step 4: Commit**

```bash
git add requirements.txt
git commit -m "feat(m3): add OpenTelemetry dependencies"
```

---

### Task 2: Create `app/telemetry.py` with No-Op Fallback

**Files:**
- Create: `app/telemetry.py`
- Test: `tests/test_telemetry.py`

- [ ] **Step 1: Write the failing test for telemetry init**

Create `tests/test_telemetry.py`:

```python
# tests/test_telemetry.py
from app.telemetry import init_telemetry, get_meter, get_tracer


def test_telemetry_init():
    """init_telemetry() sets up meter and tracer providers without error."""
    init_telemetry(service_name="test-service", device="cpu")
    meter = get_meter()
    tracer = get_tracer()
    assert meter is not None
    assert tracer is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_telemetry.py::test_telemetry_init -v`
Expected: FAIL with `ModuleNotFoundError` or `ImportError`

- [ ] **Step 3: Write `app/telemetry.py`**

```python
# app/telemetry.py
"""OpenTelemetry setup with graceful no-op fallback."""

import time

try:
    from opentelemetry import metrics, trace
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
        OTLPMetricExporter,
    )
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
        OTLPSpanExporter,
    )
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    _HAS_OTEL = True
except ImportError:
    _HAS_OTEL = False


_meter = None
_tracer = None


def init_telemetry(
    service_name: str = "vjepa2-server",
    device: str = "cpu",
    otlp_endpoint: str | None = None,
) -> None:
    """Initialize OTel meter and tracer providers.

    If OTel packages are not installed, this is a silent no-op.
    """
    global _meter, _tracer

    if not _HAS_OTEL:
        return

    resource = Resource.create(
        {
            "service.name": service_name,
            "service.version": "0.3.0",
            "vjepa2.device": device,
        }
    )

    # Metrics
    exporter_kwargs = {}
    if otlp_endpoint:
        exporter_kwargs["endpoint"] = otlp_endpoint
        exporter_kwargs["insecure"] = True
    metric_reader = PeriodicExportingMetricReader(
        OTLPMetricExporter(**exporter_kwargs),
        export_interval_millis=5000,
    )
    meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
    metrics.set_meter_provider(meter_provider)
    _meter = metrics.get_meter("vjepa2")

    # Traces
    tracer_provider = TracerProvider(resource=resource)
    span_exporter = OTLPSpanExporter(**exporter_kwargs)
    tracer_provider.add_span_processor(BatchSpanProcessor(span_exporter))
    trace.set_tracer_provider(tracer_provider)
    _tracer = trace.get_tracer("vjepa2")


def instrument_fastapi(app) -> None:
    """Instrument a FastAPI app with OTel middleware."""
    if _HAS_OTEL:
        FastAPIInstrumentor.instrument_app(app)


def get_meter():
    """Return the OTel meter, or a no-op meter if OTel is not available."""
    if _meter is not None:
        return _meter
    if _HAS_OTEL:
        return metrics.get_meter("vjepa2")
    return _NoOpMeter()


def get_tracer():
    """Return the OTel tracer, or a no-op tracer if OTel is not available."""
    if _tracer is not None:
        return _tracer
    if _HAS_OTEL:
        return trace.get_tracer("vjepa2")
    return _NoOpTracer()


class _NoOpMeter:
    """Stub meter that creates instruments which do nothing."""

    def create_counter(self, name, **kwargs):
        return _NoOpInstrument()

    def create_histogram(self, name, **kwargs):
        return _NoOpInstrument()

    def create_up_down_counter(self, name, **kwargs):
        return _NoOpInstrument()

    def create_gauge(self, name, **kwargs):
        return _NoOpInstrument()


class _NoOpInstrument:
    """Stub instrument that accepts any call silently."""

    def add(self, *args, **kwargs):
        pass

    def record(self, *args, **kwargs):
        pass

    def set(self, *args, **kwargs):
        pass


class _NoOpTracer:
    """Stub tracer that returns a no-op span context manager."""

    def start_as_current_span(self, name, **kwargs):
        return _NoOpSpanContext()


class _NoOpSpanContext:
    """Context manager that does nothing."""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def set_attribute(self, key, value):
        pass
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_telemetry.py::test_telemetry_init -v`
Expected: PASS

- [ ] **Step 5: Write the no-op fallback test**

Add to `tests/test_telemetry.py`:

```python
from unittest.mock import patch


def test_telemetry_noop_fallback():
    """When OTel is unavailable, get_meter/get_tracer return no-op stubs."""
    import app.telemetry as tel

    original_has_otel = tel._HAS_OTEL
    original_meter = tel._meter
    original_tracer = tel._tracer
    try:
        tel._HAS_OTEL = False
        tel._meter = None
        tel._tracer = None

        meter = tel.get_meter()
        tracer = tel.get_tracer()

        # No-op meter creates instruments that don't crash
        counter = meter.create_counter("test")
        counter.add(1)

        histogram = meter.create_histogram("test")
        histogram.record(0.5)

        # No-op tracer creates span context managers that don't crash
        with tracer.start_as_current_span("test"):
            pass
    finally:
        tel._HAS_OTEL = original_has_otel
        tel._meter = original_meter
        tel._tracer = original_tracer
```

- [ ] **Step 6: Run test to verify it passes**

Run: `pytest tests/test_telemetry.py::test_telemetry_noop_fallback -v`
Expected: PASS

- [ ] **Step 7: Verify all existing tests still pass**

Run: `pytest tests/ -v`
Expected: All 27 existing tests + 2 new tests pass (29 total).

- [ ] **Step 8: Commit**

```bash
git add app/telemetry.py tests/test_telemetry.py
git commit -m "feat(m3): add telemetry module with OTel init and no-op fallback"
```

---

### Task 3: Add Clip-Level Metrics to `app/model.py`

**Files:**
- Modify: `app/model.py:35-62`
- Test: `tests/test_telemetry.py`

- [ ] **Step 1: Write the failing test for clip metrics**

Add to `tests/test_telemetry.py`:

```python
import numpy as np
from unittest.mock import MagicMock, patch
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader


def test_clip_metrics_recorded():
    """After an inference call, clip processing histogram has an observation."""
    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader])

    import app.telemetry as tel
    original_meter = tel._meter
    try:
        tel._meter = provider.get_meter("vjepa2")

        with (
            patch("app.model.AutoModelForVideoClassification") as MockModel,
            patch("app.model.AutoVideoProcessor") as MockProcessor,
        ):
            import torch

            processor_instance = MagicMock()
            processor_instance.return_value = {
                "pixel_values_videos": torch.randn(1, 16, 3, 256, 256)
            }
            MockProcessor.from_pretrained.return_value = processor_instance

            model_instance = MagicMock()
            model_instance.config.id2label = {i: f"Action {i}" for i in range(174)}
            logits = torch.randn(1, 174)
            model_instance.return_value = MagicMock(logits=logits)
            model_instance.to.return_value = model_instance
            model_instance.eval.return_value = model_instance
            MockModel.from_pretrained.return_value = model_instance

            from app.model import VJepa2Model

            model = VJepa2Model(model_path="test", device="cpu")
            frames = np.random.randint(0, 255, (16, 256, 256, 3), dtype=np.uint8)
            model.predict(frames, top_k=5)

        metrics_data = reader.get_metrics_data()
        metric_names = []
        for resource_metric in metrics_data.resource_metrics:
            for scope_metric in resource_metric.scope_metrics:
                for metric in scope_metric.metrics:
                    metric_names.append(metric.name)

        assert "vjepa2_clip_processing_seconds" in metric_names
        assert "vjepa2_clips_processed_total" in metric_names
    finally:
        tel._meter = original_meter
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_telemetry.py::test_clip_metrics_recorded -v`
Expected: FAIL — no metrics recorded yet.

- [ ] **Step 3: Add metrics instrumentation to `app/model.py`**

Replace the full contents of `app/model.py` with:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_telemetry.py::test_clip_metrics_recorded -v`
Expected: PASS

- [ ] **Step 5: Run all tests**

Run: `pytest tests/ -v`
Expected: All tests pass (existing 27 + new telemetry tests).

- [ ] **Step 6: Commit**

```bash
git add app/model.py tests/test_telemetry.py
git commit -m "feat(m3): add clip-level metrics to model inference"
```

---

### Task 4: Add RT Violation Metric Test

**Files:**
- Test: `tests/test_telemetry.py`

- [ ] **Step 1: Write the RT violation test**

Add to `tests/test_telemetry.py`:

```python
import pytest


def test_realtime_violation_counted():
    """When clip processing exceeds stride/fps, violation counter increments."""
    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader])

    import app.telemetry as tel
    original_meter = tel._meter
    try:
        tel._meter = provider.get_meter("vjepa2")

        with (
            patch("app.model.AutoModelForVideoClassification") as MockModel,
            patch("app.model.AutoVideoProcessor") as MockProcessor,
            patch("app.model.time") as mock_time,
        ):
            import torch

            processor_instance = MagicMock()
            processor_instance.return_value = {
                "pixel_values_videos": torch.randn(1, 16, 3, 256, 256)
            }
            MockProcessor.from_pretrained.return_value = processor_instance

            model_instance = MagicMock()
            model_instance.config.id2label = {i: f"Action {i}" for i in range(174)}
            logits = torch.randn(1, 174)
            model_instance.return_value = MagicMock(logits=logits)
            model_instance.to.return_value = model_instance
            model_instance.eval.return_value = model_instance
            MockModel.from_pretrained.return_value = model_instance

            # Simulate 0.5s processing time — exceeds stride=8 / fps=30 = 0.267s
            mock_time.monotonic.side_effect = [0.0, 0.5]

            from app.model import VJepa2Model

            model = VJepa2Model(model_path="test", device="cpu")
            frames = np.random.randint(0, 255, (16, 256, 256, 3), dtype=np.uint8)
            model.predict(frames, top_k=5, stride=8, source_fps=30.0)

        metrics_data = reader.get_metrics_data()
        for resource_metric in metrics_data.resource_metrics:
            for scope_metric in resource_metric.scope_metrics:
                for metric in scope_metric.metrics:
                    if metric.name == "vjepa2_clip_realtime_violations_total":
                        point = metric.data.data_points[0]
                        assert point.value >= 1
                        return

        pytest.fail("vjepa2_clip_realtime_violations_total metric not found")
    finally:
        tel._meter = original_meter
```

- [ ] **Step 2: Run test to verify it passes**

Run: `pytest tests/test_telemetry.py::test_realtime_violation_counted -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_telemetry.py
git commit -m "test(m3): add realtime violation counter test"
```

---

### Task 5: Add Trace Spans to `app/video.py`

**Files:**
- Modify: `app/video.py:58-62`
- Test: `tests/test_telemetry.py`

- [ ] **Step 1: Write the failing test for trace spans**

Add to `tests/test_telemetry.py`:

```python
from opentelemetry.sdk.trace import TracerProvider as SdkTracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory import InMemorySpanExporter


def test_trace_spans_created(sample_video_path):
    """An inference call produces trace spans for decode, preprocess, inference, postprocess."""
    span_exporter = InMemorySpanExporter()
    tracer_provider = SdkTracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(span_exporter))

    import app.telemetry as tel
    original_tracer = tel._tracer
    original_meter = tel._meter
    try:
        tel._tracer = tracer_provider.get_tracer("vjepa2")
        # Use no-op meter to avoid needing a real meter provider
        tel._meter = tel._NoOpMeter()

        with (
            patch("app.model.AutoModelForVideoClassification") as MockModel,
            patch("app.model.AutoVideoProcessor") as MockProcessor,
        ):
            import torch

            processor_instance = MagicMock()
            processor_instance.return_value = {
                "pixel_values_videos": torch.randn(1, 16, 3, 256, 256)
            }
            MockProcessor.from_pretrained.return_value = processor_instance

            model_instance = MagicMock()
            model_instance.config.id2label = {i: f"Action {i}" for i in range(174)}
            logits = torch.randn(1, 174)
            model_instance.return_value = MagicMock(logits=logits)
            model_instance.to.return_value = model_instance
            model_instance.eval.return_value = model_instance
            MockModel.from_pretrained.return_value = model_instance

            from app.model import VJepa2Model
            from app.video import decode_video

            model = VJepa2Model(model_path="test", device="cpu")
            frames = decode_video(sample_video_path, num_frames=16)
            model.predict(frames, top_k=5)

        span_names = [span.name for span in span_exporter.get_finished_spans()]
        assert "decode_video" in span_names
        assert "preprocess" in span_names
        assert "inference" in span_names
        assert "postprocess" in span_names
    finally:
        tel._tracer = original_tracer
        tel._meter = original_meter
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_telemetry.py::test_trace_spans_created -v`
Expected: FAIL — `decode_video` span not in span names.

- [ ] **Step 3: Add trace span to `app/video.py`**

Update the `decode_video` function in `app/video.py` (lines 58-62):

Replace:

```python
def decode_video(source: str | Path, num_frames: int) -> np.ndarray:
    """Convenience: return frames from first clip.
    Backward-compatible with milestone 1 callers."""
    clip = next(iter_clips(source, num_frames, stride=num_frames))
    return clip.frames
```

With:

```python
def decode_video(source: str | Path, num_frames: int) -> np.ndarray:
    """Convenience: return frames from first clip.
    Backward-compatible with milestone 1 callers."""
    from app.telemetry import get_tracer

    tracer = get_tracer()
    with tracer.start_as_current_span("decode_video"):
        clip = next(iter_clips(source, num_frames, stride=num_frames))
        return clip.frames
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_telemetry.py::test_trace_spans_created -v`
Expected: PASS

- [ ] **Step 5: Run all tests**

Run: `pytest tests/ -v`
Expected: All tests pass.

- [ ] **Step 6: Commit**

```bash
git add app/video.py tests/test_telemetry.py
git commit -m "feat(m3): add trace spans to video decode and model inference"
```

---

### Task 6: Integrate Telemetry into `app/main.py`

**Files:**
- Modify: `app/main.py`

- [ ] **Step 1: Update `app/main.py` to init telemetry and add request metrics**

Replace the full contents of `app/main.py` with:

```python
# app/main.py
import os
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import yaml
from fastapi import FastAPI, File, Form, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from app.model import VJepa2Model, select_device
from app.schemas import (
    InferenceResponse,
    ModelMetadata,
    OutputTensor,
    Prediction,
    StreamComplete,
    StreamConfig,
    StreamPrediction,
)
from app.telemetry import get_meter, init_telemetry, instrument_fastapi
from app.video import decode_video, iter_clips

# Load config
_config_path = os.environ.get("CONFIG_PATH", "configs/model_config.yaml")
with open(_config_path) as f:
    CONFIG = yaml.safe_load(f)

_model: VJepa2Model | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model
    device = select_device(os.environ.get("DEVICE"))

    init_telemetry(
        service_name="vjepa2-server",
        device=device,
        otlp_endpoint=os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"),
    )

    meter = get_meter()
    model_load_gauge = meter.create_gauge(
        "vjepa2_model_load_duration_seconds",
        description="Time to load model at startup",
        unit="s",
    )

    model_path = os.environ.get(
        "MODEL_PATH",
        CONFIG["model"].get("model_path", CONFIG["model"]["hf_model_id"]),
    )

    load_start = time.monotonic()
    _model = VJepa2Model(
        model_path=model_path,
        device=device,
    )
    model_load_gauge.set(time.monotonic() - load_start)

    instrument_fastapi(app)

    yield
    _model = None


app = FastAPI(title="V-JEPA2 Inference", lifespan=lifespan)


@app.get("/v2/health/live")
def health_live():
    return {"status": "alive"}


@app.get("/v2/health/ready")
def health_ready():
    if _model is None:
        return JSONResponse({"error": "Model not ready"}, status_code=503)
    return {"status": "ready"}


@app.get("/v2/models/vjepa2")
def model_metadata():
    return ModelMetadata(
        name=CONFIG["model"]["name"],
        versions=["vit-l-16-ssv2"],
        platform="pytorch",
    )


@app.post("/v2/models/vjepa2/infer")
async def infer(
    file: UploadFile = File(...),
    top_k: int = Form(default=CONFIG["inference"]["default_top_k"]),
    num_frames: int = Form(default=CONFIG["inference"]["num_frames"]),
    stride: int | None = Form(default=None),
):
    if _model is None:
        return JSONResponse({"error": "Model not ready"}, status_code=503)

    meter = get_meter()
    request_duration = meter.create_histogram(
        "vjepa2_request_duration_seconds",
        description="End-to-end API call duration",
        unit="s",
    )
    requests_total = meter.create_counter(
        "vjepa2_requests_total",
        description="Total API requests",
    )
    frames_total = meter.create_counter(
        "vjepa2_frames_processed_total",
        description="Total video frames decoded",
    )

    request_start = time.monotonic()

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        if stride is not None:
            clips_results = []
            for clip in iter_clips(tmp_path, num_frames=num_frames, stride=stride):
                predictions = _model.predict(
                    clip.frames, top_k=top_k, stride=stride
                )
                frames_total.add(clip.end_frame - clip.start_frame)
                clips_results.append({
                    "clip_index": len(clips_results),
                    "start_frame": clip.start_frame,
                    "end_frame": clip.end_frame,
                    "partial": (clip.end_frame - clip.start_frame) < num_frames,
                    "predictions": [p.model_dump() for p in predictions],
                })
            requests_total.add(1, {"endpoint": "/infer", "status": "200"})
            request_duration.record(time.monotonic() - request_start)
            return {
                "model_name": CONFIG["model"]["name"],
                "model_version": "vit-l-16-ssv2",
                "id": f"req-{uuid.uuid4().hex[:8]}",
                "clips": clips_results,
            }
        else:
            frames = decode_video(tmp_path, num_frames=num_frames)
            predictions = _model.predict(frames, top_k=top_k)
            frames_total.add(num_frames)
            requests_total.add(1, {"endpoint": "/infer", "status": "200"})
            request_duration.record(time.monotonic() - request_start)
            return InferenceResponse(
                model_name=CONFIG["model"]["name"],
                model_version="vit-l-16-ssv2",
                id=f"req-{uuid.uuid4().hex[:8]}",
                outputs=[
                    OutputTensor(
                        name="predictions",
                        shape=[len(predictions)],
                        datatype="FP32",
                        data=predictions,
                    )
                ],
            )
    except ValueError as e:
        requests_total.add(1, {"endpoint": "/infer", "status": "400"})
        request_duration.record(time.monotonic() - request_start)
        return JSONResponse({"error": str(e)}, status_code=400)
    except Exception as e:
        requests_total.add(1, {"endpoint": "/infer", "status": "400"})
        request_duration.record(time.monotonic() - request_start)
        return JSONResponse(
            {"error": f"Could not decode video: {e}"}, status_code=400
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@app.websocket("/v2/models/vjepa2/stream")
async def stream(websocket: WebSocket):
    await websocket.accept()

    if _model is None:
        await websocket.send_json({"error": "Model not ready"})
        await websocket.close(code=1011)
        return

    meter = get_meter()
    active_connections = meter.create_up_down_counter(
        "vjepa2_active_connections",
        description="Current WebSocket connections",
    )
    active_connections.add(1)

    try:
        raw = await websocket.receive_json()
        config = StreamConfig(**raw)
    except Exception as e:
        active_connections.add(-1)
        await websocket.send_json({"error": f"Invalid config: {e}"})
        await websocket.close(code=1008)
        return

    source_path = Path(config.source)
    allowed_dir = Path(CONFIG["server"]["allowed_source_dir"])

    # Security: resolve and check path is within allowed directory
    try:
        resolved = source_path.resolve()
        if not str(resolved).startswith(str(allowed_dir.resolve())):
            active_connections.add(-1)
            await websocket.send_json({"error": "Source path not allowed"})
            await websocket.close(code=1008)
            return
    except Exception:
        active_connections.add(-1)
        await websocket.send_json({"error": "Source path not allowed"})
        await websocket.close(code=1008)
        return

    if not source_path.exists():
        active_connections.add(-1)
        await websocket.send_json({"error": f"File not found: {config.source}"})
        await websocket.close(code=1008)
        return

    frames_processed = 0
    try:
        for clip in iter_clips(source_path, config.num_frames, config.stride):
            predictions = _model.predict(
                clip.frames, top_k=config.top_k, stride=config.stride
            )
            msg = StreamPrediction(
                timestamp_ms=int(clip.start_frame * 1000 / 30),
                frame_range=[clip.start_frame, clip.end_frame],
                predictions=predictions,
            )
            await websocket.send_json(msg.model_dump())
            frames_processed = clip.end_frame
    except WebSocketDisconnect:
        active_connections.add(-1)
        return
    except Exception as e:
        active_connections.add(-1)
        await websocket.send_json({"error": f"Inference failed: {e}"})
        await websocket.close(code=1011)
        return

    active_connections.add(-1)
    await websocket.send_json(
        StreamComplete(status="complete", frames_processed=frames_processed).model_dump()
    )
    await websocket.close()
```

- [ ] **Step 2: Run all tests**

Run: `pytest tests/ -v`
Expected: All tests pass. The existing tests use mocked models and `init_telemetry` isn't called in tests, so the no-op fallback kicks in.

- [ ] **Step 3: Commit**

```bash
git add app/main.py
git commit -m "feat(m3): integrate telemetry into main app with request metrics"
```

---

### Task 7: OTel Collector Configuration

**Files:**
- Create: `configs/otel-collector.yaml`

- [ ] **Step 1: Create `configs/otel-collector.yaml`**

```yaml
# configs/otel-collector.yaml
receivers:
  otlp:
    protocols:
      grpc:
        endpoint: 0.0.0.0:4317

  prometheus/dcgm:
    config:
      scrape_configs:
        - job_name: dcgm-exporter
          scrape_interval: 5s
          static_configs:
            - targets: ["dcgm-exporter:9400"]

  hostmetrics:
    collection_interval: 10s
    scrapers:
      cpu: {}
      memory: {}
      process:
        include:
          match_type: strict
          names: ["python"]

processors:
  batch:
    send_batch_size: 1024
    timeout: 5s

exporters:
  prometheus:
    endpoint: 0.0.0.0:8889
    namespace: ""

  otlp/jaeger:
    endpoint: jaeger:4317
    tls:
      insecure: true

service:
  pipelines:
    metrics:
      receivers: [otlp, prometheus/dcgm, hostmetrics]
      processors: [batch]
      exporters: [prometheus]
    traces:
      receivers: [otlp]
      processors: [batch]
      exporters: [otlp/jaeger]
```

- [ ] **Step 2: Commit**

```bash
git add configs/otel-collector.yaml
git commit -m "feat(m3): add OTel Collector configuration"
```

---

### Task 8: Prometheus Configuration

**Files:**
- Create: `configs/prometheus.yaml`

- [ ] **Step 1: Create `configs/prometheus.yaml`**

```yaml
# configs/prometheus.yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: otel-collector
    static_configs:
      - targets: ["otel-collector:8889"]
```

- [ ] **Step 2: Commit**

```bash
git add configs/prometheus.yaml
git commit -m "feat(m3): add Prometheus scrape configuration"
```

---

### Task 9: Grafana Provisioning

**Files:**
- Create: `configs/grafana/datasources.yaml`
- Create: `configs/grafana/dashboards/dashboard.yaml`
- Create: `configs/grafana/dashboards/vjepa2.json`

- [ ] **Step 1: Create datasources provisioning**

```yaml
# configs/grafana/datasources.yaml
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true

  - name: Jaeger
    type: jaeger
    access: proxy
    url: http://jaeger:16686
```

- [ ] **Step 2: Create dashboard provisioning config**

```yaml
# configs/grafana/dashboards/dashboard.yaml
apiVersion: 1

providers:
  - name: default
    orgId: 1
    type: file
    disableDeletion: false
    updateIntervalSeconds: 10
    options:
      path: /var/lib/grafana/dashboards
      foldersFromFilesStructure: false
```

- [ ] **Step 3: Create the dashboard JSON**

Create `configs/grafana/dashboards/vjepa2.json`. This is a Grafana dashboard JSON with the following structure:

- **uid**: `"vjepa2-inference"`
- **title**: `"V-JEPA2 Inference"`
- **6 rows** as defined in the spec

Panel queries reference:

| Panel | PromQL Query |
|-------|-------------|
| Clip Latency p50 | `histogram_quantile(0.5, rate(vjepa2_clip_processing_seconds_bucket[5m]))` |
| Clip Latency p95 | `histogram_quantile(0.95, rate(vjepa2_clip_processing_seconds_bucket[5m]))` |
| Clip Latency p99 | `histogram_quantile(0.99, rate(vjepa2_clip_processing_seconds_bucket[5m]))` |
| Clip Throughput | `rate(vjepa2_clips_processed_total[1m])` |
| RT Violations | `rate(vjepa2_clip_realtime_violations_total[1m])` |
| CPU Util | `system_cpu_utilization` |
| Process Memory | `process_memory_rss` |
| GPU Util | `DCGM_FI_DEV_GPU_UTIL` |
| GPU Memory | `DCGM_FI_DEV_FB_USED / DCGM_FI_DEV_FB_TOTAL * 100` |
| Request Latency p50 | `histogram_quantile(0.5, rate(vjepa2_request_duration_seconds_bucket[5m]))` |
| Request Throughput | `rate(vjepa2_requests_total[1m])` |
| HTTP Errors | `rate(vjepa2_requests_total{status=~"4..\|5.."}[1m])` |
| Active Connections | `vjepa2_active_connections` |
| Model Load Time | `vjepa2_model_load_duration_seconds` |

Use `timeseries` panel type for time-series charts, `stat` panel type for stat panels, and `barchart` for the pipeline breakdown. Generate the complete Grafana JSON with all panels, proper grid positions (24-column layout), and datasource references to `"Prometheus"`.

- [ ] **Step 4: Commit**

```bash
git add configs/grafana/
git commit -m "feat(m3): add Grafana provisioning with pre-built dashboard"
```

---

### Task 10: Update `compose.yaml` with Observability Services

**Files:**
- Modify: `compose.yaml`

- [ ] **Step 1: Update `compose.yaml`**

Replace the full contents of `compose.yaml` with:

```yaml
services:
  vjepa2-server-cuda:
    profiles: [cuda]
    build:
      context: .
      dockerfile: Containerfile.cuda
    ports:
      - "8080:8080"
    volumes:
      - vjepa2-model-vitl:/model:ro
      - ./samples:/input:ro
      - ./output:/output
    environment:
      - DEVICE=cuda
      - OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4317

  vjepa2-server-cpu:
    profiles: [cpu]
    build:
      context: .
      dockerfile: Containerfile.cpu
    ports:
      - "8080:8080"
    volumes:
      - vjepa2-model-vitl:/model:ro
      - ./samples:/input:ro
      - ./output:/output
    environment:
      - DEVICE=cpu
      - OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4317

  otel-collector:
    profiles: [observability]
    image: otel/opentelemetry-collector-contrib:latest
    ports:
      - "4317:4317"
      - "8889:8889"
    volumes:
      - ./configs/otel-collector.yaml:/etc/otelcol-contrib/config.yaml:ro

  prometheus:
    profiles: [observability]
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./configs/prometheus.yaml:/etc/prometheus/prometheus.yml:ro

  jaeger:
    profiles: [observability]
    image: jaegertracing/jaeger:latest
    ports:
      - "16686:16686"
      - "4318:4317"

  grafana:
    profiles: [observability]
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_AUTH_ANONYMOUS_ENABLED=true
      - GF_AUTH_ANONYMOUS_ORG_ROLE=Admin
    volumes:
      - ./configs/grafana/datasources.yaml:/etc/grafana/provisioning/datasources/datasources.yaml:ro
      - ./configs/grafana/dashboards/dashboard.yaml:/etc/grafana/provisioning/dashboards/dashboard.yaml:ro
      - ./configs/grafana/dashboards/vjepa2.json:/var/lib/grafana/dashboards/vjepa2.json:ro

  dcgm-exporter:
    profiles: [gpu-metrics]
    image: nvidia/dcgm-exporter:latest
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]
    ports:
      - "9400:9400"

volumes:
  vjepa2-model-vitl:
    external: true
```

- [ ] **Step 2: Commit**

```bash
git add compose.yaml
git commit -m "feat(m3): add observability services to compose.yaml"
```

---

### Task 11: Update README with Observability Docs

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add observability section to README**

Add a new section after the existing usage/quickstart section:

````markdown
## Observability

The app includes an optional observability stack powered by OpenTelemetry, Prometheus, Jaeger, and Grafana.

### Quick Start

```bash
# Run with CPU inference + observability
podman compose --profile cpu --profile observability up --build

# Run with CUDA inference + observability + GPU metrics
podman compose --profile cuda --profile observability --profile gpu-metrics up --build
```

### Access Points

| Service | URL | Purpose |
|---------|-----|---------|
| App | http://localhost:8080 | V-JEPA2 inference API |
| Grafana | http://localhost:3000 | Pre-built dashboard |
| Jaeger | http://localhost:16686 | Trace visualization |
| Prometheus | http://localhost:9090 | Metrics queries |

### Dashboard

Grafana comes with a pre-provisioned "V-JEPA2 Inference" dashboard showing:

- **Clip Golden Signals** — latency, throughput, real-time violations, resource utilization
- **Pipeline Phase Breakdown** — where clip processing time is spent (decode, preprocess, inference, postprocess)
- **API Golden Signals** — request latency, throughput, HTTP errors, active connections
- **Stat Panels** — device type, model load time, current real-time ratio, uptime

### Viewing Traces

After running an inference request, open Jaeger at http://localhost:16686 and search for the `vjepa2-server` service. Each inference request shows a trace waterfall with spans for decode, preprocess, inference, and postprocess phases.

### OpenShift Deployment

The same app instrumentation works on OpenShift without code changes. The OTel Collector, Prometheus, and Jaeger are provided by:

- Cluster Observability Operator (OpenTelemetryCollector CR)
- Built-in cluster monitoring (ServiceMonitor/PodMonitor CRs)
- Red Hat distributed tracing (TempoStack or Jaeger CR)

Set `OTEL_EXPORTER_OTLP_ENDPOINT` to point to your cluster's OTel Collector.
````

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs(m3): add observability usage documentation to README"
```

---

### Task 12: Final Verification

**Files:** None (verification only)

- [ ] **Step 1: Run all unit tests**

Run: `pytest tests/ -v`
Expected: All tests pass (27 existing + 5 new = 32 total).

- [ ] **Step 2: Verify container builds**

Run: `podman build -f Containerfile.cpu -t vjepa2-server-cpu:test .`
Expected: Build succeeds. OTel packages are installed from requirements.txt.

- [ ] **Step 3: Verify compose config is valid**

Run: `podman compose --profile cpu --profile observability config`
Expected: Valid YAML output showing all services.

- [ ] **Step 4: Clean up test image**

Run: `podman rmi vjepa2-server-cpu:test`
