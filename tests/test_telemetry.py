# tests/test_telemetry.py
import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider as SdkTracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from app.telemetry import init_telemetry, get_meter, get_tracer


def test_telemetry_init():
    """init_telemetry() sets up meter and tracer providers without error."""
    init_telemetry(service_name="test-service", device="cpu")
    meter = get_meter()
    tracer = get_tracer()
    assert meter is not None
    assert tracer is not None


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
