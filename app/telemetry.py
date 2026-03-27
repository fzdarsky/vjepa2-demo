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
