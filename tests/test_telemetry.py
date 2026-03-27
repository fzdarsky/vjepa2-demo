# tests/test_telemetry.py
from app.telemetry import init_telemetry, get_meter, get_tracer
from unittest.mock import patch


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
