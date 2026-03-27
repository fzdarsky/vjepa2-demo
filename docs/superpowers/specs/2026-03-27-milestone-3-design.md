# Milestone 3 — Observability & Metrics

## Overview

Add observability to the V-JEPA2 inference service using OpenTelemetry for metrics and distributed tracing. The instrumentation works in two deployment modes: locally with a bundled Grafana/Prometheus/Jaeger stack via Podman Compose, and on OpenShift where the platform provides the observability backend. A pre-built Grafana dashboard delivers a turnkey demo experience.

**Primary audience**: Demo/showcase — "this app has proper observability" with a polished dashboard out of the box.

**Milestone 3 scope**: Instrument the FastAPI app with OTel SDK (metrics + traces), add OTel Collector as the central telemetry hub, bundle Prometheus/Jaeger/Grafana behind a Compose profile, provision a Grafana dashboard, and include GPU/CPU/memory utilization metrics.

## Requirements

### Functional

- App emits metrics and trace spans via OpenTelemetry SDK over OTLP to an OTel Collector
- OTel Collector fans out to Prometheus (metrics) and Jaeger (traces)
- Pre-built Grafana dashboard with clip-level and API-level golden signals, resource utilization, and pipeline phase breakdown
- GPU metrics (compute utilization, VRAM) via dcgm-exporter when running the CUDA variant
- CPU and memory metrics via OTel Collector's hostmetrics receiver
- Observability stack is optional, activated via `--profile observability` in Podman Compose
- Trace spans for each inference pipeline phase: decode, preprocess, inference, postprocess

### Non-Functional

- Existing 27 unit tests continue to pass unchanged
- Telemetry gracefully degrades to no-op when OTel packages are not installed (test/dev environments)
- No measurable impact on inference latency from instrumentation
- Same app instrumentation works on OpenShift without code changes

### Out of Scope (Milestone 3)

- Stream sources — RTSP, browser (milestone 4)
- Alerting rules or PagerDuty/Slack integrations
- Custom OpenShift operator CRs (documented, not automated)
- Log aggregation / structured logging

## Architecture

### Data Flow

```text
┌─────────────────────────┐     ┌─────────────────────────┐
│  vjepa2-server          │     │  dcgm-exporter          │
│  (FastAPI + OTel SDK)   │     │  (CUDA profile only)    │
│                         │     │                         │
│  metrics + traces       │     │  GPU util, VRAM, temp   │
│  via OTLP gRPC :4317   │     │  via Prometheus :9400   │
└───────────┬─────────────┘     └───────────┬─────────────┘
            │                               │
            │         OTLP gRPC             │   Prometheus scrape
            ▼                               ▼
┌─────────────────────────────────────────────────────────┐
│  OTel Collector                                         │
│                                                         │
│  Receivers: OTLP (app), Prometheus (dcgm), hostmetrics  │
│  Exporters: Prometheus (:8889), OTLP→Jaeger (:4318)     │
└──────────┬──────────────────────────────┬───────────────┘
           │                              │
           ▼                              ▼
┌──────────────────┐          ┌──────────────────┐
│  Prometheus      │          │  Jaeger          │
│  :9090           │          │  :16686 (UI)     │
│  Scrapes :8889   │          │  :4318 (OTLP)    │
└────────┬─────────┘          └────────┬─────────┘
         │                             │
         └──────────┬──────────────────┘
                    ▼
          ┌──────────────────┐
          │  Grafana         │
          │  :3000           │
          │  Datasources:    │
          │   - Prometheus   │
          │   - Jaeger       │
          └──────────────────┘
```

### OpenShift Deployment

The same OTel SDK instrumentation works unchanged on OpenShift. The infrastructure differences:

- **OTel Collector** → provided by Cluster Observability Operator (OpenTelemetryCollector CR)
- **Prometheus** → built-in cluster monitoring stack (ServiceMonitor/PodMonitor CRs)
- **Jaeger** → Red Hat distributed tracing (TempoStack or Jaeger CR)
- **Grafana** → cluster Grafana or user-provisioned

No app code changes — only deployment/config differences.

## App Instrumentation

### Module: `app/telemetry.py`

New module that owns all OTel setup. Called from `lifespan()` in `main.py`.

Responsibilities:
- Initialize `MeterProvider` with OTLP exporter
- Initialize `TracerProvider` with OTLP span exporter
- Set resource attributes (service name, version, device type)
- Export a `meter` and `tracer` for use by other modules
- Graceful no-op fallback: if `opentelemetry` packages are not installed, provide stub meter/tracer that do nothing. No crashes, no conditional imports scattered across the codebase.

### Metrics

Metrics are organized around the "clip" as the primary unit of work (one model forward pass on N frames), consistent across REST single-video, REST with stride, WebSocket streaming, and future RTSP streams.

#### Clip-Level Metrics (Primary)

| Metric | Type | Description |
|--------|------|-------------|
| `vjepa2_clip_processing_seconds` | Histogram | Wall time to process one clip (decode + preprocess + inference + postprocess) |
| `vjepa2_clips_processed_total` | Counter | Total clips processed |
| `vjepa2_clip_realtime_violations_total` | Counter | Clips where processing time exceeded `stride / source_fps` (realtime ratio > 1.0) |

#### API-Level Metrics (Secondary)

| Metric | Type | Description |
|--------|------|-------------|
| `vjepa2_request_duration_seconds` | Histogram | End-to-end API call duration |
| `vjepa2_requests_total` | Counter | Total API requests by endpoint and status code |
| `vjepa2_active_connections` | UpDownCounter | Current WebSocket connections |
| `vjepa2_frames_processed_total` | Counter | Total video frames decoded |

#### System/Infra Metrics (External)

| Metric | Source | Description |
|--------|--------|-------------|
| GPU compute utilization % | dcgm-exporter | GPU busy percentage |
| GPU memory used / total | dcgm-exporter | VRAM usage |
| CPU utilization % | OTel hostmetrics receiver | Host CPU usage |
| Process RSS memory | OTel hostmetrics receiver | App memory footprint |

#### One-Time Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `vjepa2_model_load_duration_seconds` | Gauge | Time to load model at startup |

### Trace Spans

Each clip inference produces a parent span with child spans for the pipeline phases:

```text
clip_inference (parent)
├── decode_video      — video decoding / frame extraction
├── preprocess        — processor transforms (resize, normalize)
├── inference         — model forward pass (GPU or CPU)
└── postprocess       — softmax, top-k selection
```

Span durations reveal which pipeline phase is the bottleneck — useful for determining whether the workload is GPU compute bound, CPU bound, or memory bound.

### Code Changes

| File | Change |
|------|--------|
| `app/telemetry.py` | NEW — OTel setup, meter/tracer exports, no-op fallback |
| `app/main.py` | MODIFIED — call `init_telemetry()` in lifespan, request-level metrics via middleware |
| `app/model.py` | MODIFIED — trace spans around inference, clip processing histogram, RT violation counter |
| `app/video.py` | MODIFIED — trace span around `decode_video` |

## Compose Services

The observability stack is activated via `--profile observability`, keeping it optional.

### New Services

| Service | Profile(s) | Image | Ports | Purpose |
|---------|------------|-------|-------|---------|
| `otel-collector` | observability | `otel/opentelemetry-collector-contrib` | 4317, 8889 | Central telemetry hub |
| `prometheus` | observability | `prom/prometheus` | 9090 | Metrics storage |
| `jaeger` | observability | `jaegertracing/jaeger` | 16686, 4318 | Trace storage + UI |
| `grafana` | observability | `grafana/grafana` | 3000 | Dashboard |
| `dcgm-exporter` | gpu-metrics | `nvidia/dcgm-exporter` | 9400 | GPU metrics |

`dcgm-exporter` uses its own `gpu-metrics` profile since Compose activates a service when *any* of its listed profiles match (OR logic, not AND). Users opt in explicitly: `--profile cuda --profile observability --profile gpu-metrics`. The OTel Collector config scrapes dcgm-exporter's endpoint unconditionally — if dcgm-exporter isn't running, the scrape fails silently and GPU panels in Grafana show "no data".

### Usage

```bash
# App only (unchanged from milestone 2)
podman compose --profile cpu up

# App + full observability stack
podman compose --profile cuda --profile observability up

# App + observability + GPU metrics
podman compose --profile cuda --profile observability --profile gpu-metrics up

# Access points:
#   App:        http://localhost:8080
#   Grafana:    http://localhost:3000
#   Jaeger UI:  http://localhost:16686
#   Prometheus: http://localhost:9090
```

### OTel Collector Configuration

The Collector config (`configs/otel-collector.yaml`) defines:

**Receivers:**
- `otlp` — gRPC on :4317 for app metrics + traces
- `prometheus` — scrapes dcgm-exporter :9400/metrics (when available)
- `hostmetrics` — CPU and memory utilization from the host

**Processors:**
- `batch` — batches telemetry for efficient export

**Exporters:**
- `prometheus` — exposes metrics on :8889 for Prometheus to scrape
- `otlp/jaeger` — sends traces to Jaeger on :4318

**Pipelines:**
- `metrics` — otlp + prometheus + hostmetrics → batch → prometheus exporter
- `traces` — otlp → batch → otlp/jaeger exporter

## Grafana Dashboard

Pre-provisioned via Grafana's file-based provisioning (no manual setup). The dashboard has 6 rows:

### Row 1-2: Clip Golden Signals (2×2 grid)

| Panel | Signal | Content |
|-------|--------|---------|
| Top-left | Latency | `vjepa2_clip_processing_seconds` histogram quantiles (p50/p95/p99) |
| Top-right | Throughput | `rate(vjepa2_clips_processed_total)` — clips/sec |
| Bottom-left | Errors | `rate(vjepa2_clip_realtime_violations_total)` — RT violations/sec |
| Bottom-right | Utilization | Split sub-panel: CPU % + RSS (hostmetrics) and GPU % + VRAM (dcgm) |

### Row 3: Pipeline Phase Breakdown

Stacked bar or stacked area chart showing average duration by phase (decode, preprocess, inference, postprocess), derived from trace span durations. Reveals the bottleneck phase at a glance.

### Row 4-5: API Golden Signals (2×2 grid)

| Panel | Signal | Content |
|-------|--------|---------|
| Top-left | Latency | `vjepa2_request_duration_seconds` histogram quantiles (p50/p95/p99) |
| Top-right | Throughput | `rate(vjepa2_requests_total)` by endpoint — requests/sec |
| Bottom-left | Errors | `rate(vjepa2_requests_total{status=~"4..\|5.."})` — HTTP error rate |
| Bottom-right | Saturation | `vjepa2_active_connections` — WebSocket gauge |

### Row 6: Stat Panels

| Panel | Source |
|-------|--------|
| Device | Resource attribute (CUDA/CPU/MPS) |
| Model Load Time | `vjepa2_model_load_duration_seconds` |
| Current RT Ratio | Last `vjepa2_clip_processing_seconds` / (stride/fps) |
| Uptime | Process start time |

### Provisioning Files

```text
configs/
├── otel-collector.yaml        # Collector pipeline config
├── prometheus.yaml            # Prometheus scrape config (targets: otel-collector:8889)
└── grafana/
    ├── datasources.yaml       # Auto-provision Prometheus + Jaeger datasources
    └── dashboards/
        ├── dashboard.yaml     # Grafana dashboard provisioning config
        └── vjepa2.json        # Dashboard JSON
```

## Testing

### Unit Tests (New)

| Test | What it verifies |
|------|-----------------|
| `test_telemetry_init` | `init_telemetry()` sets up meter + tracer providers without error |
| `test_telemetry_noop` | When OTel deps are missing, telemetry is no-op (no crashes) |
| `test_metrics_recorded` | After an inference call, clip processing histogram has at least one observation |
| `test_realtime_violation_counted` | When clip processing exceeds stride/fps threshold, violation counter increments |
| `test_trace_spans_created` | An inference request produces parent + child spans (decode, preprocess, inference, postprocess) |

Tests use OTel's `InMemoryMetricReader` and `InMemorySpanExporter` — no Prometheus/Jaeger needed.

### Existing Tests

All 27 existing unit tests pass unchanged. The no-op fallback in `app/telemetry.py` ensures telemetry code paths are inert when OTel packages are not installed.

### Integration Testing (Manual)

Documented in README:
1. `podman compose --profile cpu --profile observability up`
2. `curl -F file=@samples/video.mp4 http://localhost:8080/v2/models/vjepa2/infer`
3. Open Grafana at :3000 — verify dashboard shows metrics
4. Open Jaeger at :16686 — verify trace spans appear

## Dependencies

### New Python Packages (requirements.txt)

```text
opentelemetry-api>=1.20.0
opentelemetry-sdk>=1.20.0
opentelemetry-exporter-otlp>=1.20.0
opentelemetry-instrumentation-fastapi>=0.41b0
```

### Container Images (Compose only)

- `otel/opentelemetry-collector-contrib`
- `prom/prometheus`
- `jaegertracing/jaeger`
- `grafana/grafana`
- `nvidia/dcgm-exporter` (CUDA profile)

## Project Structure Changes

```text
jepa-demo/
├── app/
│   ├── telemetry.py              # NEW
│   ├── main.py                   # MODIFIED
│   ├── model.py                  # MODIFIED
│   ├── video.py                  # MODIFIED
│   └── ...                       # UNCHANGED
├── configs/
│   ├── model_config.yaml         # UNCHANGED
│   ├── otel-collector.yaml       # NEW
│   ├── prometheus.yaml           # NEW
│   └── grafana/
│       ├── datasources.yaml      # NEW
│       └── dashboards/
│           ├── dashboard.yaml    # NEW
│           └── vjepa2.json       # NEW
├── compose.yaml                  # MODIFIED
├── requirements.txt              # MODIFIED
├── tests/
│   ├── test_telemetry.py         # NEW
│   └── ...                       # UNCHANGED
├── README.md                     # MODIFIED
└── docs/
    └── superpowers/
        └── specs/
            └── 2026-03-27-milestone-3-design.md
```

## Error Handling

| Scenario | Behavior |
|----------|----------|
| OTel packages not installed | `app/telemetry.py` provides no-op meter/tracer. App runs without telemetry. |
| OTel Collector unreachable | OTel SDK buffers and retries. App continues serving — metrics/traces are dropped silently after buffer fills. |
| dcgm-exporter not running (CPU profile) | Collector logs scrape failure. GPU panels in Grafana show "no data". |
| Prometheus/Jaeger down | Collector buffers briefly, then drops. Grafana shows stale or missing data. |
| Grafana provisioning fails | Dashboard not available. App and metrics pipeline unaffected. |

## Future Milestones (Context Only)

### Milestone 4 — Stream Sources

- RTSP streams from cameras (UniFi IP cameras / Protect NVR)
- Browser camera streams via WebTransport or WebRTC
- Browser frontend for video + inference results side-by-side
- `vjepa2_stream_lag_seconds` metric (cumulative real-time drift during live streaming)
