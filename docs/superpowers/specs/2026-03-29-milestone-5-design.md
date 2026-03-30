# Milestone 5: OpenShift Deployment

## Context

The V-JEPA2 inference demo (Milestones 1-4) runs locally via Podman Compose with CPU/CUDA container variants, ModelCar OCI model images, and full OTel observability. The next step is deploying on OpenShift to validate the deployment model before moving to GPU-accelerated clusters and OpenShift AI integration.

**Goals:**
1. Deploy the service on OpenShift with plain K8s resources (Deployment, Service, Route)
2. Support both CPU and NVIDIA GPU inference via Kustomize overlays
3. Integrate observability with the cluster's Cluster Observability Operator
4. Design for easy future migration to OpenShift AI / KServe

**Target cluster:** OpenShift 4.20.13 (K8s 1.33), Single-Node, x86_64, 7.5 CPU cores, 30GB RAM, no GPU. Cluster Observability Operator v1.4.0, Grafana Operator v5.22.2, cert-manager v1.18.1 installed.

## Design Decisions

### Why Not KServe Now

KServe was evaluated but rejected for this milestone:

1. **WebSocket incompatibility:** The app's WebSocket streaming (browser camera, RTSP) and MJPEG preview endpoints are long-lived connections. KServe routes through Knative, which treats these as stuck requests and triggers unwanted autoscaling behavior.
2. **Image volume conflict:** K8s image volumes (KEP-4639) are a Pod-level primitive. KServe's `storageUri` is a competing, higher-level model delivery abstraction. Using both creates friction.
3. **Overhead on SNO:** The RHOAI operator stack is heavy for a 7.5-core single-node test cluster.

**Migration path preserved:** By using KServe-compatible conventions now (container named `kserve-container`, port 8080, V2 Inference Protocol probes), the future migration is mechanical — extract a `ServingRuntime` CR from the Deployment spec.

### Model Delivery: K8s Image Volumes

Uses KEP-4639 `image` volume source (beta in K8s 1.33, confirmed working on the target cluster via dry-run) to mount ModelCar OCI images directly:

```yaml
volumes:
- name: model-weights
  image:
    reference: quay.io/fzdarsky/vjepa2-model-vitl:latest
    pullPolicy: IfNotPresent
```

No init containers, no PVCs, no HuggingFace downloads. The existing ModelCar images (`quay.io/fzdarsky/vjepa2-model-vitl`, `vjepa2-model-vitg`) are used directly.

### Observability: OTel Collector + COO ServiceMonitor

COO v1.4.0 provides `MonitoringStack` and `ServiceMonitor` (via `monitoring.rhobs` API group) but no OpenTelemetry or Tempo CRDs. Rather than installing additional operators:

- Deploy OTel Collector as a plain K8s Deployment (adapted from existing `configs/otel-collector.yaml`)
- COO `ServiceMonitor` scrapes the collector's Prometheus exporter port (8889)
- `spanmetrics` connector converts traces to request duration histograms (compensates for broken Jaeger operator)
- Grafana Operator CRs for dashboard and datasource

## Architecture

### Manifest Structure

```
deploy/openshift/
  base/
    kustomization.yaml
    namespace.yaml              # vjepa2 namespace
    deployment.yaml             # Pod with image volume, KServe-compatible naming
    service.yaml                # ClusterIP port 8080
    route.yaml                  # Edge TLS, WebSocket timeout 3600s
    otel-configmap.yaml         # OTel Collector configuration
    otel-collector.yaml         # OTel Collector Deployment + Service
    servicemonitor.yaml         # Scrape OTel Collector's Prometheus port
    grafana.yaml                # Grafana instance + datasource + dashboard CRs
  overlays/
    cpu/
      kustomization.yaml        # Patches: image, DEVICE=cpu, resources
    cuda/
      kustomization.yaml        # Patches: image, DEVICE=cuda, GPU resources + tolerations
```

### Deployment

| Property | Value |
|----------|-------|
| Namespace | `vjepa2` |
| Container name | `kserve-container` |
| Port | 8080 |
| Image (base) | `quay.io/fzdarsky/vjepa2-server-cpu:latest` |
| Liveness | `httpGet /v2/health/live` (period 10s) |
| Readiness | `httpGet /v2/health/ready` (initialDelaySeconds 60, period 10s) |
| Model volume | `image` source → `/model` (read-only) |
| Env: MODEL_PATH | `/model` |
| Env: DEVICE | `cpu` (base), overridden by overlay |
| Env: OTEL_EXPORTER_OTLP_ENDPOINT | `http://otel-collector.vjepa2.svc:4317` |

### Route

Edge TLS with extended timeout for WebSocket streaming:

```yaml
apiVersion: route.openshift.io/v1
kind: Route
metadata:
  name: vjepa2-server
  annotations:
    haproxy.router.openshift.io/timeout: 3600s
spec:
  tls:
    termination: edge
    insecureEdgeTerminationPolicy: Redirect
  port:
    targetPort: http
  to:
    kind: Service
    name: vjepa2-server
```

The existing `static/app.js` derives WebSocket URLs from `window.location`, so it auto-detects `wss://` when served over HTTPS.

### CPU Overlay

- Image: `quay.io/fzdarsky/vjepa2-server-cpu:latest`
- `DEVICE=cpu`
- Resources: requests 2 CPU / 8Gi; limits 4 CPU / 12Gi
- ViT-L model (~1.4GB weights) + PyTorch overhead fits comfortably on SNO

### CUDA Overlay

- Image: `quay.io/fzdarsky/vjepa2-server-cuda:latest`
- `DEVICE=cuda`
- Resources: requests 4 CPU / 16Gi; limits `nvidia.com/gpu: 1`
- Tolerations: `nvidia.com/gpu` key (NoSchedule)
- NodeSelector: `feature.node.kubernetes.io/pci-10de.present: "true"` (NVIDIA PCI vendor ID, set by NFD from GPU Operator)
- Requires NVIDIA GPU Operator on target cluster

### OTel Collector

Adapted from `configs/otel-collector.yaml`:

- **Receivers:** OTLP gRPC on 4317 (removed `prometheus/dcgm` and `hostmetrics` — not applicable in K8s)
- **Connectors:** `spanmetrics` for trace-to-metric conversion
- **Exporters:** Prometheus on 8889 (removed `otlp/jaeger` — Jaeger operator broken)
- **Pipelines:**
  - metrics: otlp + spanmetrics → batch → prometheus
  - traces: otlp → batch → spanmetrics

### ServiceMonitor

```yaml
apiVersion: monitoring.rhobs/v1
kind: ServiceMonitor
metadata:
  name: otel-collector
spec:
  selector:
    matchLabels:
      app: otel-collector
  endpoints:
  - port: metrics
    interval: 15s
```

### Grafana

Via Grafana Operator v5.22.2:
- `Grafana` CR: new instance in `vjepa2` namespace
- `GrafanaDatasource` CR: pointing to COO-managed Prometheus (or Thanos Querier)
- `GrafanaDashboard` CR: importing existing `configs/grafana/dashboards/vjepa2.json`

## KServe Migration Path (Future)

When ready for OpenShift AI on a GPU cluster:

1. **Extract ServingRuntime** from the Deployment spec — same container image, ports, probes, env vars
2. **Create InferenceService** CR for `/v2/models/vjepa2/infer`
3. **Keep separate Deployment** for WebSocket streaming + web UI (CPU-only)
4. **Model delivery:** Evaluate KServe's OCI storage initializer vs. continuing with image volumes
5. No app code changes — V2 Inference Protocol already implemented

## Resource Budget (SNO)

| Component | CPU | Memory |
|-----------|-----|--------|
| vjepa2-server | 2 req / 4 limit | 8Gi req / 12Gi limit |
| OTel Collector | 0.2 req | 256Mi |
| Grafana | 0.25 req | 256Mi |
| Platform overhead | ~4.5 | ~20Gi |
| **Total** | ~7 | ~29Gi |
