# Milestone 5 — OpenShift Deployment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deploy the V-JEPA2 inference service on OpenShift using Kustomize manifests with CPU/CUDA overlays, K8s image volumes for ModelCar model delivery, and OTel observability integrated with the Cluster Observability Operator.

**Architecture:** Plain Deployment + Service + Route with Kustomize base/overlays. KServe-compatible naming conventions (container name, ports, probes) for future OpenShift AI migration. OTel Collector as a standalone Deployment with COO ServiceMonitor for metrics scraping. Grafana Operator CRs for dashboards.

**Tech Stack:** Kustomize, OpenShift Route, K8s image volumes (KEP-4639), OTel Collector, COO ServiceMonitor (monitoring.rhobs/v1), Grafana Operator v5

**Spec:** `docs/superpowers/specs/2026-03-29-milestone-5-design.md`

---

## File Structure

| File | Responsibility |
|------|---------------|
| `deploy/openshift/base/kustomization.yaml` | NEW — Kustomize base: lists all resources |
| `deploy/openshift/base/namespace.yaml` | NEW — `vjepa2` Namespace |
| `deploy/openshift/base/deployment.yaml` | NEW — vjepa2-server Deployment with image volume, probes, env |
| `deploy/openshift/base/service.yaml` | NEW — ClusterIP Service port 8080 |
| `deploy/openshift/base/route.yaml` | NEW — Edge TLS Route with WebSocket timeout |
| `deploy/openshift/base/otel-configmap.yaml` | NEW — OTel Collector config (adapted from `configs/otel-collector.yaml`) |
| `deploy/openshift/base/otel-collector.yaml` | NEW — OTel Collector Deployment + Service |
| `deploy/openshift/base/servicemonitor.yaml` | NEW — COO ServiceMonitor scraping OTel Collector |
| `deploy/openshift/base/grafana.yaml` | NEW — Grafana CR + GrafanaDatasource + GrafanaDashboard |
| `deploy/openshift/overlays/cpu/kustomization.yaml` | NEW — CPU overlay: image, env, resources |
| `deploy/openshift/overlays/cuda/kustomization.yaml` | NEW — CUDA overlay: image, env, GPU resources, tolerations |
| `deploy/openshift/README.md` | NEW — Deployment and migration guide |

---

### Task 1: Create Namespace and Core Deployment

**Files:**
- Create: `deploy/openshift/base/namespace.yaml`
- Create: `deploy/openshift/base/deployment.yaml`

- [ ] **Step 1: Create directory structure**

```bash
mkdir -p deploy/openshift/base deploy/openshift/overlays/cpu deploy/openshift/overlays/cuda
```

- [ ] **Step 2: Create namespace.yaml**

Create `deploy/openshift/base/namespace.yaml`:

```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: vjepa2
  labels:
    app.kubernetes.io/part-of: vjepa2
```

- [ ] **Step 3: Create deployment.yaml**

Create `deploy/openshift/base/deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vjepa2-server
  namespace: vjepa2
  labels:
    app: vjepa2-server
    app.kubernetes.io/name: vjepa2-server
    app.kubernetes.io/part-of: vjepa2
    app.kubernetes.io/component: inference
spec:
  replicas: 1
  selector:
    matchLabels:
      app: vjepa2-server
  template:
    metadata:
      labels:
        app: vjepa2-server
        app.kubernetes.io/name: vjepa2-server
        app.kubernetes.io/part-of: vjepa2
        app.kubernetes.io/component: inference
    spec:
      containers:
      - name: kserve-container
        image: quay.io/fzdarsky/vjepa2-server-cpu:latest
        ports:
        - name: http
          containerPort: 8080
          protocol: TCP
        env:
        - name: MODEL_PATH
          value: /model
        - name: DEVICE
          value: cpu
        - name: OTEL_EXPORTER_OTLP_ENDPOINT
          value: http://otel-collector.vjepa2.svc:4317
        resources:
          requests:
            cpu: "2"
            memory: 8Gi
          limits:
            cpu: "4"
            memory: 12Gi
        livenessProbe:
          httpGet:
            path: /v2/health/live
            port: http
          periodSeconds: 10
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /v2/health/ready
            port: http
          initialDelaySeconds: 60
          periodSeconds: 10
          failureThreshold: 3
        volumeMounts:
        - name: model-weights
          mountPath: /model
          readOnly: true
      volumes:
      - name: model-weights
        image:
          reference: quay.io/fzdarsky/vjepa2-model-vitl:latest
          pullPolicy: IfNotPresent
```

- [ ] **Step 4: Commit**

```bash
git add deploy/openshift/base/namespace.yaml deploy/openshift/base/deployment.yaml
git commit -m "feat(m5): add namespace and deployment manifests"
```

---

### Task 2: Create Service and Route

**Files:**
- Create: `deploy/openshift/base/service.yaml`
- Create: `deploy/openshift/base/route.yaml`

- [ ] **Step 1: Create service.yaml**

Create `deploy/openshift/base/service.yaml`:

```yaml
apiVersion: v1
kind: Service
metadata:
  name: vjepa2-server
  namespace: vjepa2
  labels:
    app: vjepa2-server
    app.kubernetes.io/name: vjepa2-server
    app.kubernetes.io/part-of: vjepa2
spec:
  selector:
    app: vjepa2-server
  ports:
  - name: http
    port: 8080
    targetPort: http
    protocol: TCP
  type: ClusterIP
```

- [ ] **Step 2: Create route.yaml**

Create `deploy/openshift/base/route.yaml`:

```yaml
apiVersion: route.openshift.io/v1
kind: Route
metadata:
  name: vjepa2-server
  namespace: vjepa2
  labels:
    app: vjepa2-server
    app.kubernetes.io/name: vjepa2-server
    app.kubernetes.io/part-of: vjepa2
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
    weight: 100
```

- [ ] **Step 3: Commit**

```bash
git add deploy/openshift/base/service.yaml deploy/openshift/base/route.yaml
git commit -m "feat(m5): add service and route manifests"
```

---

### Task 3: Create OTel Collector Deployment

**Files:**
- Create: `deploy/openshift/base/otel-configmap.yaml`
- Create: `deploy/openshift/base/otel-collector.yaml`
- Reference: `configs/otel-collector.yaml`

- [ ] **Step 1: Create otel-configmap.yaml**

Adapted from `configs/otel-collector.yaml`: removed `prometheus/dcgm` receiver (no dcgm-exporter in K8s), removed `hostmetrics` receiver (not needed), removed `otlp/jaeger` exporter (Jaeger operator broken). Traces flow through `spanmetrics` connector to generate request duration metrics.

Create `deploy/openshift/base/otel-configmap.yaml`:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: otel-collector-config
  namespace: vjepa2
  labels:
    app: otel-collector
    app.kubernetes.io/name: otel-collector
    app.kubernetes.io/part-of: vjepa2
data:
  config.yaml: |
    receivers:
      otlp:
        protocols:
          grpc:
            endpoint: 0.0.0.0:4317

    connectors:
      spanmetrics:
        histogram:
          explicit:
            buckets: [10, 50, 100, 500, 1000, 2000, 5000, 10000]
        metrics_flush_interval: 5s

    processors:
      batch:
        send_batch_size: 1024
        timeout: 5s

    exporters:
      prometheus:
        endpoint: 0.0.0.0:8889
        namespace: ""

    service:
      pipelines:
        metrics:
          receivers: [otlp, spanmetrics]
          processors: [batch]
          exporters: [prometheus]
        traces:
          receivers: [otlp]
          processors: [batch]
          exporters: [spanmetrics]
```

- [ ] **Step 2: Create otel-collector.yaml**

Create `deploy/openshift/base/otel-collector.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: otel-collector
  namespace: vjepa2
  labels:
    app: otel-collector
    app.kubernetes.io/name: otel-collector
    app.kubernetes.io/part-of: vjepa2
spec:
  replicas: 1
  selector:
    matchLabels:
      app: otel-collector
  template:
    metadata:
      labels:
        app: otel-collector
        app.kubernetes.io/name: otel-collector
        app.kubernetes.io/part-of: vjepa2
    spec:
      containers:
      - name: otel-collector
        image: otel/opentelemetry-collector-contrib:latest
        args:
        - --config=/etc/otelcol-contrib/config.yaml
        ports:
        - name: otlp-grpc
          containerPort: 4317
          protocol: TCP
        - name: metrics
          containerPort: 8889
          protocol: TCP
        resources:
          requests:
            cpu: 200m
            memory: 256Mi
          limits:
            cpu: 500m
            memory: 512Mi
        volumeMounts:
        - name: config
          mountPath: /etc/otelcol-contrib/config.yaml
          subPath: config.yaml
          readOnly: true
      volumes:
      - name: config
        configMap:
          name: otel-collector-config
---
apiVersion: v1
kind: Service
metadata:
  name: otel-collector
  namespace: vjepa2
  labels:
    app: otel-collector
    app.kubernetes.io/name: otel-collector
    app.kubernetes.io/part-of: vjepa2
spec:
  selector:
    app: otel-collector
  ports:
  - name: otlp-grpc
    port: 4317
    targetPort: otlp-grpc
    protocol: TCP
  - name: metrics
    port: 8889
    targetPort: metrics
    protocol: TCP
  type: ClusterIP
```

- [ ] **Step 3: Commit**

```bash
git add deploy/openshift/base/otel-configmap.yaml deploy/openshift/base/otel-collector.yaml
git commit -m "feat(m5): add OTel Collector deployment and config"
```

---

### Task 4: Create ServiceMonitor

**Files:**
- Create: `deploy/openshift/base/servicemonitor.yaml`

- [ ] **Step 1: Create servicemonitor.yaml**

Uses the `monitoring.rhobs/v1` API group from the Cluster Observability Operator (not `monitoring.coreos.com` from built-in cluster monitoring).

Create `deploy/openshift/base/servicemonitor.yaml`:

```yaml
apiVersion: monitoring.rhobs/v1
kind: ServiceMonitor
metadata:
  name: otel-collector
  namespace: vjepa2
  labels:
    app: otel-collector
    app.kubernetes.io/name: otel-collector
    app.kubernetes.io/part-of: vjepa2
spec:
  selector:
    matchLabels:
      app: otel-collector
  endpoints:
  - port: metrics
    interval: 15s
```

- [ ] **Step 2: Commit**

```bash
git add deploy/openshift/base/servicemonitor.yaml
git commit -m "feat(m5): add ServiceMonitor for OTel Collector metrics"
```

---

### Task 5: Create Grafana Resources

**Files:**
- Create: `deploy/openshift/base/grafana.yaml`
- Reference: `configs/grafana/dashboards/vjepa2.json`

- [ ] **Step 1: Create grafana.yaml**

This file contains three Grafana Operator v5 CRs: a Grafana instance, a datasource pointing to the OTel Collector's Prometheus endpoint, and a dashboard importing the existing `vjepa2.json`.

The `GrafanaDashboard` CR uses `json` field to inline the dashboard JSON. The datasource UID in the dashboard JSON is `prometheus` — the `GrafanaDatasource` CR must match this UID.

Create `deploy/openshift/base/grafana.yaml`:

```yaml
apiVersion: grafana.integreatly.org/v1beta1
kind: Grafana
metadata:
  name: vjepa2-grafana
  namespace: vjepa2
  labels:
    app: vjepa2-grafana
    app.kubernetes.io/name: vjepa2-grafana
    app.kubernetes.io/part-of: vjepa2
    dashboards: vjepa2
spec:
  config:
    auth.anonymous:
      enabled: "true"
    security:
      admin_user: admin
      admin_password: admin
  route:
    spec:
      tls:
        termination: edge
        insecureEdgeTerminationPolicy: Redirect
---
apiVersion: grafana.integreatly.org/v1beta1
kind: GrafanaDatasource
metadata:
  name: vjepa2-prometheus
  namespace: vjepa2
  labels:
    app.kubernetes.io/part-of: vjepa2
spec:
  instanceSelector:
    matchLabels:
      dashboards: vjepa2
  datasource:
    name: Prometheus
    type: prometheus
    uid: prometheus
    access: proxy
    url: http://otel-collector.vjepa2.svc:8889
    isDefault: true
    editable: true
---
apiVersion: grafana.integreatly.org/v1beta1
kind: GrafanaDashboard
metadata:
  name: vjepa2-inference
  namespace: vjepa2
  labels:
    app.kubernetes.io/part-of: vjepa2
spec:
  instanceSelector:
    matchLabels:
      dashboards: vjepa2
  json: >
    <INLINE_DASHBOARD_JSON>
```

**Important:** Replace `<INLINE_DASHBOARD_JSON>` with the full contents of `configs/grafana/dashboards/vjepa2.json`. The JSON must be on a single line after the `>` YAML block scalar indicator, OR use `|` literal block style with proper indentation. The simplest approach: read the file and paste it inline.

When implementing this step, read `configs/grafana/dashboards/vjepa2.json` and inline its contents into the `json` field. Use the `|` block scalar for readability:

```yaml
  json: |
    {
      "uid": "vjepa2-inference",
      ...entire dashboard JSON...
    }
```

- [ ] **Step 2: Commit**

```bash
git add deploy/openshift/base/grafana.yaml
git commit -m "feat(m5): add Grafana instance, datasource, and dashboard CRs"
```

---

### Task 6: Create Kustomization Base

**Files:**
- Create: `deploy/openshift/base/kustomization.yaml`

- [ ] **Step 1: Create base kustomization.yaml**

Create `deploy/openshift/base/kustomization.yaml`:

```yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization

resources:
- namespace.yaml
- deployment.yaml
- service.yaml
- route.yaml
- otel-configmap.yaml
- otel-collector.yaml
- servicemonitor.yaml
- grafana.yaml

commonLabels:
  app.kubernetes.io/part-of: vjepa2
```

- [ ] **Step 2: Validate with kustomize build**

```bash
oc kustomize deploy/openshift/base/
```

Expected: All YAML resources rendered without errors. Verify the output contains Namespace, Deployment, Service, Route, ConfigMap, Deployment (otel), Service (otel), ServiceMonitor, Grafana, GrafanaDatasource, GrafanaDashboard.

- [ ] **Step 3: Commit**

```bash
git add deploy/openshift/base/kustomization.yaml
git commit -m "feat(m5): add base kustomization"
```

---

### Task 7: Create CPU Overlay

**Files:**
- Create: `deploy/openshift/overlays/cpu/kustomization.yaml`

- [ ] **Step 1: Create CPU overlay kustomization.yaml**

Create `deploy/openshift/overlays/cpu/kustomization.yaml`:

```yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization

resources:
- ../../base

patches:
- target:
    kind: Deployment
    name: vjepa2-server
  patch: |
    - op: replace
      path: /spec/template/spec/containers/0/image
      value: quay.io/fzdarsky/vjepa2-server-cpu:latest
    - op: replace
      path: /spec/template/spec/containers/0/env/1/value
      value: cpu
    - op: replace
      path: /spec/template/spec/containers/0/resources
      value:
        requests:
          cpu: "2"
          memory: 8Gi
        limits:
          cpu: "4"
          memory: 12Gi
```

- [ ] **Step 2: Validate with kustomize build**

```bash
oc kustomize deploy/openshift/overlays/cpu/
```

Expected: Rendered output shows `image: quay.io/fzdarsky/vjepa2-server-cpu:latest`, `DEVICE: cpu`, resource requests 2 CPU / 8Gi.

- [ ] **Step 3: Commit**

```bash
git add deploy/openshift/overlays/cpu/kustomization.yaml
git commit -m "feat(m5): add CPU overlay"
```

---

### Task 8: Create CUDA Overlay

**Files:**
- Create: `deploy/openshift/overlays/cuda/kustomization.yaml`

- [ ] **Step 1: Create CUDA overlay kustomization.yaml**

Create `deploy/openshift/overlays/cuda/kustomization.yaml`:

```yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization

resources:
- ../../base

patches:
- target:
    kind: Deployment
    name: vjepa2-server
  patch: |
    - op: replace
      path: /spec/template/spec/containers/0/image
      value: quay.io/fzdarsky/vjepa2-server-cuda:latest
    - op: replace
      path: /spec/template/spec/containers/0/env/1/value
      value: cuda
    - op: replace
      path: /spec/template/spec/containers/0/resources
      value:
        requests:
          cpu: "4"
          memory: 16Gi
        limits:
          cpu: "4"
          memory: 16Gi
          nvidia.com/gpu: "1"
    - op: add
      path: /spec/template/spec/tolerations
      value:
      - key: nvidia.com/gpu
        operator: Exists
        effect: NoSchedule
    - op: add
      path: /spec/template/spec/nodeSelector
      value:
        feature.node.kubernetes.io/pci-10de.present: "true"
    - op: replace
      path: /spec/template/spec/volumes/0/image/reference
      value: quay.io/fzdarsky/vjepa2-model-vitg:latest
```

Note: The CUDA overlay also switches to the ViT-G model (`vjepa2-model-vitg`) since GPU clusters have enough VRAM for the larger model. This can be overridden by the user.

- [ ] **Step 2: Validate with kustomize build**

```bash
oc kustomize deploy/openshift/overlays/cuda/
```

Expected: Rendered output shows `image: quay.io/fzdarsky/vjepa2-server-cuda:latest`, `DEVICE: cuda`, `nvidia.com/gpu: "1"` in limits, tolerations for `nvidia.com/gpu`, nodeSelector for NVIDIA PCI ID, model reference `vjepa2-model-vitg`.

- [ ] **Step 3: Commit**

```bash
git add deploy/openshift/overlays/cuda/kustomization.yaml
git commit -m "feat(m5): add CUDA overlay with GPU resources and tolerations"
```

---

### Task 9: Write Deployment Guide

**Files:**
- Create: `deploy/openshift/README.md`

- [ ] **Step 1: Create README.md**

Create `deploy/openshift/README.md`:

```markdown
# V-JEPA2 OpenShift Deployment

## Prerequisites

- OpenShift 4.20+ (K8s 1.33+ for image volume support)
- `oc` CLI logged in with cluster-admin
- Cluster Observability Operator installed (for ServiceMonitor)
- Grafana Operator installed (for dashboard)
- Container images pushed to quay.io:
  - `quay.io/fzdarsky/vjepa2-server-cpu:latest`
  - `quay.io/fzdarsky/vjepa2-server-cuda:latest`
  - `quay.io/fzdarsky/vjepa2-model-vitl:latest`
  - `quay.io/fzdarsky/vjepa2-model-vitg:latest`

## Deploy (CPU)

```bash
oc apply -k deploy/openshift/overlays/cpu/
```

Wait for the pod to become ready (model loading takes ~60s):

```bash
oc -n vjepa2 rollout status deployment/vjepa2-server
```

Get the route URL:

```bash
oc -n vjepa2 get route vjepa2-server -o jsonpath='{.spec.host}'
```

Open `https://<route-host>/` in a browser to access the web UI.

## Deploy (CUDA)

Requires NVIDIA GPU Operator installed on the cluster.

```bash
oc apply -k deploy/openshift/overlays/cuda/
```

## Switching Models

The default model is ViT-L (CPU overlay) or ViT-G (CUDA overlay). To change the model, patch the image volume reference:

```bash
oc -n vjepa2 patch deployment vjepa2-server --type=json \
  -p '[{"op":"replace","path":"/spec/template/spec/volumes/0/image/reference","value":"quay.io/fzdarsky/vjepa2-model-vitg:latest"}]'
```

## Observability

- **Metrics:** OTel Collector exposes Prometheus metrics on port 8889, scraped by ServiceMonitor
- **Grafana:** Access via `oc -n vjepa2 get route vjepa2-grafana -o jsonpath='{.spec.host}'`
- **Dashboard:** "V-JEPA2 Inference" dashboard auto-provisioned

## Cleanup

```bash
oc delete -k deploy/openshift/overlays/cpu/
```

## Future: OpenShift AI Migration

The deployment uses KServe-compatible conventions:
- Container named `kserve-container`
- Port 8080
- V2 Inference Protocol health probes

To migrate to KServe:
1. Create a `ServingRuntime` CR from the Deployment spec
2. Create an `InferenceService` CR for `/v2/models/vjepa2/infer`
3. Keep a separate Deployment for WebSocket streaming + web UI
```

- [ ] **Step 2: Commit**

```bash
git add deploy/openshift/README.md
git commit -m "docs(m5): add OpenShift deployment guide"
```

---

### Task 10: Deploy and Validate on SNO

**Files:** None (cluster operations only)

- [ ] **Step 1: Log in to the cluster**

```bash
oc whoami
```

Expected: Shows your username. If `Unauthorized`, run `oc login`.

- [ ] **Step 2: Apply CPU overlay**

```bash
oc apply -k deploy/openshift/overlays/cpu/
```

Expected: namespace, deployment, service, route, configmap, otel-collector deployment, otel-collector service, servicemonitor, grafana, grafanadatasource, grafanadashboard created.

- [ ] **Step 3: Wait for pods to start**

```bash
oc -n vjepa2 get pods -w
```

Expected: `vjepa2-server-*` and `otel-collector-*` pods reach `Running` status. The vjepa2-server may take ~60-120s to become Ready (model loading).

- [ ] **Step 4: Verify readiness**

```bash
ROUTE=$(oc -n vjepa2 get route vjepa2-server -o jsonpath='{.spec.host}')
curl -sk "https://${ROUTE}/v2/health/ready"
```

Expected: `{"status":"ready","device":"cpu"}` (or similar JSON with device info).

- [ ] **Step 5: Verify image volume**

```bash
oc -n vjepa2 describe pod -l app=vjepa2-server | grep -A5 "model-weights"
```

Expected: Shows the image volume with reference `quay.io/fzdarsky/vjepa2-model-vitl:latest`.

- [ ] **Step 6: Access web UI**

Open `https://<ROUTE>/` in a browser. Verify the Upload tab loads and the status shows the model as ready.

- [ ] **Step 7: Verify Grafana**

```bash
GRAFANA_ROUTE=$(oc -n vjepa2 get route vjepa2-grafana-route -o jsonpath='{.spec.host}' 2>/dev/null || oc -n vjepa2 get route -l app=vjepa2-grafana -o jsonpath='{.items[0].spec.host}')
echo "https://${GRAFANA_ROUTE}"
```

Open in browser. Navigate to Dashboards → "V-JEPA2 Inference". Verify panels load (they'll show "No data" until inference requests are made).

- [ ] **Step 8: End-to-end test**

Upload a sample video via the web UI or curl:

```bash
curl -sk -X POST "https://${ROUTE}/v2/models/vjepa2/infer" \
  -F "file=@samples/example.mp4" \
  -F "top_k=3"
```

Then check Grafana — `vjepa2_requests_total` and `vjepa2_clips_processed_total` should increment.

---

## Verification Summary

| Check | Command | Expected |
|-------|---------|----------|
| Pods running | `oc -n vjepa2 get pods` | 2 pods Running (server + otel) |
| Readiness | `curl -sk https://<route>/v2/health/ready` | `{"status":"ready",...}` |
| Image volume | `oc describe pod -l app=vjepa2-server` | model-weights from OCI ref |
| Web UI | Browser → `https://<route>/` | Upload/Camera/RTSP tabs |
| WebSocket | Camera tab → Start | Live streaming works |
| Grafana | Browser → Grafana route | Dashboard loads |
| Metrics flow | Grafana after inference | Counters increment |
| Kustomize CPU | `oc kustomize overlays/cpu/` | Valid YAML, cpu image |
| Kustomize CUDA | `oc kustomize overlays/cuda/` | Valid YAML, cuda image, GPU resources |
