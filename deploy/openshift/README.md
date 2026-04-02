# V-JEPA2 OpenShift Deployment

## Prerequisites

- OpenShift 4.20+ (K8s 1.33+ for image volume support)
- `oc` CLI logged in with cluster-admin privileges
- Container images pushed to quay.io:
  - `quay.io/fzdarsky/vjepa2-server-cpu:latest`
  - `quay.io/fzdarsky/vjepa2-server-cuda:latest`
  - `quay.io/fzdarsky/vjepa2-model-vitl:latest`
  - `quay.io/fzdarsky/vjepa2-model-vitg:latest`

## Step 1: Install Operators

The observability stack requires two operators. Install them if not already present:

```bash
oc apply -k deploy/openshift/operators/
```

This installs:

- **Cluster Observability Operator** (Red Hat) — provides `MonitoringStack` and `ServiceMonitor` CRDs for user-workload monitoring
- **Grafana Operator** (Community) — provides `Grafana`, `GrafanaDatasource`, and `GrafanaDashboard` CRDs

Wait for the operators to become ready:

```bash
oc get csv -n cluster-observability-operator -o custom-columns='NAME:.metadata.name,PHASE:.status.phase' | grep observ
oc get csv -n openshift-operators -o custom-columns='NAME:.metadata.name,PHASE:.status.phase' | grep grafana
```

Both should show `Succeeded`.

> **Note:** On clusters where other operators in `openshift-operators` use `Manual` install plan approval (e.g. ServiceMesh), OLM may bundle the Grafana operator's InstallPlan with those pending approvals. Check and approve if needed:
>
> ```bash
> oc get installplan -n openshift-operators
> oc patch installplan <plan-name> -n openshift-operators --type merge -p '{"spec":{"approved":true}}'
> ```

## Step 2: Deploy the Application

Choose the CPU or CUDA overlay:

### CPU

```bash
oc apply -k deploy/openshift/overlays/cpu/
```

### CUDA (GPU)

Requires the NVIDIA GPU Operator installed on the cluster.

```bash
oc apply -k deploy/openshift/overlays/cuda/
```

### Wait for Readiness

```bash
oc -n vjepa2 rollout status deployment/vjepa2-server
```

Model loading takes ~60s (CPU) or ~30s (CUDA). The readiness probe has a 60s initial delay.

## Step 3: Access the Application

Get the route URLs:

```bash
echo "App:     https://$(oc -n vjepa2 get route vjepa2-server -o jsonpath='{.spec.host}')"
echo "Grafana: https://$(oc -n vjepa2 get route vjepa2-grafana-route -o jsonpath='{.spec.host}')"
echo "Jaeger:  https://$(oc -n vjepa2 get route jaeger -o jsonpath='{.spec.host}')"
```

- **App** — web UI with upload, camera, and RTSP input modes
- **Grafana** — pre-provisioned "V-JEPA2 Inference" dashboard (default login: admin/admin)
- **Jaeger** — distributed trace viewer, search for service `vjepa2-server`

## Architecture

The deployment creates the following resources in the `vjepa2` namespace:

| Component | Kind | Purpose |
| --- | --- | --- |
| vjepa2-server | Deployment + Service + Route | Inference server with image volume for model weights |
| otel-collector | Deployment + Service | Receives OTel traces/metrics, exports to Prometheus and Jaeger |
| jaeger | Deployment + Service + Route | Standalone Jaeger AllInOne for trace storage and visualization |
| prometheus | StatefulSet (via MonitoringStack) | Scrapes OTel Collector via ServiceMonitor |
| grafana | Deployment + Route (via Grafana Operator) | Dashboards with auto-provisioned datasource |

Data flow: `App --OTLP--> OTel Collector --traces--> Jaeger`
`App --OTLP--> OTel Collector --metrics--> Prometheus <-- Grafana`

## Switching Models

The default model is ViT-L. To switch to ViT-G (larger, more accurate, requires more VRAM):

```bash
oc -n vjepa2 patch deployment vjepa2-server --type=json \
  -p '[{"op":"replace","path":"/spec/template/spec/volumes/0/image/reference","value":"quay.io/fzdarsky/vjepa2-model-vitg:latest"}]'
```

## Cleanup

```bash
oc delete -k deploy/openshift/overlays/cuda/   # or overlays/cpu/
```

## Troubleshooting

### Resource Pressure on Single-Node OpenShift

On resource-constrained SNO clusters (e.g. RHOAI demo instances), pods may stay `Pending` due to CPU allocation pressure. To free resources:

```bash
# Scale down unused model deployments
oc scale deployment <model-predictor> -n <namespace> --replicas=0

# Scale down RHOAI dashboard replicas (2 replicas use ~5000m CPU)
oc scale deployment rhods-dashboard -n redhat-ods-applications --replicas=1
```

Check current allocation with:

```bash
oc describe node | grep -A5 "Allocated resources"
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
