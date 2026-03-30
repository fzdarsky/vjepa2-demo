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
