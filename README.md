# V-JEPA2 Video Inference Demo

Video action recognition using Meta's [V-JEPA2](https://github.com/facebookresearch/jepa) model. Upload videos, record from your browser camera, or connect an RTSP camera and get action predictions from 174 classes (Something-Something v2).

## Deploying with podman-compose

### Prerequisites

- [Podman](https://podman.io/) (on macOS: `podman machine init && podman machine start`)
- `podman-compose` (`pip install podman-compose`)
- Red Hat registry access for building (`podman login registry.redhat.io`)

### Start the Stack

```bash
# Create a model volume from the ModelCar image (one-time)
podman volume create --driver image \
  --opt image=quay.io/fzdarsky/vjepa2-model-vitl:latest \
  vjepa2-model-vitl

# Build the server image (one-time, or after code changes)
podman build -f Containerfile.cpu -t vjepa2-server-cpu .

# Start the server with observability
podman-compose --profile cpu --profile observability up -d
```

For NVIDIA GPU acceleration, use `Containerfile.cuda` and `--profile cuda` instead.

### Native Server with Containerized Observability

On macOS, run the server natively for Apple Silicon MPS acceleration while keeping the observability stack in containers:

```bash
# Start only the observability stack
podman-compose --profile observability up -d

# Set up Python environment (one-time)
python3.12 -m venv .venv && source .venv/bin/activate
pip install torch torchvision && pip install -r requirements.txt

# Start the server natively (auto-detects MPS on Apple Silicon)
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317 \
MODEL_PATH=./model \
python -m app serve
```

The native server sends telemetry to the containerized OTel Collector via `localhost:4317`. Grafana, Jaeger, and Prometheus work the same way.

### Access Points

| Service | URL | Purpose |
|---------|-----|---------|
| Web UI | https://localhost:8443 | Video inference (upload, camera, RTSP) |
| Grafana | http://localhost:3000 | Performance dashboard |
| Jaeger | http://localhost:16686 | Trace visualization |
| Prometheus | http://localhost:9090 | Metrics queries |

## Deploying on OpenShift

Kustomize manifests deploy the full stack (inference server, OTel Collector, Jaeger, Grafana) on OpenShift 4.20+ with CPU or CUDA overlays. Model weights are delivered via K8s image volumes (KEP-4639) — no PVCs or init containers needed.

```bash
# Install required operators (one-time)
oc apply -k deploy/openshift/operators/

# Deploy with CUDA (or overlays/cpu/ for CPU-only)
oc apply -k deploy/openshift/overlays/cuda/
```

See [deploy/openshift/README.md](deploy/openshift/README.md) for the full deployment guide including prerequisites, troubleshooting, and architecture details.

## Building Model Images

The model weights are distributed as OCI "ModelCar" images — container images that hold only model files, mounted as volumes at runtime. Pre-built images are available at `quay.io/fzdarsky/`, but you can rebuild them with `scripts/build-model-images.sh`.

```bash
# Build and push both models (monolithic, single safetensors layer)
./scripts/build-model-images.sh all

# Build only ViT-L, don't push
./scripts/build-model-images.sh --no-push vitl

# Shard safetensors into 4 layers for parallel registry pulls
./scripts/build-model-images.sh --shard vitl

# Shard into 8 layers, custom registry
./scripts/build-model-images.sh --shard 8 --registry quay.io/myorg vitg
```

**Options:**

| Flag             | Default              | Description                                              |
| ---------------- | -------------------- | -------------------------------------------------------- |
| `--shard [N]`    | off (4 if enabled)   | Split safetensors into N OCI layers for concurrent pulls |
| `--registry URL` | `quay.io/fzdarsky`   | Target container registry                                |
| `--work-dir DIR` | `~/jepa-model-build` | Working directory for downloads and builds               |
| `--no-push`      | push enabled         | Build only, skip pushing to registry                     |

**Sharding** splits the single `model.safetensors` into multiple files (e.g. `model-00001-of-00004.safetensors`) plus a `model.safetensors.index.json`. Each shard becomes a separate OCI layer, so container runtimes pull them concurrently. The index ensures the sharded model remains loadable by HuggingFace `transformers` and vLLM.

**Requirements:** Python 3.12+ with `torch`, `safetensors`, and `huggingface_hub`; Podman with registry credentials (`podman login`).

## Using the Web UI

Open https://localhost:8443. Three input modes are available:

- **Upload** — select or drag-and-drop a video file for batch inference
- **Camera** — record from your browser camera with inference running while you record
- **RTSP** — connect to an RTSP stream (e.g. `rtsp://192.168.1.x/...`) for continuous inference

The sidebar lets you tune inference parameters:

- **top_k** — number of predictions per clip (default: 3)
- **stride** — how many frames to advance between clips (default: 16, i.e. non-overlapping). Lower values produce overlapping clips for smoother results at the cost of more computation.

Results stream in as clips are processed. Each result shows a thumbnail and top-k action predictions with confidence scores. Camera mode shows a recording timer and progress indicator.

After a session completes, click **Download** to get an annotated MP4 with predictions overlaid on each frame.

## Dashboard

Grafana (http://localhost:3000) has a pre-provisioned **V-JEPA2 Inference** dashboard with four sections:

- **Clip Golden Signals** — p50/p95/p99 clip latency, throughput (clips/sec), real-time ratio, and resource utilization (CPU, memory, GPU). The real-time ratio shows whether the system can keep up: values above 1.0 (red line) mean inference is slower than real-time.
- **Pipeline Phase Breakdown** — where clip processing time is spent (decode, preprocess, inference, postprocess), shown as both percentage and absolute duration. Useful for identifying bottlenecks.
- **API Golden Signals** — request latency, throughput, HTTP errors, and active WebSocket connections.
- **Stats** — total clips/frames processed, average clip duration, total requests.

## Traces

After running inference, open Jaeger (http://localhost:16686) and search for the `vjepa2-server` service. Each request produces a trace waterfall:

- **Batch inference** (`/infer`) — `video_inference` span containing one `clip_inference` span per clip
- **Streaming** (camera/RTSP) — `stream_inference` span with `clip_inference` children, showing how clips overlap with ingestion

## CLI

```bash
# Run inference on a video
podman run --rm -v vjepa2-model-vitl:/model:ro -v ./videos:/input:ro \
  quay.io/fzdarsky/vjepa2-server-cpu infer /input/video.mp4

# JSONL output (pipeable)
podman run --rm -v vjepa2-model-vitl:/model:ro -v ./videos:/input:ro \
  quay.io/fzdarsky/vjepa2-server-cpu infer /input/video.mp4 --format jsonl

# Process all videos in a directory
podman run --rm -v vjepa2-model-vitl:/model:ro -v ./videos:/input:ro \
  quay.io/fzdarsky/vjepa2-server-cpu infer
```

**Options:** `--stride N` (clip overlap), `--num-frames N` (default 16), `--top-k N` (predictions per clip), `--format text|json|jsonl`, `--save-frames` (export decoded frames)

Use the larger ViT-G model for higher accuracy: pass `--num-frames 64` and create a volume from `quay.io/fzdarsky/vjepa2-model-vitg`.
