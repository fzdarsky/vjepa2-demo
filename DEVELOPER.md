# Developer Guide

## API Endpoints

- `GET /v2/health/live` — liveness probe
- `GET /v2/health/ready` — readiness probe (model loaded)
- `GET /v2/models/vjepa2` — model metadata
- `POST /v2/models/vjepa2/infer` — video inference (multipart file upload)
- `WS /v2/models/vjepa2/stream` — WebSocket streaming (file-based)
- `WS /v2/models/vjepa2/stream/browser` — browser camera streaming
- `WS /v2/models/vjepa2/stream/rtsp` — RTSP camera streaming
- `GET /v2/models/vjepa2/sessions/{id}/preview` — MJPEG live preview
- `GET /v2/models/vjepa2/sessions/{id}/video` — annotated video download

## Available Models

| Image | HuggingFace ID | Arch | Frames | Res | Size |
| ----- | -------------- | ---- | ------ | --- | ---- |
| `quay.io/fzdarsky/vjepa2-model-vitl` | `facebook/vjepa2-vitl-fpc16-256-ssv2` | ViT-L | 16 | 256px | ~1.4 GB |
| `quay.io/fzdarsky/vjepa2-model-vitg` | `facebook/vjepa2-vitg-fpc64-384-ssv2` | ViT-G | 64 | 384px | ~4.3 GB |

Both models are finetuned on Something-Something v2 (174 action classes) and packaged as uncompressed [ModelCar](https://kserve.github.io/website/latest/modelserving/storage/oci/) OCI images for fast volume creation.

## Building Container Images

Two container variants are available:

| Variant | Base Image | Arch | Accelerator |
| ------- | ---------- | ---- | ----------- |
| CPU | `ubi9/python-312` | x86_64, aarch64 | CPU |
| CUDA | RHOAI pipeline runtime | x86_64 | NVIDIA GPU |

```bash
# Build CPU variant
podman build -t vjepa2-server-cpu -f Containerfile.cpu .

# Build CUDA variant
podman build -t vjepa2-server-cuda -f Containerfile.cuda .
```

Both images pull from `registry.redhat.io` (Red Hat subscription required). Run `podman login registry.redhat.io` first.

## Native Execution (macOS MPS)

For GPU-accelerated inference on Apple Silicon without containers, run the app natively. PyTorch auto-detects the MPS device.

```bash
# Set up virtual environment
python3.12 -m venv .venv && source .venv/bin/activate

# Install dependencies (PyPI provides MPS-capable PyTorch on macOS)
pip install torch torchvision
pip install -r requirements.txt

# Download model
python -m app download --model facebook/vjepa2-vitl-fpc16-256-ssv2 --output ./model-staging

# Run inference with MPS acceleration
MODEL_PATH=./model-staging python -m app infer samples/video.mp4

# Or start the API server
MODEL_PATH=./model-staging python -m app serve
```

MPS acceleration is not available inside containers — Podman's Linux VM exposes Vulkan (via virtio-gpu), not Metal/MPS.

## Compose Profiles

| Profile | Services |
| ------- | -------- |
| `cpu` | CPU inference server |
| `cuda` | CUDA inference server |
| `observability` | OTel Collector, Prometheus, Jaeger, Grafana |
| `gpu-metrics` | DCGM Exporter (NVIDIA GPU metrics) |

Use `podman-compose` (not `docker-compose`) — it integrates directly with Podman's credential store and volume drivers.
