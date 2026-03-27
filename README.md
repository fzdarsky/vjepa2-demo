# V-JEPA2 Video Inference Demo

Video action recognition using Meta's [V-JEPA2](https://github.com/facebookresearch/jepa) model, packaged as a container with CLI and REST API.

## Quick Start

```bash
# Pull the inference server and a model
podman pull quay.io/fzdarsky/vjepa2-server-cpu:latest
podman pull quay.io/fzdarsky/vjepa2-model-vitl:latest

# Create a model volume from the ModelCar image
podman volume create --driver image \
  --opt image=quay.io/fzdarsky/vjepa2-model-vitl:latest \
  vjepa2-model-vitl

# Run inference on a video file
podman run --rm \
  -v vjepa2-model-vitl:/model:ro \
  -v ./my-videos:/input:ro \
  quay.io/fzdarsky/vjepa2-server-cpu infer /input/video.mp4
```

## Available Models

| Image | HuggingFace ID | Arch | Frames | Res | Size |
| ----- | -------------- | ---- | ------ | --- | ---- |
| `quay.io/fzdarsky/vjepa2-model-vitl` | `facebook/vjepa2-vitl-fpc16-256-ssv2` | ViT-L | 16 | 256px | ~1.4 GB |
| `quay.io/fzdarsky/vjepa2-model-vitg` | `facebook/vjepa2-vitg-fpc64-384-ssv2` | ViT-G | 64 | 384px | ~4.3 GB |

Both models are finetuned on Something-Something v2 (174 action classes) and packaged as uncompressed [ModelCar](https://kserve.github.io/website/latest/modelserving/storage/oci/) OCI images for fast volume creation. ViT-G uses 64 frames per clip at higher resolution — pass `--num-frames 64` when using it.

## CLI Usage

```bash
# Clip-based inference with text output (default)
podman run --rm -v vjepa2-model-vitl:/model:ro -v ./videos:/input:ro \
  quay.io/fzdarsky/vjepa2-server-cpu infer /input/video.mp4

# JSONL output (one JSON object per clip, pipeable)
podman run --rm -v vjepa2-model-vitl:/model:ro -v ./videos:/input:ro \
  quay.io/fzdarsky/vjepa2-server-cpu infer /input/video.mp4 --format jsonl

# Export frames to see what the model sees
podman run --rm -v vjepa2-model-vitl:/model:ro -v ./videos:/input:ro -v ./output:/output \
  quay.io/fzdarsky/vjepa2-server-cpu infer /input/video.mp4 --save-frames

# Process all videos in /input/
podman run --rm -v vjepa2-model-vitl:/model:ro -v ./videos:/input:ro \
  quay.io/fzdarsky/vjepa2-server-cpu infer
```

**Options:** `--stride N` (clip overlap), `--num-frames N` (default 16), `--top-k N` (predictions per clip), `--format text|json|jsonl`

## API Server

```bash
# Start the server
podman run --rm -p 8080:8080 \
  -v vjepa2-model-vitl:/model:ro \
  -v ./videos:/input:ro \
  quay.io/fzdarsky/vjepa2-server-cpu serve

# Single-clip inference
curl -X POST http://localhost:8080/v2/models/vjepa2/infer \
  -F "file=@video.mp4"

# Multi-clip inference with stride
curl -X POST http://localhost:8080/v2/models/vjepa2/infer \
  -F "file=@video.mp4" -F "stride=16"
```

**Endpoints:**

- `GET /v2/health/live` — liveness probe
- `GET /v2/health/ready` — readiness probe (model loaded)
- `GET /v2/models/vjepa2` — model metadata
- `POST /v2/models/vjepa2/infer` — video inference
- `WS /v2/models/vjepa2/stream` — WebSocket streaming

## Building from Source

### Container Variants

Two container images are available, optimized for different platforms:

| Variant | Base Image | Arch | Accelerator | Build |
| ------- | ---------- | ---- | ----------- | ----- |
| CPU | `ubi9/python-312` | x86_64, aarch64 | CPU | `podman compose --profile cpu up --build` |
| CUDA | RHOAI pipeline runtime | x86_64 | NVIDIA GPU | `podman compose --profile cuda up --build` |

Both variants require a ModelCar volume with model weights:

```bash
# Create model volume (one-time setup)
podman volume create --driver image \
  --opt image=quay.io/fzdarsky/vjepa2-model-vitl:latest \
  vjepa2-model-vitl

# Build and run CPU variant
podman compose --profile cpu up --build

# Build and run CUDA variant (requires NVIDIA GPU + nvidia-container-toolkit)
podman compose --profile cuda up --build
```

To build individual images without compose:

```bash
podman build -t vjepa2-server-cpu -f Containerfile.cpu .
podman build -t vjepa2-server-cuda -f Containerfile.cuda .
```

**Note:** Both images pull from `registry.redhat.io` (Red Hat subscription required). Run `podman login registry.redhat.io` first.

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

MPS acceleration is not available inside containers — Podman's Linux VM exposes Vulkan (via virtio-gpu), not Metal/MPS. Use native execution for Apple Silicon GPU performance.
