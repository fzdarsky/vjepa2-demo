# V-JEPA2 Video Inference Demo

Video action recognition using Meta's [V-JEPA2](https://github.com/facebookresearch/jepa) model, packaged as a container with CLI and REST API.

## Quick Start

```bash
# Pull the inference server and a model
podman pull quay.io/fzdarsky/vjepa2:latest
podman pull quay.io/fzdarsky/vjepa2-vitl:latest

# Create a model volume from the ModelCar image
podman volume create --driver image \
  --opt image=quay.io/fzdarsky/vjepa2-vitl:latest \
  vjepa2-vitl

# Run inference on a video file
podman run --rm \
  -v vjepa2-vitl:/model:ro \
  -v ./my-videos:/input:ro \
  quay.io/fzdarsky/vjepa2 infer /input/video.mp4
```

## Available Models

| Image | HuggingFace ID | Arch | Frames | Res | Size |
| ----- | -------------- | ---- | ------ | --- | ---- |
| `quay.io/fzdarsky/vjepa2-vitl` | `facebook/vjepa2-vitl-fpc16-256-ssv2` | ViT-L | 16 | 256px | ~1.4 GB |
| `quay.io/fzdarsky/vjepa2-vitg` | `facebook/vjepa2-vitg-fpc64-384-ssv2` | ViT-G | 64 | 384px | ~4.3 GB |

Both models are finetuned on Something-Something v2 (174 action classes) and packaged as uncompressed [ModelCar](https://kserve.github.io/website/latest/modelserving/storage/oci/) OCI images for fast volume creation. ViT-G uses 64 frames per clip at higher resolution — pass `--num-frames 64` when using it.

## CLI Usage

```bash
# Clip-based inference with text output (default)
podman run --rm -v vjepa2-vitl:/model:ro -v ./videos:/input:ro \
  quay.io/fzdarsky/vjepa2 infer /input/video.mp4

# JSONL output (one JSON object per clip, pipeable)
podman run --rm -v vjepa2-vitl:/model:ro -v ./videos:/input:ro \
  quay.io/fzdarsky/vjepa2 infer /input/video.mp4 --format jsonl

# Export frames to see what the model sees
podman run --rm -v vjepa2-vitl:/model:ro -v ./videos:/input:ro -v ./output:/output \
  quay.io/fzdarsky/vjepa2 infer /input/video.mp4 --save-frames

# Process all videos in /input/
podman run --rm -v vjepa2-vitl:/model:ro -v ./videos:/input:ro \
  quay.io/fzdarsky/vjepa2 infer
```

**Options:** `--stride N` (clip overlap), `--num-frames N` (default 16), `--top-k N` (predictions per clip), `--format text|json|jsonl`

## API Server

```bash
# Start the server
podman run --rm -p 8080:8080 \
  -v vjepa2-vitl:/model:ro \
  -v ./videos:/input:ro \
  quay.io/fzdarsky/vjepa2 serve

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

```bash
# Build the inference server
podman build -t vjepa2 -f Containerfile .

# Download model weights
podman run --rm -v ./model:/model vjepa2 download \
  --model facebook/vjepa2-vitl-fpc16-256-ssv2

# Build a ModelCar image (uncompressed for fast startup)
podman build --disable-compression -f Modelfile.vitl \
  -t quay.io/fzdarsky/vjepa2-vitl:latest .
```
