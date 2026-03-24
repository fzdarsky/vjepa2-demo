# V-JEPA2 Video Inference Demo — Design Spec

## Overview

A containerized REST API for running V-JEPA2 video inference, starting as a local Podman container demo on MacBook and designed for eventual deployment on OpenShift with NVIDIA GPUs.

**Milestone 1 scope**: Single container serving a FastAPI application that accepts video file uploads and returns action recognition predictions using V-JEPA2 ViT-L/16 (300M params) fine-tuned on Something-Something v2, loaded via HuggingFace transformers.

## Requirements

### Functional

- Accept video file uploads via REST endpoint and return top-K action predictions with confidence scores
- Support continuous inference over a video source via WebSocket
- Expose health and model metadata endpoints compatible with KServe V2 inference protocol
- Auto-detect compute device (CUDA > MPS > CPU)

### Non-Functional

- Run in a single container on MacBook M4 Pro (ARM, CPU inference) via Podman Desktop
- Container starts and becomes ready within 60 seconds
- API contract follows KServe V2 inference protocol conventions (see Deviations section) for future OpenShift AI compatibility
- Designed for migration to Red Hat UBI base image and KServe/Caikit serving

### Out of Scope (Milestone 1)

- RTSP / webcam live stream ingestion
- GPU-accelerated video decoding (NVDEC)
- Red Hat UBI base image
- KServe / Caikit integration
- Authentication, rate limiting, TLS

## Architecture

### Single Container

```text
┌──────────────────────────────────────────────┐
│  Podman Container                            │
│                                              │
│  ┌──────────────────────────────────────┐    │
│  │  FastAPI Application (port 8080)     │    │
│  │                                      │    │
│  │  /v2/health/ready                    │    │
│  │  /v2/health/live                     │    │
│  │  /v2/models/vjepa2                   │    │
│  │  /v2/models/vjepa2/infer      [POST] │    │
│  │  /v2/models/vjepa2/stream [WebSocket]│    │
│  └──────────┬──────────────┬────────────┘    │
│             │              │                 │
│  ┌──────────▼──────┐ ┌─────▼─────────────┐   │
│  │  video.py       │ │  model.py         │   │
│  │  PyAV decode    │ │  V-JEPA2 backbone │   │
│  │  Frame sampling │ │  SSv2 probe       │   │
│  └─────────────────┘ │  Inference logic  │   │
│                      └───────────────────┘   │
│                                              │
│  ┌──────────────────────────────────────┐    │
│  │  Model Weights (~1.2GB)              │    │
│  │  ViT-L/16 fine-tuned on SSv2         │    │
│  └──────────────────────────────────────┘    │
└──────────────────────────────────────────────┘
```

### Device Selection

```python
if torch.cuda.is_available():
    device = "cuda"          # OpenShift with NVIDIA GPU
elif torch.backends.mps.is_available():
    device = "mps"           # MacBook native (outside container)
else:
    device = "cpu"           # Container on Mac (Linux VM, no Metal)
```

Podman Desktop on Apple Silicon runs containers in a Linux VM via Apple Virtualization.framework. MPS is unavailable inside the container. CPU inference is sufficient for ViT-L/16 on M4 Pro for demo purposes. MPS device detection is included for forward compatibility (milestone 2, native execution) but is not testable in milestone 1.

## Project Structure

```text
jepa-demo/
├── Containerfile
├── compose.yaml
├── requirements.txt
├── README.md
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI app, lifespan, routes
│   ├── model.py                # V-JEPA2 model loading + inference
│   ├── video.py                # Video decoding (PyAV), frame sampling
│   └── schemas.py              # KServe V2 request/response models
├── configs/
│   └── model_config.yaml       # Model variant, checkpoint path, parameters
├── labels/
│   └── ssv2_labels.json        # SSv2 class index → label name mapping (174 classes)
├── scripts/
│   └── download_model.sh       # Downloads HuggingFace model for container build
├── samples/
│   └── sample.mp4              # Small test video for validation
└── docs/
```

## API Contract

### KServe V2 Endpoints

#### Health

```code
GET /v2/health/ready  →  200 (model loaded) | 503 (loading)
GET /v2/health/live   →  200 (process alive)
```

#### Model Metadata

```code
GET /v2/models/vjepa2  →  200
```

```json
{
  "name": "vjepa2",
  "versions": ["vit-l-16-ssv2"],
  "platform": "pytorch",
  "inputs": [
    {"name": "video", "datatype": "BYTES", "shape": [-1]}
  ],
  "outputs": [
    {"name": "predictions", "datatype": "FP32", "shape": [-1]}
  ]
}
```

#### Inference (File Upload)

```code
POST /v2/models/vjepa2/infer
Content-Type: multipart/form-data

file: <video.mp4>
top_k: 5          (optional, default: 5)
num_frames: 16    (optional, default: from model config)
```

Response:

```json
{
  "model_name": "vjepa2",
  "model_version": "vit-l-16-ssv2",
  "id": "req-abc123",
  "outputs": [
    {
      "name": "predictions",
      "shape": [5],
      "datatype": "FP32",
      "data": [
        {"label": "Pushing something from left to right", "score": 0.87},
        {"label": "Moving something across a surface", "score": 0.06},
        {"label": "Sliding something across a surface", "score": 0.03},
        {"label": "Pushing something so it slightly moves", "score": 0.02},
        {"label": "Poking something so it slightly moves", "score": 0.01}
      ]
    }
  ]
}
```

#### WebSocket Streaming

```code
GET /v2/models/vjepa2/stream  →  WebSocket upgrade
```

Client sends:

```json
{
  "source": "/path/to/video.mp4",
  "top_k": 5,
  "num_frames": 16,
  "stride": 8
}
```

Server sends predictions repeatedly as JSON messages:

```json
{
  "timestamp_ms": 1500,
  "frame_range": [24, 40],
  "predictions": [
    {"label": "Pushing something from left to right", "score": 0.87},
    {"label": "Moving something across a surface", "score": 0.06}
  ]
}
```

Server sends a final message when the source ends:

```json
{"status": "complete", "frames_processed": 240}
```

**Design notes:**

- `stride` controls overlap: with `num_frames=16` and `stride=8`, inference runs every 8 frames with 50% overlap. This becomes the knob for latency vs. compute when streaming live video.
- `source` must be a path accessible inside the container (e.g., mounted via volume in compose.yaml). In milestone 2, this extends to RTSP URLs and device identifiers.

### KServe V2 Deviations

This API is **KServe V2-inspired**, not strictly conformant. Deviations from the spec, with rationale:

| Deviation | KServe V2 Strict | Our API | Rationale |
| --------- | --------------- | ------- | --------- |
| Request format | JSON body with `inputs[]` containing base64-encoded tensor data | Multipart file upload | More natural for video; KServe's binary extension mechanism supports this pattern |
| Response `data` | Flat array of numeric values | Array of `{label, score}` objects | More useful for clients; avoids requiring clients to maintain a separate label index |
| Output `datatype` | `FP32` implies numeric array | Mixed object array | Reflects the actual response structure |

Milestone 3 will provide strict KServe V2 conformance when deploying via KServe InferenceService on OpenShift.

### Constraints

- **Maximum upload size**: 100MB per video file. Enforced via uvicorn `--limit-request-body 104857600` CLI flag.
- **Accepted formats**: MP4, AVI, MOV, MKV — any container format supported by FFmpeg/PyAV with H.264, H.265, or VP9 video codecs.
- **Frame sampling**: Frames are sampled by frame index (not timestamp), uniformly spaced. If the video has fewer frames than `num_frames`, the request returns 400.
- **WebSocket source paths**: Must resolve within a configured allowed directory (default: `/samples/`). Paths containing `..` or resolving outside the allowed directory are rejected. This prevents path-traversal attacks.
- **`DEVICE` environment variable**: If set, overrides auto-detection. Valid values: `cpu`, `mps`, `cuda`. If the requested device is unavailable, the application fails fast at startup with a clear error message.

## Components

### app/video.py — Video Decoding

Responsibilities:

- Open a video file using PyAV
- Sample N frames uniformly by frame index (not timestamp) across the video duration
- Return raw frames as a list of PIL Images (no resize, crop, or normalization — that's the HuggingFace processor's job)
- For WebSocket mode: maintain a sliding window buffer, yield frame lists every `stride` frames
- If the video has fewer frames than `num_frames`, raise an error (returns 400 for REST, error message for WebSocket)

**Preprocessing pipeline**: `video.py` handles only decoding and frame selection. The HuggingFace processor (loaded in `model.py`) handles resize, center-crop to square, and normalization. This avoids double-processing and ensures preprocessing matches the model's training pipeline.

Key design choice: **PyAV over decord**. decord offers faster random access and direct-to-tensor decoding, but does not build on macOS. PyAV wraps FFmpeg, works on ARM, and the performance difference is negligible for single-video inference. On OpenShift, decord or NVIDIA PyNvVideoCodec can be swapped in without changing the interface.

Interface:

```python
def decode_video(source: str | Path, num_frames: int) -> list[Image.Image]:
    """Decode a video file and return uniformly sampled frames as PIL Images.
    Raises ValueError if video has fewer than num_frames frames."""

def stream_frames(source: str | Path, num_frames: int, stride: int) -> Iterator[list[Image.Image]]:
    """Yield sliding window frame lists from a video source.
    Each list contains num_frames PIL Images."""
```

### app/model.py — Model Loading & Inference

Responsibilities:

- Load V-JEPA2 ViT-L/16 fine-tuned on SSv2 via HuggingFace transformers
- Load the HuggingFace processor for frame preprocessing (resize, center-crop, normalize)
- Run inference: preprocessed frames → logits → predictions
- Map logit indices to SSv2 label names
- Manage device placement (CPU/MPS/CUDA)

Model loading uses `facebook/vjepa2-vitl-fpc16-256-ssv2` from HuggingFace. This is the smallest V-JEPA2 model with an SSv2 classification head included — no separate probe loading needed. The model is downloaded and cached at container build time.

The correct HuggingFace preprocessor class must be verified against the model card at implementation time (likely `AutoProcessor` or `AutoImageProcessor` — `AutoVideoProcessor` is not a standard transformers class). The preprocessor handles resize, center-crop to 256x256, and normalization.

SSv2 label names (174 classes) are not included in the V-JEPA2 repo. A `labels/ssv2_labels.json` file mapping indices 0-173 to human-readable labels (e.g., `"Pushing something from left to right"`) will be sourced from the [Something-Something v2 dataset](https://developer.qualcomm.com/software/ai-datasets/something-something) and included in the repo.

Interface:

```python
class VJepa2Model:
    def __init__(self, config: ModelConfig, device: str): ...

    def predict(self, frames: torch.Tensor, top_k: int = 5) -> list[Prediction]:
        """Run inference on a frame tensor.
        Input: [num_frames, 3, H, W]
        Output: list of (label, score) predictions, sorted by score descending."""
```

The `predict` method is stateless with respect to video — it takes a tensor and returns predictions. This is the invariant that stays the same across file upload, WebSocket streaming, and future KServe integration.

### app/schemas.py — KServe V2 Models

Pydantic models for:

- `InferenceResponse` — KServe V2 response envelope
- `ModelMetadata` — model name, version, input/output spec
- `Prediction` — label + score pair
- `StreamConfig` — WebSocket session parameters
- `StreamPrediction` — timestamped prediction for streaming

### app/main.py — FastAPI Application

Responsibilities:

- FastAPI app with lifespan handler (load model on startup)
- Route definitions for all endpoints
- File upload handling, input validation
- WebSocket connection management
- Error handling (invalid video format, model not ready, etc.)

### configs/model_config.yaml

```yaml
model:
  name: vjepa2
  hf_model_id: facebook/vjepa2-vitl-fpc16-256-ssv2
  num_classes: 174
  labels: /app/labels/ssv2_labels.json

inference:
  num_frames: 16
  resolution: 256
  default_top_k: 5

server:
  host: 0.0.0.0
  port: 8080
  max_upload_bytes: 104857600    # 100MB
  allowed_source_dir: /samples   # WebSocket source path restriction
```

## Container Build

### Containerfile (multi-stage)

**Stage 1 — Downloader**: Uses `python:3.12-slim` to download the HuggingFace model (`facebook/vjepa2-vitl-fpc16-256-ssv2`) using `huggingface_hub`. Model files are cached as a build layer (~1.2GB).

**Stage 2 — Runtime**: Based on `python:3.12-slim` (ARM64-native). Installs PyTorch CPU via pip from the default PyPI index, plus FastAPI, PyAV, uvicorn, transformers, and other dependencies. Copies application code and model weights. Exposes port 8080.

**Why not `pytorch/pytorch` base image?** The official PyTorch Docker images are x86_64-only. On MacBook M4 Pro, Podman runs an ARM64 Linux VM — an x86_64 image would require QEMU emulation (extremely slow). Using `python:3.12-slim` with pip-installed PyTorch CPU gives us a native ARM64 container. For milestone 3 (OpenShift with NVIDIA GPUs on x86_64), we switch to a UBI + CUDA base image.

**PyTorch installation**: Install via `pip install torch` from the default PyPI index (not `--index-url https://download.pytorch.org/whl/cpu`, which only carries x86_64 wheels). PyPI publishes aarch64 Linux wheels with CUDA stubs disabled at runtime. Verify wheel resolution in a test build early.

### compose.yaml

```yaml
services:
  vjepa2:
    build: .
    ports:
      - "8080:8080"
    volumes:
      - ./samples:/samples:ro    # Mount sample videos
    environment:
      - DEVICE=cpu               # Override device selection
```

Run with `podman compose up --build`.

## Error Handling

| Scenario | Response |
| -------- | -------- |
| Model not loaded yet | 503 with `{"error": "Model not ready"}` |
| Invalid/corrupt video file | 400 with `{"error": "Could not decode video: <detail>"}` |
| Video too short (fewer than num_frames) | 400 with `{"error": "Video has N frames, need at least M"}` |
| Unsupported format | 400 with `{"error": "Unsupported video format"}` |
| Internal inference error | 500 with `{"error": "Inference failed: <detail>"}` |
| WebSocket invalid config | Send `{"error": "Invalid config: <detail>"}`, then close with code 1008 |
| WebSocket source not found | Send `{"error": "File not found: <path>"}`, then close with code 1008 |
| WebSocket path traversal | Send `{"error": "Source path not allowed"}`, then close with code 1008 |
| WebSocket mid-stream failure | Send `{"error": "Inference failed: <detail>"}`, then close with code 1011 (internal error) |
| Requested device unavailable | Application fails to start with exit code 1 and error log |

## Future Milestones (Context Only)

### Milestone 2 — Native Performance & Larger Models

- Native execution (outside container) for MPS acceleration on M4 Pro
- Test with ViT-G/16 (1B+ params)
- RTSP and webcam stream sources for WebSocket endpoint
- Inference latency and throughput metrics

### Milestone 3 — OpenShift Deployment

- Red Hat UBI base image + CUDA runtime
- KServe InferenceService on OpenShift AI
- ViT-g/16+ on NVIDIA GPUs with NVDEC hardware decoding
- Replace FastAPI with KServe custom predictor or Caikit module
- Kafka/Redis output for streaming predictions
- Horizontal scaling, GPU sharing, monitoring
