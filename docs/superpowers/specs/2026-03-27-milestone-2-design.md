# Milestone 2 — Red Hat Base Images & GPU Acceleration

## Overview

Migrate the V-JEPA2 inference container from `python:3.12-slim` (Debian) to Red Hat base images, with two container variants: a CUDA variant for NVIDIA GPUs and a CPU variant for general-purpose / Apple Silicon dev use. Add a documented native MPS execution path for macOS performance comparison. Include EC2 infrastructure for CUDA testing.

**Milestone 2 scope**: Replace the single Debian-based Containerfile with two Red Hat-based Containerfiles (CUDA and CPU), update compose.yaml with profiles, add Terraform config for EC2 GPU testing, and document native MPS execution.

## Requirements

### Functional

- CUDA container variant based on `rhoai/odh-pipeline-runtime-pytorch-cuda-py312-rhel9` for NVIDIA GPU inference
- CPU container variant based on `ubi9/python-312` for CPU inference on x86_64 and aarch64
- Both variants support the same API, CLI, and ModelCar volume pattern from milestones 1/1.5
- Native MPS execution path documented for macOS Apple Silicon performance comparison
- EC2 GPU test infrastructure via Terraform for validating the CUDA variant

### Non-Functional

- CPU variant runs natively on ARM (aarch64) — no QEMU emulation
- Existing 27 unit tests continue to pass unchanged
- Container layer ordering optimized for app-code-change rebuilds (PyTorch cached, app code last)

### Out of Scope (Milestone 2)

- Observability / metrics (milestone 3)
- Stream sources — RTSP, browser (milestone 4)
- KServe / Caikit integration
- Multi-GPU or GPU sharing
- Containerized MPS acceleration (not feasible — Podman's virtio-gpu exposes Vulkan, not MPS)

## Architecture

### Container Variants

The single `Containerfile` is replaced by two variant-specific Containerfiles. Each is independently optimized for its target platform. No attempt is made to pin exact library versions across variants — the RHOAI image ships its own tested PyTorch+CUDA stack, and the CPU variant uses whatever PyPI provides.

```text
┌─────────────────────────────────┐    ┌─────────────────────────────────┐
│  Containerfile.cuda             │    │  Containerfile.cpu              │
│                                 │    │                                 │
│  Base: rhoai/odh-pipeline-      │    │  Base: ubi9/python-312          │
│    runtime-pytorch-cuda-        │    │  Arch: x86_64 + aarch64         │
│    py312-rhel9:v3.4.0-ea.1      │    │                                 │
│  Arch: x86_64 only              │    │  Layer 1: UBI9 Python base      │
│                                 │    │  Layer 2: System deps (dnf)     │
│  Layer 1: RHOAI base            │    │  Layer 3: PyTorch from PyPI     │
│  Layer 2: App deps (pip)        │    │  Layer 4: App deps (pip)        │
│  Layer 3: App code + configs    │    │  Layer 5: App code + configs    │
│                                 │    │                                 │
│  PyTorch: pre-installed         │    │  PyTorch: pip install torch     │
│  CUDA: 12.9.1                   │    │           (PyPI default index)  │
│  Python: 3.12                   │    │  Python: 3.12                   │
└─────────────────────────────────┘    └─────────────────────────────────┘
```

### Execution Paths

```text
                    ┌──────────────────────┐
                    │  V-JEPA2 App Code    │
                    │  (unchanged)         │
                    │                      │
                    │  select_device()     │
                    │  CUDA > MPS > CPU    │
                    └──────┬───────────────┘
                           │
              ┌────────────┼────────────────┐
              │            │                │
    ┌─────────▼──┐  ┌──────▼─────┐  ┌───────▼──────┐
    │ CUDA       │  │ MPS        │  │ CPU          │
    │ Container  │  │ Native     │  │ Container    │
    │ (EC2/OCP)  │  │ (macOS)    │  │ (any arch)   │
    │            │  │            │  │              │
    │ Dockerfile │  │ README     │  │ Dockerfile   │
    │ .cuda      │  │ docs only  │  │ .cpu         │
    └────────────┘  └────────────┘  └──────────────┘
```

### Why Not Containerized MPS?

Podman Desktop on Apple Silicon runs containers in a Linux VM via Apple Virtualization.framework. GPU passthrough uses virtio-gpu, which exposes a Vulkan interface (via Mesa's Venus driver → MoltenVK → Metal). This works for applications with Vulkan compute backends (e.g., llama.cpp's ggml-vulkan gave a 40x speedup), but PyTorch has no Vulkan compute backend — it supports CUDA, MPS (macOS-native only), and CPU. MPS requires direct access to the Metal API, which is unavailable inside a Linux VM.

## Container Details

### Containerfile.cuda

```dockerfile
FROM registry.redhat.io/rhoai/odh-pipeline-runtime-pytorch-cuda-py312-rhel9:v3.4.0-ea.1

WORKDIR /app

# App dependencies (torch/torchvision pre-installed in base)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY configs/ ./configs/

VOLUME /model
VOLUME /input
VOLUME /output

RUN mkdir -p /input /output

EXPOSE 8080

ENTRYPOINT ["python", "-m", "app"]
CMD ["serve"]
```

### Containerfile.cpu

```dockerfile
FROM registry.redhat.io/ubi9/python-312:latest

WORKDIR /app

# System dependencies for PyAV (FFmpeg) — dnf instead of apt-get
RUN dnf install -y --nodocs \
      libavcodec-free-devel libavformat-free-devel \
      libavutil-free-devel libswscale-free-devel && \
    dnf clean all

# PyTorch CPU from PyPI default index (has aarch64 wheels)
RUN pip install --no-cache-dir torch>=2.6.0 torchvision>=0.21.0

# App dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY configs/ ./configs/

VOLUME /model
VOLUME /input
VOLUME /output

RUN mkdir -p /input /output

EXPOSE 8080

ENTRYPOINT ["python", "-m", "app"]
CMD ["serve"]
```

**Note on FFmpeg packages**: UBI9/RHEL9 ships the "free" FFmpeg codec libraries (`libavcodec-free`, not `libavcodec`). These exclude patent-encumbered codecs but include H.264 decode via OpenH264. The exact package names must be verified during implementation — the names above are best-effort based on RHEL9 repos.

### compose.yaml

```yaml
services:
  vjepa2-server-cuda:
    profiles: [cuda]
    build:
      context: .
      dockerfile: Containerfile.cuda
    ports:
      - "8080:8080"
    volumes:
      - vjepa2-model-vitl:/model:ro
      - ./samples:/input:ro
      - ./output:/output
    environment:
      - DEVICE=cuda

  vjepa2-server-cpu:
    profiles: [cpu]
    build:
      context: .
      dockerfile: Containerfile.cpu
    ports:
      - "8080:8080"
    volumes:
      - vjepa2-model-vitl:/model:ro
      - ./samples:/input:ro
      - ./output:/output
    environment:
      - DEVICE=cpu

volumes:
  vjepa2-model-vitl:
    external: true
```

Run with:

```bash
podman compose --profile cuda up --build
podman compose --profile cpu up --build
```

### requirements.txt

Remove `torch` and `torchvision` lines. PyTorch installation is handled per-Containerfile since the source differs (pre-installed in RHOAI vs. pip from PyPI). Remaining dependencies:

```text
transformers>=4.53.0
accelerate>=1.6.0
pillow>=11.0
av>=14.0.0
fastapi>=0.115.0
uvicorn[standard]>=0.34.0
python-multipart>=0.0.20
pyyaml>=6.0
websockets>=14.0
numpy>=2.0
```

Test dependencies (`pytest`, `pytest-asyncio`, `httpx`) are not installed in the container — they remain in a separate dev requirements file or are installed only for local/CI testing.

## Native MPS Execution

Documentation-only path added to README. No code changes required — `select_device()` in `app/model.py` already auto-detects MPS.

Documented steps:

1. Prerequisites: Python 3.12, macOS with Apple Silicon
2. Create virtualenv, install dependencies: `pip install -r requirements.txt` plus `pip install torch torchvision` (PyPI provides MPS-capable PyTorch on macOS)
3. Download model: `python -m app download --model facebook/vjepa2-model-vitl-fpc16-256-ssv2 --output ./model-staging`
4. Run inference: `MODEL_PATH=./model-staging python -m app infer samples/video.mp4`
5. Run server: `MODEL_PATH=./model-staging python -m app serve`

## EC2 GPU Test Infrastructure

Terraform configuration in `tests/ec2-gpu/` for spinning up an NVIDIA GPU instance to validate the CUDA container.

### Files

```text
tests/ec2-gpu/
├── main.tf          # EC2 instance, security group, key pair
├── variables.tf     # instance type, region, AMI, SSH key
├── outputs.tf       # public IP, SSH command
├── user-data.sh     # cloud-init: install Podman, pull image, run tests
└── README.md        # usage instructions
```

### Instance spec

- **Instance type**: `g4dn.xlarge` (1x NVIDIA T4, 16GB VRAM), ~$0.53/hr
- **AMI**: RHEL 9 (GPU-optimized or standard + driver install via user-data)
- **Region**: configurable, default `us-east-1`

### user-data.sh workflow

1. Install Podman and nvidia-container-toolkit
2. Authenticate to `registry.redhat.io` (credentials via Terraform variable)
3. Build the CUDA container image
4. Create ModelCar volume, download model
5. Run inference on a sample video, verify output
6. Print timing and device info to cloud-init log

### Usage

```bash
cd tests/ec2-gpu
terraform init
terraform apply -var="rh_registry_user=..." -var="rh_registry_pass=..."
# SSH in to inspect results, or check cloud-init log
terraform destroy   # tear down when done
```

## Project Structure Changes

```text
jepa-demo/
├── Containerfile              # DELETED
├── Containerfile.cuda         # NEW
├── Containerfile.cpu          # NEW
├── compose.yaml               # MODIFIED — profiles for cuda/cpu
├── requirements.txt           # MODIFIED — torch/torchvision removed
├── README.md                  # MODIFIED — MPS docs, multi-variant build docs
├── Modelfile                  # UNCHANGED
├── app/                       # UNCHANGED
├── configs/                   # UNCHANGED
├── tests/
│   ├── ec2-gpu/               # NEW
│   │   ├── main.tf
│   │   ├── variables.tf
│   │   ├── outputs.tf
│   │   ├── user-data.sh
│   │   └── README.md
│   ├── test_*.py              # UNCHANGED
│   └── conftest.py            # UNCHANGED
└── docs/
    └── superpowers/
        └── specs/
            └── 2026-03-27-milestone-2-design.md
```

## Error Handling

| Scenario | Behavior |
| -------- | -------- |
| `registry.redhat.io` auth failure during build | Build fails with clear pull error. User must `podman login registry.redhat.io` first. |
| CUDA requested but unavailable (CPU container) | Existing `select_device()` raises `RuntimeError("CUDA requested but not available")` — unchanged. |
| FFmpeg codec libraries missing in UBI9 | Build fails at `dnf install`. Package names verified during implementation. |
| EC2 instance launch fails (quota/region) | Terraform reports error. README documents common issues. |
| nvidia-container-toolkit not working on EC2 | user-data.sh logs errors. README documents troubleshooting. |

## Future Milestones (Context Only)

### Milestone 3 — Observability & Metrics

- Inference latency, throughput, and GPU utilization metrics
- OpenTelemetry / Prometheus export
- Compatible with OpenShift AI and Cluster Observability Operator

### Milestone 4 — Stream Sources

- RTSP streams from cameras (UniFi IP cameras / Protect NVR)
- Browser camera streams via WebTransport or WebRTC
- Browser frontend for video + inference results side-by-side
