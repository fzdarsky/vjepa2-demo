#!/bin/bash
# Build script for vjepa2-demo container images
#
# Usage:
#   ./build.sh cpu              # Build CPU image (both archs)
#   ./build.sh cpu-manifest     # Build CPU + push + manifest
#   ./build.sh cuda             # Build CUDA image (amd64 only)
#   ./build.sh cuda-manifest    # Build CUDA + push
#   ./build.sh all              # Build all images with manifests
#   PUSH=true ./build.sh all    # Build and push to registry
#
# Environment:
#   REGISTRY:    Container registry (default: quay.io/fzdarsky)
#   TAG:         Image tag (default: latest)
#   PUSH:        Push to registry after build (default: false)
#   REMOTE_AMD64_HOST: SSH host for amd64 builds (default: lab-05)
#   REMOTE_ARM64_HOST: SSH host for arm64 builds (default: spark-2)
#
# Note: For long-running remote builds, protect your session:
#   nohup ./build.sh all > build.log 2>&1 &
#   tail -f build.log

set -e

REGISTRY=${REGISTRY:-quay.io/fzdarsky}
TAG=${TAG:-latest}
PUSH=${PUSH:-false}
REMOTE_AMD64_HOST=${REMOTE_AMD64_HOST:-lab-05}
REMOTE_ARM64_HOST=${REMOTE_ARM64_HOST:-spark-2}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

ARCH=$(uname -m)
log_info "Local architecture: $ARCH"

# --- Helpers ---

rsync_to_remote() {
    local host="$1" dest="$2"
    rsync -avz --exclude='.git' --exclude='.venv' --exclude='venv' --exclude='output' \
        "$SCRIPT_DIR/" "${host}:${dest}"
}

# --- CPU (multi-arch) ---

build_cpu() {
    log_info "Building CPU images: ${REGISTRY}/vjepa2-server-cpu:${TAG}"

    local image_base="${REGISTRY}/vjepa2-server-cpu"

    log_info "Syncing to build hosts..."
    rsync_to_remote "${REMOTE_AMD64_HOST}" "~/vjepa2-demo"
    rsync_to_remote "${REMOTE_ARM64_HOST}" "~/vjepa2-demo"

    log_info "Building ${image_base}:amd64-${TAG} on ${REMOTE_AMD64_HOST}..."
    ssh "${REMOTE_AMD64_HOST}" "cd ~/vjepa2-demo && \
        podman build -f Containerfile.cpu -t ${image_base}:amd64-${TAG} ."

    log_info "Building ${image_base}:arm64-${TAG} on ${REMOTE_ARM64_HOST}..."
    ssh "${REMOTE_ARM64_HOST}" "cd ~/vjepa2-demo && \
        sudo podman build -f Containerfile.cpu -t ${image_base}:arm64-${TAG} ."

    log_info "CPU images built on both architectures"
}

build_cpu_manifest() {
    log_info "Creating multi-arch manifest for CPU image..."

    local image_base="${REGISTRY}/vjepa2-server-cpu"

    log_info "Pushing arch-specific images..."
    ssh "${REMOTE_AMD64_HOST}" "podman push ${image_base}:amd64-${TAG}"
    ssh "${REMOTE_ARM64_HOST}" "sudo podman push --authfile /run/containers/0/auth.json ${image_base}:arm64-${TAG}"

    log_info "Creating manifest ${image_base}:${TAG}..."
    podman manifest rm "${image_base}:${TAG}" 2>/dev/null || true
    podman manifest create "${image_base}:${TAG}"
    podman manifest add "${image_base}:${TAG}" "docker://${image_base}:arm64-${TAG}"
    podman manifest add "${image_base}:${TAG}" "docker://${image_base}:amd64-${TAG}"

    log_info "Pushing manifest..."
    podman manifest push --all "${image_base}:${TAG}" "docker://${image_base}:${TAG}"

    log_info "Multi-arch CPU manifest created: ${image_base}:${TAG}"
}

# --- CUDA (amd64 only — RHOAI base image has no arm64 variant) ---

build_cuda() {
    log_info "Building CUDA image: ${REGISTRY}/vjepa2-server-cuda:${TAG} (amd64 only)"

    local image_base="${REGISTRY}/vjepa2-server-cuda"

    log_info "Syncing to ${REMOTE_AMD64_HOST}..."
    rsync_to_remote "${REMOTE_AMD64_HOST}" "~/vjepa2-demo"

    log_info "Building ${image_base}:amd64-${TAG} on ${REMOTE_AMD64_HOST}..."
    ssh "${REMOTE_AMD64_HOST}" "cd ~/vjepa2-demo && \
        podman build -f Containerfile.cuda -t ${image_base}:amd64-${TAG} ."

    log_info "CUDA image built"
}

build_cuda_manifest() {
    log_info "Pushing CUDA image (amd64 only)..."

    local image_base="${REGISTRY}/vjepa2-server-cuda"

    ssh "${REMOTE_AMD64_HOST}" "podman push ${image_base}:amd64-${TAG}"
    ssh "${REMOTE_AMD64_HOST}" "podman tag ${image_base}:amd64-${TAG} ${image_base}:${TAG} && podman push ${image_base}:${TAG}"

    log_info "CUDA image pushed: ${image_base}:${TAG}"
}

# --- Main ---

case "${1:-}" in
    cpu)
        build_cpu
        ;;
    cpu-manifest)
        build_cpu
        build_cpu_manifest
        ;;
    cuda)
        build_cuda
        ;;
    cuda-manifest)
        build_cuda
        build_cuda_manifest
        ;;
    all)
        build_cpu
        build_cpu_manifest
        build_cuda
        build_cuda_manifest
        ;;
    *)
        echo "Usage: $0 {cpu|cpu-manifest|cuda|cuda-manifest|all}"
        echo ""
        echo "Commands:"
        echo "  cpu              Build CPU image on both archs"
        echo "  cpu-manifest     Build CPU + push + create manifest"
        echo "  cuda             Build CUDA image (amd64 only)"
        echo "  cuda-manifest    Build CUDA + push"
        echo "  all              Build all images with manifests"
        echo ""
        echo "Environment:"
        echo "  REGISTRY=$REGISTRY"
        echo "  TAG=$TAG"
        echo "  REMOTE_AMD64_HOST=$REMOTE_AMD64_HOST"
        echo "  REMOTE_ARM64_HOST=$REMOTE_ARM64_HOST"
        exit 1
        ;;
esac

log_info "Done!"
