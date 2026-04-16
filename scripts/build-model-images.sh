#!/bin/bash
# Build and push OCI model images for V-JEPA2
#
# This script:
# 1. Downloads models from HuggingFace
# 2. Optionally shards safetensors for parallel layer pulls
# 3. Builds OCI images with --disable-compression
# 4. Pushes to registry
#
# Usage:
#   ./scripts/build-model-images.sh [options] [vitl|vitg|all]
#
# Options:
#   --shard [N]     Shard safetensors into N parts (default: 4)
#   --registry URL  Target registry (default: quay.io/fzdarsky)
#   --work-dir DIR  Working directory (default: ~/jepa-model-build)
#   --no-push       Build only, don't push
#   --help          Show this help
#
# Requirements:
#   - Python 3.12+ with torch, safetensors, huggingface_hub
#   - Podman with registry credentials configured

set -euo pipefail

# Defaults
REGISTRY="${REGISTRY:-quay.io/fzdarsky}"
WORK_DIR="${WORK_DIR:-$HOME/jepa-model-build}"
SHARD=""
NUM_SHARDS=4
NO_PUSH=false
TARGET="all"

# Model definitions
declare -A MODELS=(
    ["vitl"]="facebook/vjepa2-vitl-fpc16-256-ssv2"
    ["vitg"]="facebook/vjepa2-vitg-fpc64-384-ssv2"
)

usage() {
    sed -n '2,/^$/p' "$0" | grep '^#' | sed 's/^# \?//'
    exit 0
}

log() {
    echo "[$(date '+%H:%M:%S')] $*"
}

download_model() {
    local model_id="$1"
    local output_dir="$2"

    log "Downloading $model_id to $output_dir..."

    python3 << EOF
from huggingface_hub import snapshot_download
snapshot_download("$model_id", local_dir="$output_dir")
print("Download complete")
EOF
}

shard_safetensors() {
    local model_dir="$1"
    local num_shards="$2"

    log "Sharding safetensors in $model_dir into $num_shards parts..."

    python3 << EOF
from safetensors.torch import load_file, save_file
import os

model_dir = "$model_dir"
num_shards = $num_shards
safetensors_path = os.path.join(model_dir, "model.safetensors")

if not os.path.exists(safetensors_path):
    print(f"No model.safetensors found in {model_dir}")
    exit(1)

print(f"Loading {safetensors_path}...")
tensors = load_file(safetensors_path)
print(f"Loaded {len(tensors)} tensors")

# Split by tensor name (sorted for reproducibility)
keys = sorted(tensors.keys())
chunk_size = len(keys) // num_shards

for i in range(num_shards):
    start = i * chunk_size
    end = len(keys) if i == num_shards - 1 else (i + 1) * chunk_size
    shard = {k: tensors[k] for k in keys[start:end]}
    shard_path = os.path.join(model_dir, f"model-{i+1:05d}-of-{num_shards:05d}.safetensors")
    save_file(shard, shard_path)
    size_mb = os.path.getsize(shard_path) / 1024 / 1024
    print(f"  Shard {i+1}: {len(shard)} tensors, {size_mb:.1f} MB")

# Remove original to save space
os.remove(safetensors_path)
print("Sharding complete (original removed)")
EOF
}

build_image() {
    local model_dir="$1"
    local image_tag="$2"
    local sharded="$3"

    log "Building image $image_tag (sharded=$sharded)..."

    # Create a temporary Modelfile
    local modelfile=$(mktemp)

    if [[ "$sharded" == "true" ]]; then
        # Find all shard files
        local shards=$(ls "$model_dir"/model-*.safetensors 2>/dev/null | sort)
        if [[ -z "$shards" ]]; then
            log "ERROR: No shard files found in $model_dir"
            exit 1
        fi

        cat > "$modelfile" << 'EOF'
FROM scratch

# Config files (small, enables config-only updates)
COPY config.json /
COPY video_preprocessor_config.json /
EOF

        # Add each shard as a separate layer
        for shard in $shards; do
            echo "COPY $(basename "$shard") /" >> "$modelfile"
        done
    else
        cat > "$modelfile" << 'EOF'
FROM scratch

# Config files
COPY config.json /
COPY video_preprocessor_config.json /

# Model weights (single layer)
COPY model.safetensors /
EOF
    fi

    log "Modelfile:"
    cat "$modelfile" | sed 's/^/  /'

    # Build with:
    # - --disable-compression: safetensors don't compress, avoid overhead
    # - --platform linux/amd64: consistent architecture for data-only images
    podman build \
        --disable-compression \
        --platform linux/amd64 \
        -f "$modelfile" \
        -t "$image_tag" \
        "$model_dir"

    rm "$modelfile"
    log "Build complete: $image_tag"
}

push_image() {
    local image_tag="$1"

    log "Pushing $image_tag..."
    podman push "$image_tag"
    log "Push complete: $image_tag"
}

verify_image() {
    local image_tag="$1"

    log "Verifying $image_tag..."

    skopeo inspect --raw "docker://$image_tag" 2>/dev/null | python3 -c "
import json, sys
m = json.load(sys.stdin)
layers = m.get('layers', [])
print(f'  Layers: {len(layers)}')
total_mb = 0
for i, l in enumerate(layers):
    media = l.get('mediaType', '?').split('.')[-1]
    size_mb = l.get('size', 0) / 1024 / 1024
    total_mb += size_mb
    print(f'    Layer {i+1}: {media}, {size_mb:.1f} MB')
print(f'  Total: {total_mb:.1f} MB')
" || log "  (skopeo inspect failed, image may not be pushed yet)"
}

build_model() {
    local model_name="$1"
    local hf_id="${MODELS[$model_name]}"
    local model_dir="$WORK_DIR/$model_name"
    local image_tag="$REGISTRY/vjepa2-model-$model_name:latest"
    local sharded="false"

    log "=========================================="
    log "Building $model_name"
    log "  HuggingFace: $hf_id"
    log "  Image: $image_tag"
    log "  Sharding: ${SHARD:-disabled}"
    log "=========================================="

    mkdir -p "$model_dir"

    # Step 1: Download from HuggingFace
    if [[ ! -f "$model_dir/model.safetensors" ]] && [[ ! -f "$model_dir/model-00001-of-00004.safetensors" ]]; then
        download_model "$hf_id" "$model_dir"
    else
        log "Model already downloaded, skipping..."
    fi

    # Step 2: Optionally shard safetensors
    if [[ -n "$SHARD" ]]; then
        if [[ -f "$model_dir/model.safetensors" ]]; then
            shard_safetensors "$model_dir" "$NUM_SHARDS"
        else
            log "Shards already exist, skipping..."
        fi
        sharded="true"
    fi

    # Step 3: Build OCI image
    build_image "$model_dir" "$image_tag" "$sharded"

    # Step 4: Push to registry
    if [[ "$NO_PUSH" == "false" ]]; then
        push_image "$image_tag"
        verify_image "$image_tag"
    else
        log "Skipping push (--no-push)"
    fi

    log "=========================================="
    log "$model_name complete!"
    log "=========================================="
}

main() {
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --shard)
                SHARD="true"
                if [[ "${2:-}" =~ ^[0-9]+$ ]]; then
                    NUM_SHARDS="$2"
                    shift
                fi
                shift
                ;;
            --registry)
                REGISTRY="$2"
                shift 2
                ;;
            --work-dir)
                WORK_DIR="$2"
                shift 2
                ;;
            --no-push)
                NO_PUSH=true
                shift
                ;;
            --help|-h)
                usage
                ;;
            vitl|vitg|all)
                TARGET="$1"
                shift
                ;;
            *)
                echo "Unknown option: $1"
                usage
                ;;
        esac
    done

    log "V-JEPA2 Model Image Builder"
    log "  Work directory: $WORK_DIR"
    log "  Registry: $REGISTRY"
    log "  Sharding: ${SHARD:-disabled}${SHARD:+ ($NUM_SHARDS shards)}"
    log "  Push: $([ "$NO_PUSH" == "true" ] && echo "disabled" || echo "enabled")"

    mkdir -p "$WORK_DIR"

    case "$TARGET" in
        vitl)
            build_model vitl
            ;;
        vitg)
            build_model vitg
            ;;
        all)
            build_model vitl
            build_model vitg
            ;;
    esac

    log "All done!"
}

main "$@"
