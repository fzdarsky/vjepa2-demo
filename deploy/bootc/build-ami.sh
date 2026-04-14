#!/bin/bash
set -euo pipefail

# Configuration
RHEL_VERSION="${RHEL_VERSION:-10}"  # 9 or 10
AWS_BUCKET="${AWS_BUCKET:-vjepa2-bootc-images}"
AWS_REGION="${AWS_REGION:-us-east-2}"
REGISTRY="${REGISTRY:-quay.io/fzdarsky}"

# Select Containerfile and image name based on RHEL version
if [[ "${RHEL_VERSION}" == "10" ]]; then
    CONTAINERFILE="Containerfile.rhel10"
    IMAGE_NAME="${IMAGE_NAME:-rhel10-nv-bootc}"
else
    CONTAINERFILE="Containerfile.nv"
    IMAGE_NAME="${IMAGE_NAME:-rhel9-nv-bootc}"
fi

echo "=== Building RHEL ${RHEL_VERSION} bootc container ==="

# Build bootc container
podman build -f ${CONTAINERFILE} -t ${REGISTRY}/${IMAGE_NAME}:latest .
podman push ${REGISTRY}/${IMAGE_NAME}:latest

# Build AMI
# Use ext4 instead of XFS to avoid LSN corruption issues during AMI import
echo "=== Building AMI with bootc-image-builder ==="
sudo podman run --rm -it --privileged \
    --security-opt label=type:unconfined_t \
    -v ~/.aws/credentials:/root/.aws/credentials:ro \
    -v ./config.toml:/config.toml:ro \
    -v ./output:/output \
    quay.io/centos-bootc/bootc-image-builder:latest \
    --type ami \
    --rootfs ext4 \
    --aws-ami-name "${IMAGE_NAME}-$(date +%Y%m%d)" \
    --aws-bucket "${AWS_BUCKET}" \
    --aws-region "${AWS_REGION}" \
    ${REGISTRY}/${IMAGE_NAME}:latest

echo "=== AMI build complete ==="
