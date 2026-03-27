#!/bin/bash
set -euxo pipefail

LOG="/var/log/vjepa2-gpu-test.log"
exec > >(tee -a "$LOG") 2>&1

echo "=== V-JEPA2 GPU Test Setup ==="
date

# Install NVIDIA drivers
dnf install -y kernel-devel kernel-headers
dnf config-manager --add-repo \
  https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/cuda-rhel9.repo
dnf install -y nvidia-driver nvidia-driver-cuda
modprobe nvidia
nvidia-smi

# Install Podman and nvidia-container-toolkit
dnf install -y podman
dnf config-manager --add-repo \
  https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo
dnf install -y nvidia-container-toolkit
nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml

# Authenticate to Red Hat registry
podman login -u '${rh_registry_user}' -p '${rh_registry_pass}' registry.redhat.io

# Clone the repo and build the CUDA image
dnf install -y git
cd /home/ec2-user
git clone https://github.com/fzdarsky/jepa-demo.git
cd jepa-demo
podman build -t vjepa2-server-cuda -f Containerfile.cuda .

# Download model into a local directory, then run inference
mkdir -p /home/ec2-user/model
podman run --rm -v /home/ec2-user/model:/model vjepa2-server-cuda \
  download --model '${hf_model_id}'

# Run inference smoke test with GPU
echo "=== Running GPU inference smoke test ==="
podman run --rm \
  --device nvidia.com/gpu=all \
  -v /home/ec2-user/model:/model:ro \
  -v /home/ec2-user/jepa-demo/samples:/input:ro \
  -e DEVICE=cuda \
  vjepa2-server-cuda infer

echo "=== GPU Test Complete ==="
nvidia-smi
date
