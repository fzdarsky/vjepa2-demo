# EC2 GPU Test Infrastructure

Terraform configuration for spinning up an NVIDIA GPU instance to validate the CUDA container variant.

## Prerequisites

- AWS CLI configured with credentials
- Terraform >= 1.5
- An existing EC2 key pair in the target region
- Red Hat registry credentials (`registry.redhat.io`)

## Usage

```bash
cd tests/ec2-gpu

terraform init

terraform apply \
  -var="key_name=my-key-pair" \
  -var="rh_registry_user=your-username" \
  -var="rh_registry_pass=your-password"
```

## Monitoring Progress

Cloud-init runs the full setup (NVIDIA drivers, Podman, container build, inference test). This takes 10-15 minutes.

```bash
# Watch cloud-init progress
ssh ec2-user@<public-ip> 'sudo tail -f /var/log/cloud-init-output.log'

# Check the test log
ssh ec2-user@<public-ip> 'cat /var/log/vjepa2-gpu-test.log'
```

## Tear Down

```bash
terraform destroy \
  -var="key_name=my-key-pair" \
  -var="rh_registry_user=x" \
  -var="rh_registry_pass=x"
```

## Cost

- `g4dn.xlarge`: ~$0.53/hr (1x NVIDIA T4, 16GB VRAM)
- Typical test cycle: under 15 minutes (~$0.13)
- **Remember to `terraform destroy` when done**

## Troubleshooting

| Issue | Fix |
| ----- | --- |
| `InsufficientInstanceCapacity` | Try a different AZ: `-var="aws_region=us-west-2"` |
| `VcpuLimitExceeded` | Request a G instance quota increase in AWS console |
| NVIDIA driver fails to load | Check kernel version matches driver: `uname -r` |
| `podman login` fails | Verify Red Hat registry credentials at `access.redhat.com` |
