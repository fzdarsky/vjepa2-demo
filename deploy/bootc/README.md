# RHEL 9 GPU Bootc Image

Build a custom RHEL 9 AMI with pre-installed NVIDIA drivers using bootc-image-builder.

## Why?

Runtime NVIDIA driver installation on EC2 is problematic:
- Slow (~15 min for DKMS compilation)
- Kernel mismatches after stop/start require rebuilds
- Network timeouts cause deployment failures

This approach bakes NVIDIA drivers into the AMI at build time.

## Prerequisites

1. **Red Hat registry access** - `registry.redhat.io` requires RHEL subscription
   ```bash
   podman login registry.redhat.io
   ```

2. **S3 bucket** - Create bucket for AMI upload
   ```bash
   aws s3 mb s3://vjepa2-bootc-images --region us-east-2
   ```

3. **IAM vmimport role** - Required for EC2 AMI import (see below)

## Quick Start

```bash
cd deploy/bootc

# Build and push bootc container
podman build -f Containerfile.nv -t quay.io/fzdarsky/rhel9-nv-bootc:latest .
podman push quay.io/fzdarsky/rhel9-nv-bootc:latest

# Build AMI (requires S3 bucket and vmimport role)
./build-ami.sh
```

## AWS vmimport Role Setup

```bash
# Create trust policy
cat > trust-policy.json <<EOF
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": {"Service": "vmie.amazonaws.com"},
    "Action": "sts:AssumeRole",
    "Condition": {"StringEquals": {"sts:Externalid": "vmimport"}}
  }]
}
EOF

aws iam create-role --role-name vmimport \
  --assume-role-policy-document file://trust-policy.json

# Attach policy
cat > role-policy.json <<EOF
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Action": ["s3:GetBucketLocation", "s3:GetObject", "s3:ListBucket"],
    "Resource": ["arn:aws:s3:::vjepa2-bootc-images", "arn:aws:s3:::vjepa2-bootc-images/*"]
  }, {
    "Effect": "Allow",
    "Action": ["ec2:ModifySnapshotAttribute", "ec2:CopySnapshot", "ec2:RegisterImage", "ec2:Describe*"],
    "Resource": "*"
  }]
}
EOF

aws iam put-role-policy --role-name vmimport \
  --policy-name vmimport \
  --policy-document file://role-policy.json
```

## Using the Custom AMI

Once the AMI is built, use it with Terraform:

```bash
cd ../aws-ec2
terraform apply \
  -var="key_name=aws-dev" \
  -var="ami_id=ami-0123456789abcdef0"
```

The instance will boot with NVIDIA drivers ready — no runtime installation needed.

## Files

| File | Purpose |
|------|---------|
| `Containerfile.nv` | Bootc container with NVIDIA drivers |
| `nvidia-cdi-generate.service` | Systemd service to generate CDI spec on boot |
| `config.toml` | bootc-image-builder customizations |
| `build-ami.sh` | Build script for container and AMI |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `IMAGE_NAME` | `rhel9-nv-bootc` | Container/AMI name |
| `AWS_BUCKET` | `vjepa2-bootc-images` | S3 bucket for AMI upload |
| `AWS_REGION` | `us-east-2` | Target AWS region |
| `REGISTRY` | `quay.io/fzdarsky` | Container registry |

## Notes

- **Secure Boot**: DKMS-compiled modules are self-signed, which breaks Secure Boot
- **Kernel updates**: The driver is compiled against the specific kernel in the bootc image
- **Build time**: Container build takes ~30 min (DKMS compilation); AMI conversion adds ~15 min

## Key Implementation Details

### ext4 Filesystem (not XFS)

The AMI uses `--rootfs ext4` instead of the default XFS. XFS images suffer from LSN (Log Sequence Number) corruption during the AMI import process, causing boot failures with errors like:

```text
XFS: Corruption warning: Metadata has LSN ahead of current LSN
```

### Explicit DKMS Build

In container builds, DKMS doesn't auto-trigger module compilation. The Containerfile explicitly runs:

```dockerfile
dkms build nvidia/${VERSION} -k ${KERNEL_VERSION}
dkms install nvidia/${VERSION} -k ${KERNEL_VERSION}
```

This ensures the NVIDIA kernel module is pre-compiled for the exact kernel that will boot.

### ec2-user Configuration

bootc images default to `cloud-user`. To use the standard AWS `ec2-user`:

```dockerfile
RUN useradd -m -G wheel ec2-user && \
    echo "ec2-user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/ec2-user
RUN echo -e "system_info:\n  default_user:\n    name: ec2-user" > /etc/cloud/cloud.cfg.d/99-ec2-user.cfg
```
