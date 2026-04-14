# V-JEPA2 AWS EC2 Deployment

Deploy the V-JEPA2 inference server to AWS EC2 with full observability stack.

## Prerequisites

- AWS CLI configured with appropriate credentials
- An existing EC2 key pair (default: `aws-dev`)
- `jq` installed for parsing JSON output

## Quick Start

```bash
cd deploy/aws-ec2

# GPU deployment (default: g6.xlarge with L4 GPU)
./deploy.sh launch

# CPU-only deployment
INSTANCE_TYPE=m5.xlarge ./deploy.sh launch

# GPU instance with CPU inference (A/B testing)
ACCELERATION=cpu ./deploy.sh launch

# Check status
./deploy.sh status

# SSH into instance
./deploy.sh ssh

# Tear down
./deploy.sh destroy
```

## Commands

| Command | Description |
|---------|-------------|
| `launch` | Launch a new instance |
| `status` | Show instance status and URLs |
| `ssh` | SSH into the instance |
| `stop` | Stop the instance (preserves data) |
| `start` | Start a stopped instance |
| `destroy` | Terminate the instance |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `AWS_REGION` | `us-east-2` | AWS region |
| `INSTANCE_TYPE` | `g6.xlarge` | EC2 instance type |
| `KEY_NAME` | `aws-dev` | SSH key pair name |
| `ACCELERATION` | `auto` | `auto`/`cpu`/`cuda` |
| `AMI_ID` | (RHEL 10 bootc) | Custom AMI ID |
| `REGISTRY` | `quay.io/fzdarsky` | Container registry |

## Acceleration Modes

| Mode | Description |
|------|-------------|
| `auto` | Use CUDA on GPU instances, CPU otherwise |
| `cuda` | Force CUDA (fails on non-GPU instance) |
| `cpu` | Force CPU (useful for A/B testing on GPU instances) |

GPU instance families: g4dn, g5, g6, p3, p4d, p4de, p5

## Endpoints

After deployment (~3 minutes with custom AMI):

| Service | Port | URL |
|---------|------|-----|
| Inference API | 8080 | `http://<public-ip>:8080/v2/models/vjepa2/infer` |
| Grafana | 3000 | `http://<public-ip>:3000` |
| Jaeger | 16686 | `http://<public-ip>:16686` |
| Prometheus | 9090 | `http://<public-ip>:9090` |

## GPU Instance Options

| Instance | GPU | VRAM | vCPU | RAM | Cost/hr | Notes |
|----------|-----|------|------|-----|---------|-------|
| g4dn.xlarge | T4 (Turing) | 16GB | 4 | 16GB | ~$0.53 | Budget option |
| g6.xlarge | L4 (Ada) | 24GB | 4 | 16GB | ~$0.80 | Recommended |
| g6.4xlarge | L4 (Ada) | 24GB | 16 | 64GB | ~$1.32 | For parallel benchmarks |

## Custom AMI

The deployment uses a custom RHEL 10 bootc AMI with pre-installed NVIDIA drivers. To build your own:

```bash
# See ../bootc/README.md for build instructions
cd ../bootc
./build-ami.sh

# Deploy with your AMI
cd ../aws-ec2
AMI_ID=ami-0123456789abcdef0 ./deploy.sh launch
```

Benefits:
- **Fast boot** (~3 min vs ~15 min with runtime driver install)
- **No kernel mismatches** after stop/start cycles
- **Reliable** — no network dependencies for driver installation

## Monitoring Deployment

```bash
# Watch deployment progress
./deploy.sh ssh
sudo tail -f /var/log/vjepa2-deploy.log

# Check container status
podman ps

# Check GPU status
nvidia-smi
```

## Architecture

The deployment includes:

1. **V-JEPA2 Inference Server** - Video action recognition API
2. **OpenTelemetry Collector** - Telemetry collection and routing
3. **Prometheus** - Metrics storage
4. **Jaeger** - Distributed tracing
5. **Grafana** - Pre-configured dashboards

All services run as Podman containers orchestrated via `podman compose`.

## Troubleshooting

### Server not ready after 5 minutes

Check the deployment log:
```bash
./deploy.sh ssh
sudo cat /var/log/vjepa2-deploy.log
```

### GPU not detected

Verify NVIDIA drivers loaded:
```bash
nvidia-smi
```

Check CDI configuration:
```bash
cat /etc/cdi/nvidia.yaml
```

### Container issues

Check container logs:
```bash
podman logs vjepa2-server-cuda
# or for CPU mode:
podman logs vjepa2-server-cpu
```
