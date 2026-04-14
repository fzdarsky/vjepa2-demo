#!/bin/bash
set -euo pipefail

# V-JEPA2 EC2 Deployment Script
# Deploys the demo using pre-built RHEL 10 bootc AMI
# Supports both GPU (CUDA) and CPU inference modes

# Configuration (override with environment variables)
AWS_REGION="${AWS_REGION:-us-east-2}"
AMI_ID="${AMI_ID:-ami-0d520561fae3e3308}"  # RHEL 10 bootc with NVIDIA drivers
INSTANCE_TYPE="${INSTANCE_TYPE:-g6.xlarge}"
KEY_NAME="${KEY_NAME:-aws-dev}"
SECURITY_GROUP_NAME="${SECURITY_GROUP_NAME:-vjepa2-demo}"

# Acceleration: "auto" (detect from instance type), "cuda", or "cpu"
ACCELERATION="${ACCELERATION:-auto}"

# Container images
REGISTRY="${REGISTRY:-quay.io/fzdarsky}"

usage() {
    cat <<EOF
Usage: $0 <command>

Commands:
    launch    Launch a new instance
    status    Show instance status and URLs
    ssh       SSH into the instance
    stop      Stop the instance (preserves data)
    start     Start a stopped instance
    destroy   Terminate the instance

Environment variables:
    AWS_REGION      AWS region (default: us-east-2)
    INSTANCE_TYPE   EC2 instance type (default: g6.xlarge)
    KEY_NAME        SSH key pair name (default: aws-dev)
    ACCELERATION    "auto", "cuda", or "cpu" (default: auto)
                    - auto: use CUDA on GPU instances, CPU otherwise
                    - cuda: force CUDA (requires GPU instance)
                    - cpu: force CPU (for A/B testing on GPU instances)

Examples:
    # Launch with auto-detection (CUDA on g6.xlarge)
    $0 launch

    # Launch CPU-only instance
    INSTANCE_TYPE=m5.xlarge $0 launch

    # Launch GPU instance but use CPU inference (A/B testing)
    ACCELERATION=cpu $0 launch
EOF
    exit 1
}

# Check if instance type has GPU
is_gpu_instance() {
    local type="$1"
    local family="${type%%.*}"
    [[ "$family" =~ ^(g4dn|g5|g6|p3|p4d|p4de|p5)$ ]]
}

# Determine effective acceleration mode
get_acceleration() {
    local instance_type="$1"
    local requested="$2"

    if [[ "$requested" == "auto" ]]; then
        if is_gpu_instance "$instance_type"; then
            echo "cuda"
        else
            echo "cpu"
        fi
    else
        echo "$requested"
    fi
}

get_instance_id() {
    aws ec2 describe-instances \
        --region "$AWS_REGION" \
        --filters "Name=tag:Name,Values=vjepa2-demo" "Name=instance-state-name,Values=pending,running,stopping,stopped" \
        --query "Reservations[0].Instances[0].InstanceId" \
        --output text 2>/dev/null | grep -v "^None$" || true
}

get_instance_info() {
    aws ec2 describe-instances \
        --region "$AWS_REGION" \
        --instance-ids "$1" \
        --query "Reservations[0].Instances[0].{State:State.Name,PublicIp:PublicIpAddress,InstanceType:InstanceType}" \
        --output json
}

ensure_security_group() {
    local sg_id
    sg_id=$(aws ec2 describe-security-groups \
        --region "$AWS_REGION" \
        --filters "Name=group-name,Values=$SECURITY_GROUP_NAME" \
        --query "SecurityGroups[0].GroupId" \
        --output text 2>/dev/null | grep -v "^None$" || true)

    if [[ -z "$sg_id" ]]; then
        echo "Creating security group: $SECURITY_GROUP_NAME" >&2
        sg_id=$(aws ec2 create-security-group \
            --region "$AWS_REGION" \
            --group-name "$SECURITY_GROUP_NAME" \
            --description "V-JEPA2 demo access" \
            --query "GroupId" \
            --output text)

        # SSH, API, Grafana, Prometheus, Jaeger
        for port in 22 8080 3000 9090 16686; do
            aws ec2 authorize-security-group-ingress \
                --region "$AWS_REGION" \
                --group-id "$sg_id" \
                --protocol tcp \
                --port "$port" \
                --cidr 0.0.0.0/0 >/dev/null
        done
        echo "Security group created: $sg_id" >&2
    fi
    echo "$sg_id"
}

generate_user_data() {
    local accel="$1"
    local server_image="${REGISTRY}/vjepa2-server-${accel}:latest"
    local model_image="${REGISTRY}/vjepa2-model-vitl:latest"

    # Compose profiles based on acceleration
    local profiles="observability"
    if [[ "$accel" == "cuda" ]]; then
        profiles="cuda,observability,gpu-metrics"
    else
        profiles="cpu,observability"
    fi

    cat <<USERDATA
#!/bin/bash
set -euxo pipefail
exec > >(tee -a /var/log/vjepa2-deploy.log) 2>&1

echo "=== V-JEPA2 Deployment ==="
echo "Acceleration: ${accel}"
date

USERDATA

    # GPU setup only for CUDA mode
    if [[ "$accel" == "cuda" ]]; then
        cat <<'USERDATA'
# Verify NVIDIA drivers
nvidia-smi

# Generate CDI spec for GPU passthrough
nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
USERDATA
    else
        cat <<'USERDATA'
echo "CPU-only mode - skipping NVIDIA setup"
USERDATA
    fi

    cat <<USERDATA

# Clone repo
cd /home/ec2-user
git clone https://github.com/fzdarsky/vjepa2-demo.git
cd vjepa2-demo
chown -R ec2-user:ec2-user /home/ec2-user/vjepa2-demo

# Pull images
podman pull ${server_image}
podman pull ${model_image}
podman pull docker.io/otel/opentelemetry-collector-contrib:latest
podman pull docker.io/prom/prometheus:latest
podman pull docker.io/jaegertracing/jaeger:latest
podman pull docker.io/grafana/grafana:latest
USERDATA

    if [[ "$accel" == "cuda" ]]; then
        echo "podman pull docker.io/nvidia/dcgm-exporter:latest"
    fi

    cat <<USERDATA

# Create model volume
podman volume create --driver image --opt image=${model_image} vjepa2-model-vitl

# Start services
cd /home/ec2-user/vjepa2-demo
USERDATA

    # Generate profile flags
    echo "podman compose --profile ${profiles//,/ --profile } up -d"

    cat <<'USERDATA'

# Wait for server
echo "Waiting for server to be ready..."
for i in {1..60}; do
    if curl -sf http://localhost:8080/v2/health/ready; then
        echo "Server is ready!"
        break
    fi
    sleep 5
done

echo "=== Deployment Complete ==="
PUBLIC_IP=$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4)
echo "Server:     http://$PUBLIC_IP:8080"
echo "Grafana:    http://$PUBLIC_IP:3000"
echo "Jaeger:     http://$PUBLIC_IP:16686"
echo "Prometheus: http://$PUBLIC_IP:9090"
USERDATA

    if [[ "$accel" == "cuda" ]]; then
        echo "nvidia-smi"
    fi
    echo "podman ps"
}

cmd_launch() {
    local existing
    existing=$(get_instance_id)
    if [[ -n "$existing" ]]; then
        echo "Instance already exists: $existing"
        echo "Use '$0 destroy' first, or '$0 start' if stopped"
        exit 1
    fi

    local accel
    accel=$(get_acceleration "$INSTANCE_TYPE" "$ACCELERATION")

    # Validate CUDA on GPU instance
    if [[ "$accel" == "cuda" ]] && ! is_gpu_instance "$INSTANCE_TYPE"; then
        echo "Error: ACCELERATION=cuda requires a GPU instance type"
        echo "GPU types: g4dn.*, g5.*, g6.*, p3.*, p4d.*, p5.*"
        exit 1
    fi

    echo "Configuration:"
    echo "  Instance type: $INSTANCE_TYPE"
    echo "  Acceleration:  $accel"
    echo "  AMI:           $AMI_ID"
    echo ""

    local sg_id
    sg_id=$(ensure_security_group)

    echo "Launching instance..."
    local instance_id
    instance_id=$(aws ec2 run-instances \
        --region "$AWS_REGION" \
        --image-id "$AMI_ID" \
        --instance-type "$INSTANCE_TYPE" \
        --key-name "$KEY_NAME" \
        --security-group-ids "$sg_id" \
        --user-data "$(generate_user_data "$accel")" \
        --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=vjepa2-demo},{Key=Acceleration,Value=$accel}]" \
        --block-device-mappings "[{\"DeviceName\":\"/dev/sda1\",\"Ebs\":{\"VolumeSize\":100,\"VolumeType\":\"gp3\"}}]" \
        --query "Instances[0].InstanceId" \
        --output text)

    echo "Instance launched: $instance_id"
    echo "Waiting for public IP..."

    local public_ip=""
    for i in {1..30}; do
        public_ip=$(aws ec2 describe-instances \
            --region "$AWS_REGION" \
            --instance-ids "$instance_id" \
            --query "Reservations[0].Instances[0].PublicIpAddress" \
            --output text 2>/dev/null | grep -v "^None$" || true)
        [[ -n "$public_ip" ]] && break
        sleep 2
    done

    cat <<EOF

=== Instance Ready ===
Instance ID:  $instance_id
Public IP:    $public_ip
Acceleration: $accel

SSH:        ssh -i ~/.ssh/${KEY_NAME}.pem ec2-user@$public_ip
Deploy log: ssh -i ~/.ssh/${KEY_NAME}.pem ec2-user@$public_ip 'sudo tail -f /var/log/vjepa2-deploy.log'

After cloud-init completes (~3 min):
  Inference: http://$public_ip:8080
  Grafana:   http://$public_ip:3000
  Jaeger:    http://$public_ip:16686
EOF
}

cmd_status() {
    local instance_id
    instance_id=$(get_instance_id)
    if [[ -z "$instance_id" ]]; then
        echo "No vjepa2-demo instance found"
        exit 0
    fi

    local info
    info=$(get_instance_info "$instance_id")
    local state public_ip instance_type
    state=$(echo "$info" | jq -r '.State')
    public_ip=$(echo "$info" | jq -r '.PublicIp // "N/A"')
    instance_type=$(echo "$info" | jq -r '.InstanceType')

    # Get acceleration tag
    local accel
    accel=$(aws ec2 describe-tags \
        --region "$AWS_REGION" \
        --filters "Name=resource-id,Values=$instance_id" "Name=key,Values=Acceleration" \
        --query "Tags[0].Value" \
        --output text 2>/dev/null | grep -v "^None$" || echo "unknown")

    echo "Instance:     $instance_id"
    echo "State:        $state"
    echo "Type:         $instance_type"
    echo "Acceleration: $accel"
    echo "IP:           $public_ip"

    if [[ "$state" == "running" && "$public_ip" != "N/A" ]]; then
        cat <<EOF

URLs:
  Inference: http://$public_ip:8080
  Grafana:   http://$public_ip:3000
  Jaeger:    http://$public_ip:16686
EOF
    fi
}

cmd_ssh() {
    local instance_id
    instance_id=$(get_instance_id)
    [[ -z "$instance_id" ]] && { echo "No instance found"; exit 1; }

    local public_ip
    public_ip=$(aws ec2 describe-instances \
        --region "$AWS_REGION" \
        --instance-ids "$instance_id" \
        --query "Reservations[0].Instances[0].PublicIpAddress" \
        --output text)

    [[ "$public_ip" == "None" ]] && { echo "Instance not running"; exit 1; }

    exec ssh -i ~/.ssh/${KEY_NAME}.pem ec2-user@"$public_ip"
}

cmd_stop() {
    local instance_id
    instance_id=$(get_instance_id)
    [[ -z "$instance_id" ]] && { echo "No instance found"; exit 1; }

    echo "Stopping $instance_id..."
    aws ec2 stop-instances --region "$AWS_REGION" --instance-ids "$instance_id" >/dev/null
    echo "Instance stopping"
}

cmd_start() {
    local instance_id
    instance_id=$(get_instance_id)
    [[ -z "$instance_id" ]] && { echo "No instance found"; exit 1; }

    echo "Starting $instance_id..."
    aws ec2 start-instances --region "$AWS_REGION" --instance-ids "$instance_id" >/dev/null
    echo "Instance starting. Run '$0 status' to get new IP."
}

cmd_destroy() {
    local instance_id
    instance_id=$(get_instance_id)
    [[ -z "$instance_id" ]] && { echo "No instance found"; exit 0; }

    echo "Terminating $instance_id..."
    aws ec2 terminate-instances --region "$AWS_REGION" --instance-ids "$instance_id" >/dev/null
    echo "Instance terminated"
}

# Main
[[ $# -lt 1 ]] && usage

case "$1" in
    launch)  cmd_launch ;;
    status)  cmd_status ;;
    ssh)     cmd_ssh ;;
    stop)    cmd_stop ;;
    start)   cmd_start ;;
    destroy) cmd_destroy ;;
    *)       usage ;;
esac
