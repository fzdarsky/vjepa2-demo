#!/bin/bash
set -euo pipefail

# AWS EC2 VM Lifecycle
# Launch, stop, start, destroy, ssh, status for bootc-based GPU instances.
# Does NOT deploy services — use ../demo.sh for that.

AWS_REGION="${AWS_REGION:-eu-central-1}"
AWS_AZ="${AWS_AZ:-}"

get_default_ami() {
    case "$1" in
        us-east-2)     echo "ami-0d8baa6402e6ad095" ;;
        eu-central-1)  echo "ami-08f8a3356f7b9ff20" ;;
        eu-west-1)     echo "ami-0b352dc05b510010f" ;;
        *)             echo "" ;;
    esac
}
AMI_ID="${AMI_ID:-$(get_default_ami "$AWS_REGION")}"

INSTANCE_TYPE="${INSTANCE_TYPE:-g6.xlarge}"
KEY_NAME="${KEY_NAME:-aws-dev}"
SECURITY_GROUP_NAME="${SECURITY_GROUP_NAME:-vjepa2-demo}"
USE_SPOT="${USE_SPOT:-true}"
SPOT_MAX_PRICE="${SPOT_MAX_PRICE:-}"
INSTANCE_TAG="${INSTANCE_TAG:-vjepa2-demo}"

usage() {
    cat <<EOF
Usage: $0 <command>

Commands:
    launch    Launch a new EC2 instance (bootc AMI, no service deployment)
    status    Show instance status and IP
    ssh       SSH into the instance
    stop      Stop the instance (preserves data)
    start     Start a stopped instance
    destroy   Terminate the instance and cancel spot requests

Environment variables:
    AWS_REGION      AWS region (default: eu-central-1)
    AWS_AZ          Availability zone (default: auto)
    AMI_ID          bootc AMI ID (per-region defaults available)
    INSTANCE_TYPE   EC2 instance type (default: g6.xlarge)
    KEY_NAME        SSH key pair name (default: aws-dev)
    USE_SPOT        "true" or "false" (default: true)
    SPOT_MAX_PRICE  Max hourly price (default: on-demand cap)
    INSTANCE_TAG    Name tag for the instance (default: vjepa2-demo)

After launch, use demo.sh to deploy services:
    ../demo.sh setup
EOF
    exit 1
}

is_gpu_instance() {
    local family="${1%%.*}"
    [[ "$family" =~ ^(g4dn|g5|g6|p3|p4d|p4de|p5)$ ]]
}

get_instance_id() {
    aws ec2 describe-instances \
        --region "$AWS_REGION" \
        --filters "Name=tag:Name,Values=$INSTANCE_TAG" "Name=instance-state-name,Values=pending,running,stopping,stopped" \
        --query "Reservations[0].Instances[0].InstanceId" \
        --output text 2>/dev/null | grep -v "^None$" || true
}

get_public_ip() {
    aws ec2 describe-instances \
        --region "$AWS_REGION" \
        --instance-ids "$1" \
        --query "Reservations[0].Instances[0].PublicIpAddress" \
        --output text 2>/dev/null | grep -v "^None$" || true
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

        for port in 22 8443 3000 9090 16686; do
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
    cat <<'USERDATA'
#!/bin/bash
set -uxo pipefail
exec > >(tee -a /var/log/ec2-init.log) 2>&1

echo "=== EC2 Host Initialization ==="
date

# Harden sshd against resource starvation during heavy container pulls
mkdir -p /etc/ssh/sshd_config.d/
cat > /etc/ssh/sshd_config.d/99-hardening.conf <<'SSHEOF'
MaxStartups 100:30:200
MaxSessions 100
ClientAliveInterval 30
ClientAliveCountMax 3
PerSourceMaxStartups 15
SSHEOF

mkdir -p /etc/systemd/system/sshd.service.d/
cat > /etc/systemd/system/sshd.service.d/override.conf <<'SSHEOF'
[Service]
OOMScoreAdjust=-500
CPUWeight=200
IOWeight=200
TasksMax=150
MemoryMax=512M
LimitNOFILE=65535
SSHEOF

systemctl daemon-reload
systemctl restart sshd

systemctl stop firewalld 2>/dev/null || true
systemctl disable firewalld 2>/dev/null || true

mkdir -p /home/ec2-user/demo
chown -R ec2-user:ec2-user /home/ec2-user/demo

# Wait for CDI spec on GPU instances
if command -v nvidia-smi &>/dev/null; then
    echo "GPU detected, waiting for CDI spec..."
    nvidia-smi || true
    for i in {1..30}; do
        [[ -f /etc/cdi/nvidia.yaml ]] && { echo "CDI spec ready"; break; }
        sleep 2
    done
fi

echo "=== Host initialization complete ==="
USERDATA
}

cmd_launch() {
    [[ -z "$AMI_ID" ]] && { echo "Error: AMI_ID must be set (no default for region $AWS_REGION)"; exit 1; }

    local existing
    existing=$(get_instance_id)
    [[ -n "$existing" ]] && { echo "Instance already exists: $existing. Use 'destroy' first or 'start' if stopped."; exit 1; }

    local accel="cpu"
    is_gpu_instance "$INSTANCE_TYPE" && accel="cuda"

    echo "Launching EC2 instance:"
    echo "  Region:   $AWS_REGION"
    echo "  AZ:       ${AWS_AZ:-auto}"
    echo "  Type:     $INSTANCE_TYPE ($accel)"
    echo "  Spot:     $USE_SPOT"
    echo "  AMI:      $AMI_ID"

    local sg_id
    sg_id=$(ensure_security_group)

    local run_args=(
        --region "$AWS_REGION"
        --image-id "$AMI_ID"
        --instance-type "$INSTANCE_TYPE"
        --key-name "$KEY_NAME"
        --security-group-ids "$sg_id"
        --user-data "$(generate_user_data)"
        --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$INSTANCE_TAG},{Key=Acceleration,Value=$accel}]"
        --block-device-mappings "[{\"DeviceName\":\"/dev/sda1\",\"Ebs\":{\"VolumeSize\":100,\"VolumeType\":\"gp3\"}}]"
        --query "Instances[0].InstanceId"
        --output text
    )

    [[ -n "$AWS_AZ" ]] && run_args+=(--placement "AvailabilityZone=$AWS_AZ")

    if [[ "$USE_SPOT" == "true" ]]; then
        local spot_opts="SpotInstanceType=persistent,InstanceInterruptionBehavior=stop"
        [[ -n "$SPOT_MAX_PRICE" ]] && spot_opts="${spot_opts},MaxPrice=${SPOT_MAX_PRICE}"
        run_args+=(--instance-market-options "MarketType=spot,SpotOptions={${spot_opts}}")
    fi

    local instance_id
    instance_id=$(aws ec2 run-instances "${run_args[@]}")
    echo "Instance launched: $instance_id"

    if [[ "$USE_SPOT" == "true" ]]; then
        local spot_request_id
        spot_request_id=$(aws ec2 describe-spot-instance-requests \
            --region "$AWS_REGION" \
            --filters "Name=instance-id,Values=$instance_id" \
            --query "SpotInstanceRequests[0].SpotInstanceRequestId" \
            --output text 2>/dev/null | grep -v "^None$" || true)
        [[ -n "$spot_request_id" ]] && aws ec2 create-tags \
            --region "$AWS_REGION" \
            --resources "$spot_request_id" \
            --tags "Key=Name,Value=$INSTANCE_TAG" >/dev/null 2>&1 || true
    fi

    echo "Waiting for public IP..."
    local public_ip=""
    for _ in {1..30}; do
        public_ip=$(get_public_ip "$instance_id")
        [[ -n "$public_ip" ]] && break
        sleep 2
    done

    cat <<EOF

=== Instance Launched ===
Instance ID:  $instance_id
Public IP:    $public_ip
Type:         $INSTANCE_TYPE ($accel)

Next: deploy services with demo.sh:
  ../demo.sh setup
EOF
}

cmd_status() {
    local instance_id
    instance_id=$(get_instance_id)
    [[ -z "$instance_id" ]] && { echo "No $INSTANCE_TAG instance found"; exit 0; }

    local info
    info=$(aws ec2 describe-instances \
        --region "$AWS_REGION" \
        --instance-ids "$instance_id" \
        --query "Reservations[0].Instances[0].{State:State.Name,PublicIp:PublicIpAddress,InstanceType:InstanceType}" \
        --output json)

    local state public_ip instance_type
    state=$(echo "$info" | jq -r '.State')
    public_ip=$(echo "$info" | jq -r '.PublicIp // "N/A"')
    instance_type=$(echo "$info" | jq -r '.InstanceType')

    echo "Instance:  $instance_id"
    echo "State:     $state"
    echo "Type:      $instance_type"
    echo "IP:        $public_ip"
}

cmd_ssh() {
    local instance_id
    instance_id=$(get_instance_id)
    [[ -z "$instance_id" ]] && { echo "No instance found"; exit 1; }

    local public_ip
    public_ip=$(get_public_ip "$instance_id")
    [[ -z "$public_ip" ]] && { echo "Instance not running"; exit 1; }

    exec ssh -i ~/.ssh/"${KEY_NAME}".pem ec2-user@"$public_ip"
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

    local spot_ids
    spot_ids=$(aws ec2 describe-spot-instance-requests \
        --region "$AWS_REGION" \
        --filters "Name=state,Values=open,active" \
        --query "SpotInstanceRequests[?Tags[?Key=='Name' && Value=='$INSTANCE_TAG']].SpotInstanceRequestId" \
        --output text 2>/dev/null || true)
    if [[ -n "$instance_id" ]]; then
        local instance_spot
        instance_spot=$(aws ec2 describe-spot-instance-requests \
            --region "$AWS_REGION" \
            --filters "Name=instance-id,Values=$instance_id" \
            --query "SpotInstanceRequests[*].SpotInstanceRequestId" \
            --output text 2>/dev/null || true)
        spot_ids=$(echo "$spot_ids $instance_spot" | tr ' ' '\n' | grep -v '^$' | sort -u | tr '\n' ' ')
    fi
    if [[ -n "${spot_ids// /}" ]]; then
        echo "Cancelling spot requests: $spot_ids"
        aws ec2 cancel-spot-instance-requests --region "$AWS_REGION" --spot-instance-request-ids "$spot_ids" >/dev/null 2>&1 || true
    fi

    [[ -z "$instance_id" ]] && { echo "No instance found"; exit 0; }

    echo "Terminating $instance_id..."
    aws ec2 terminate-instances --region "$AWS_REGION" --instance-ids "$instance_id" >/dev/null
    echo "Instance terminated"
}

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
