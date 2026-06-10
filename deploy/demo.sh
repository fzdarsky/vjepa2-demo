#!/bin/bash
set -euo pipefail

# Demo Operations — deploy and manage V-JEPA2 services on any Linux host
# Works on EC2, DGX Spark, or any Fedora/RHEL box with podman + CUDA.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
REGISTRY="${REGISTRY:-quay.io/fzdarsky}"

# EC2 connection (used when targeting a remote host)
AWS_REGION="${AWS_REGION:-eu-central-1}"
KEY_NAME="${KEY_NAME:-aws-dev}"
INSTANCE_TAG="${INSTANCE_TAG:-vjepa2-demo}"
REMOTE_DIR="/home/ec2-user/demo"

usage() {
    cat <<EOF
Usage: $0 <command> [options]

Commands:
    setup       Rsync repo to host, generate certs, pull images, create model volume, compose up
    run         Start services (podman-compose up)
    stop        Stop services (podman-compose down)
    status      Show running containers and endpoints
    logs        Tail service logs
    redeploy    Pull new images and restart
    benchmark   Run inference benchmark

Options:
    --local     Run locally instead of on EC2 instance (default: EC2)

Environment variables:
    REGISTRY       Container registry (default: quay.io/fzdarsky)
    AWS_REGION     AWS region for EC2 lookup (default: eu-central-1)
    KEY_NAME       SSH key pair name (default: aws-dev)
    INSTANCE_TAG   Name tag to find EC2 instance (default: vjepa2-demo)

Examples:
    # Deploy to EC2 instance (launched via ec2.sh)
    ./demo.sh setup

    # Run locally on a GPU host
    ./demo.sh setup --local

    # Run benchmark
    ./demo.sh benchmark

    # Redeploy with new images
    ./demo.sh redeploy
EOF
    exit 1
}

# --- EC2 helpers ---

get_instance_id() {
    aws ec2 describe-instances \
        --region "$AWS_REGION" \
        --filters "Name=tag:Name,Values=$INSTANCE_TAG" "Name=instance-state-name,Values=running" \
        --query "Reservations[0].Instances[0].InstanceId" \
        --output text 2>/dev/null | grep -v "^None$" || true
}

get_ec2_ip() {
    local instance_id
    instance_id=$(get_instance_id)
    [[ -z "$instance_id" ]] && { echo "No running $INSTANCE_TAG instance found" >&2; exit 1; }
    aws ec2 describe-instances \
        --region "$AWS_REGION" \
        --instance-ids "$instance_id" \
        --query "Reservations[0].Instances[0].PublicIpAddress" \
        --output text 2>/dev/null | grep -v "^None$" || { echo "No public IP" >&2; exit 1; }
}

get_acceleration() {
    local instance_id
    instance_id=$(get_instance_id)
    aws ec2 describe-tags \
        --region "$AWS_REGION" \
        --filters "Name=resource-id,Values=$instance_id" "Name=key,Values=Acceleration" \
        --query "Tags[0].Value" \
        --output text 2>/dev/null | grep -v "^None$" || echo "cuda"
}

ssh_cmd() {
    local ip="$1"; shift
    ssh -o StrictHostKeyChecking=no -o ConnectTimeout=30 -o ServerAliveInterval=10 \
        -i ~/.ssh/"${KEY_NAME}".pem ec2-user@"$ip" "$@"
}

# --- Local helpers ---

detect_local_accel() {
    if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
        echo "cuda"
    else
        echo "cpu"
    fi
}

generate_cert() {
    local cert_dir="$1"
    local cn="${2:-localhost}"

    mkdir -p "$cert_dir"
    if [[ -f "$cert_dir/cert.pem" ]]; then
        echo "Certificate already exists"
        return
    fi

    local san="DNS:localhost"
    if [[ "$cn" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
        san="IP:$cn,DNS:localhost"
    fi

    openssl req -x509 -newkey rsa:4096 \
        -keyout "$cert_dir/key.pem" -out "$cert_dir/cert.pem" \
        -days 365 -nodes \
        -subj "/CN=$cn/O=V-JEPA2-Demo/C=US" \
        -addext "subjectAltName=$san" 2>/dev/null
    chmod 644 "$cert_dir"/*.pem
    echo "Certificate generated for $cn"
}

# --- Commands ---

cmd_setup_remote() {
    local ip accel
    ip=$(get_ec2_ip)
    accel=$(get_acceleration)

    echo "Setting up V-JEPA2 demo on $ip ($accel)..."

    # Wait for SSH
    echo "Waiting for SSH..."
    for i in {1..30}; do
        ssh_cmd "$ip" "true" 2>/dev/null && break
        sleep 5
    done

    # Rsync the entire repo (compose.yaml, configs, samples, benchmark scripts)
    echo "Syncing repo files..."
    rsync -avz --delete \
        -e "ssh -o StrictHostKeyChecking=no -i ~/.ssh/${KEY_NAME}.pem" \
        --exclude='certs/' \
        --exclude='.git' \
        --exclude='output/' \
        --exclude='__pycache__/' \
        "$REPO_ROOT/" "ec2-user@${ip}:${REMOTE_DIR}/"

    # Generate TLS certificate
    echo "Generating TLS certificate..."
    ssh_cmd "$ip" "
        cd ${REMOTE_DIR}
        if [[ ! -f certs/cert.pem ]]; then
            mkdir -p certs
            openssl req -x509 -newkey rsa:4096 \
                -keyout certs/key.pem -out certs/cert.pem \
                -days 365 -nodes \
                -subj '/CN=${ip}/O=V-JEPA2-Demo/C=US' \
                -addext 'subjectAltName=IP:${ip},DNS:localhost' 2>/dev/null
            chmod 644 certs/*.pem
            echo 'Certificate generated'
        else
            echo 'Certificate already exists'
        fi
    "

    # Pull images, create model volume, start services
    echo "Pulling images and starting services..."
    local profiles="--profile ${accel} --profile observability"
    [[ "$accel" == "cuda" ]] && profiles="$profiles --profile gpu-metrics"

    ssh_cmd "$ip" "
        set -e
        cd ${REMOTE_DIR}

        # Stop existing services
        sudo podman-compose down 2>/dev/null || true

        # Pull images with CPU pinning to keep SSH responsive
        echo 'Pulling images...'
        sudo taskset -c 0,1,2 podman pull ${REGISTRY}/vjepa2-server-${accel}:latest
        sudo taskset -c 0,1,2 podman pull ${REGISTRY}/vjepa2-model-vitl:latest

        # Create model volume from container image
        echo 'Creating model volume...'
        sudo podman volume rm vjepa2-model-vitl 2>/dev/null || true
        sudo podman volume create --driver image --opt image=${REGISTRY}/vjepa2-model-vitl:latest vjepa2-model-vitl

        # Start services
        echo 'Starting services...'
        sudo SSL_KEYFILE=/certs/key.pem SSL_CERTFILE=/certs/cert.pem \
            podman-compose ${profiles} up -d

        echo 'Services started'
        sudo podman ps --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'
    "

    # Wait for health
    echo "Waiting for server health..."
    for i in {1..60}; do
        if curl -sfk --connect-timeout 5 "https://$ip:8443/v2/health/ready" 2>/dev/null; then
            echo "Server is healthy!"
            break
        fi
        [[ $i -eq 60 ]] && echo "Warning: health check timed out"
        sleep 5
    done

    cat <<EOF

=== V-JEPA2 Demo Deployed ===
API:       https://$ip:8443
Jaeger:    http://$ip:16686
Grafana:   http://$ip:3000

Note: Self-signed certificate — browser will show security warning.
EOF
}

cmd_setup_local() {
    local accel
    accel=$(detect_local_accel)

    echo "Setting up V-JEPA2 demo locally ($accel)..."

    # Generate certificate
    generate_cert "$REPO_ROOT/certs" "localhost"

    # Pull images
    echo "Pulling images..."
    podman pull "${REGISTRY}/vjepa2-server-${accel}:latest"
    podman pull "${REGISTRY}/vjepa2-model-vitl:latest"

    # Create model volume
    echo "Creating model volume..."
    podman volume rm vjepa2-model-vitl 2>/dev/null || true
    podman volume create --driver image --opt image="${REGISTRY}/vjepa2-model-vitl:latest" vjepa2-model-vitl

    # Start services
    local profiles="--profile $accel --profile observability"
    [[ "$accel" == "cuda" ]] && profiles="$profiles --profile gpu-metrics"

    cd "$REPO_ROOT"
    SSL_KEYFILE=/certs/key.pem SSL_CERTFILE=/certs/cert.pem \
        podman-compose "$profiles" up -d

    echo ""
    echo "V-JEPA2 demo running locally. API: https://localhost:8443"
}

cmd_run() {
    local accel profiles
    if [[ "$LOCAL" == "true" ]]; then
        accel=$(detect_local_accel)
        profiles="--profile $accel --profile observability"
        [[ "$accel" == "cuda" ]] && profiles="$profiles --profile gpu-metrics"
        cd "$REPO_ROOT"
        SSL_KEYFILE=/certs/key.pem SSL_CERTFILE=/certs/cert.pem \
            podman-compose "$profiles" up -d
    else
        local ip
        ip=$(get_ec2_ip)
        accel=$(get_acceleration)
        profiles="--profile ${accel} --profile observability"
        [[ "$accel" == "cuda" ]] && profiles="$profiles --profile gpu-metrics"
        ssh_cmd "$ip" "cd ${REMOTE_DIR} && sudo SSL_KEYFILE=/certs/key.pem SSL_CERTFILE=/certs/cert.pem podman-compose ${profiles} up -d"
    fi
}

cmd_stop() {
    if [[ "$LOCAL" == "true" ]]; then
        cd "$REPO_ROOT"
        podman-compose down
    else
        local ip
        ip=$(get_ec2_ip)
        ssh_cmd "$ip" "cd ${REMOTE_DIR} && sudo podman-compose down"
    fi
}

cmd_status() {
    if [[ "$LOCAL" == "true" ]]; then
        podman ps --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'
    else
        local ip
        ip=$(get_ec2_ip)
        echo "Host: $ip"
        ssh_cmd "$ip" "sudo podman ps --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'"

        echo ""
        echo "URLs:"
        echo "  API:       https://$ip:8443"
        echo "  Jaeger:    http://$ip:16686"
        echo "  Grafana:   http://$ip:3000"
    fi
}

cmd_logs() {
    local service="${2:-}"
    if [[ "$LOCAL" == "true" ]]; then
        cd "$REPO_ROOT"
        podman-compose logs -f "$service"
    else
        local ip
        ip=$(get_ec2_ip)
        ssh_cmd "$ip" "cd ${REMOTE_DIR} && sudo podman-compose logs -f $service"
    fi
}

cmd_redeploy() {
    if [[ "$LOCAL" == "true" ]]; then
        local accel profiles
        accel=$(detect_local_accel)
        profiles="--profile $accel --profile observability"
        [[ "$accel" == "cuda" ]] && profiles="$profiles --profile gpu-metrics"
        cd "$REPO_ROOT"
        podman-compose down
        podman pull "${REGISTRY}/vjepa2-server-${accel}:latest"
        SSL_KEYFILE=/certs/key.pem SSL_CERTFILE=/certs/cert.pem \
            podman-compose "$profiles" up -d
    else
        local ip accel profiles
        ip=$(get_ec2_ip)
        accel=$(get_acceleration)
        profiles="--profile ${accel} --profile observability"
        [[ "$accel" == "cuda" ]] && profiles="$profiles --profile gpu-metrics"

        # Rsync updated files first
        rsync -avz --delete \
            -e "ssh -o StrictHostKeyChecking=no -i ~/.ssh/${KEY_NAME}.pem" \
            --exclude='certs/' \
            --exclude='.git' \
            --exclude='output/' \
            "$REPO_ROOT/" "ec2-user@${ip}:${REMOTE_DIR}/"

        ssh_cmd "$ip" "
            set -e
            cd ${REMOTE_DIR}
            sudo podman-compose down
            sudo podman pull ${REGISTRY}/vjepa2-server-${accel}:latest
            sudo SSL_KEYFILE=/certs/key.pem SSL_CERTFILE=/certs/cert.pem \
                podman-compose ${profiles} up -d
            sudo podman ps --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'
        "
    fi
}

cmd_benchmark() {
    local target="https://localhost:8443"
    if [[ "$LOCAL" != "true" ]]; then
        local ip
        ip=$(get_ec2_ip)
        target="https://$ip:8443"
    fi

    echo "Running benchmark against $target..."
    cd "$REPO_ROOT"
    if [[ -f benchmark/run_benchmark.py ]]; then
        python3 benchmark/run_benchmark.py --server "$target" --insecure
    else
        echo "No benchmark script found at benchmark/run_benchmark.py"
        exit 1
    fi
}

# --- Main ---

LOCAL="false"

args=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --local) LOCAL="true"; shift ;;
        *)       args+=("$1"); shift ;;
    esac
done
set -- "${args[@]+"${args[@]}"}"

[[ $# -lt 1 ]] && usage

case "$1" in
    setup)
        if [[ "$LOCAL" == "true" ]]; then
            cmd_setup_local
        else
            cmd_setup_remote
        fi
        ;;
    run)       cmd_run ;;
    stop)      cmd_stop ;;
    status)    cmd_status ;;
    logs)      cmd_logs "$@" ;;
    redeploy)  cmd_redeploy ;;
    benchmark) cmd_benchmark ;;
    *)         usage ;;
esac
