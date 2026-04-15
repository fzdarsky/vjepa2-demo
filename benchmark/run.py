#!/usr/bin/env python3
"""
Benchmark Runner with Results Storage

Wraps benchmark.py to capture results with full metadata and store them
in benchmark/results/ for historical tracking and report generation.

Usage:
    # Run standard benchmark (20 requests each at c=1 and c=4)
    # Hardware and device are auto-detected
    python -m benchmark.run \
        --target http://localhost:8000 \
        --video benchmark/videos/ucf101-archery.mp4 \
        --server vjepa2-server \
        --model vit-l

    # Override hardware/device if auto-detection fails
    python -m benchmark.run \
        --target http://localhost:8000 \
        --video benchmark/videos/ucf101-archery.mp4 \
        --server vjepa2-server \
        --model vit-l \
        --hardware m2-max \
        --device mps

    # Single concurrency level
    python -m benchmark.run \
        --target http://localhost:8000 \
        --video benchmark/videos/ucf101-archery.mp4 \
        --server vjepa2-server \
        --model vit-l \
        --concurrency 1

Output files follow the pattern (sorted by config, then timestamp):
    {hardware}_{device}_{server}_{model}_c{concurrency}_{timestamp}.json

Example:
    m2-max_mps_vjepa2-server_vit-l_c1_2026-04-15T10-00-00.json
"""

import argparse
import json
import platform
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def get_git_info() -> dict:
    """Get git version (tag or commit) and dirty status.

    Uses git describe to get:
    - 'v1.2.0' if exactly on a tag
    - 'v1.2.0-5-g3a818a1' if 5 commits after tag
    - '3a818a1' if no tags exist (fallback to short commit)
    """
    info = {"version": None, "commit": None, "dirty": False}
    try:
        # Get version via git describe (prefers tags)
        result = subprocess.run(
            ["git", "describe", "--tags", "--always"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            info["version"] = result.stdout.strip()

        # Also get short commit hash for correlation
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            info["commit"] = result.stdout.strip()

        # Check dirty status
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            info["dirty"] = bool(result.stdout.strip())
    except Exception:
        pass
    return info


def derive_hardware_id(hardware_info: dict) -> str:
    """Derive a short, filesystem-safe hardware identifier.

    Examples:
        "Apple M2 Max" -> "m2-max"
        "g6.xlarge" -> "g6.xlarge"
        "AMD EPYC 7R13" -> "epyc-7r13"
    """
    # Prefer EC2 instance type if available (already short and standard)
    if hardware_info.get("instance_type"):
        return hardware_info["instance_type"]

    # Derive from CPU model
    cpu_model = hardware_info.get("cpu_model", "")

    # Apple Silicon: "Apple M2 Max" -> "m2-max"
    if cpu_model.startswith("Apple "):
        chip = cpu_model[6:].lower().replace(" ", "-")
        return chip

    # Intel: "Intel(R) Core(TM) i9-12900K" -> "i9-12900k"
    if "Intel" in cpu_model:
        import re

        match = re.search(r"(i\d-\d+\w*)", cpu_model, re.IGNORECASE)
        if match:
            return match.group(1).lower()

    # AMD: "AMD EPYC 7R13" -> "epyc-7r13"
    if "AMD" in cpu_model:
        import re

        match = re.search(r"(EPYC|Ryzen)\s*(\d+\w*)", cpu_model, re.IGNORECASE)
        if match:
            return f"{match.group(1).lower()}-{match.group(2).lower()}"

    # Fallback: arch + core count
    arch = hardware_info.get("arch", "unknown")
    cores = hardware_info.get("cpu_count", "")
    return f"{arch}-{cores}c" if cores else arch


def get_hardware_info() -> dict:
    """Get detailed hardware information.

    Returns structured hardware metadata suitable for reproducible benchmarking.
    """
    import os

    info = {
        "platform": platform.system().lower(),  # darwin, linux, windows
        "arch": platform.machine(),  # arm64, x86_64
        "cpu_model": None,
        "cpu_count": os.cpu_count(),
        "memory_gb": None,
        "memory_arch": None,  # unified (Apple Silicon) or discrete
        "instance_type": None,  # EC2 instance type
    }

    # Memory via psutil
    try:
        import psutil

        info["memory_gb"] = round(psutil.virtual_memory().total / (1024**3), 1)
    except ImportError:
        pass

    # CPU model detection
    if platform.system() == "Darwin":
        # macOS: use sysctl
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                info["cpu_model"] = result.stdout.strip()
        except Exception:
            pass

        # Apple Silicon has unified memory
        if platform.machine() == "arm64":
            info["memory_arch"] = "unified"
        else:
            info["memory_arch"] = "discrete"

    elif platform.system() == "Linux":
        # Linux: parse /proc/cpuinfo
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        info["cpu_model"] = line.split(":", 1)[1].strip()
                        break
        except Exception:
            pass
        info["memory_arch"] = "discrete"

    # EC2 instance type detection (1s timeout to avoid blocking on non-EC2)
    try:
        import httpx

        response = httpx.get(
            "http://169.254.169.254/latest/meta-data/instance-type",
            timeout=1.0,
        )
        if response.status_code == 200:
            info["instance_type"] = response.text.strip()
    except Exception:
        pass

    return info


def get_server_info(target_url: str, insecure: bool = False) -> dict:
    """Query server health endpoint for device info."""
    info = {"device": None, "gpu": None}
    try:
        import httpx
        response = httpx.get(
            f"{target_url.rstrip('/')}/v2/health/ready",
            timeout=10,
            verify=not insecure,
        )
        if response.status_code == 200:
            data = response.json()
            info["device"] = data.get("device")
    except Exception:
        pass
    return info


def run_benchmark(args: list[str], output_file: str) -> dict | None:
    """Run benchmark.py and capture JSON output."""
    cmd = [sys.executable, "-m", "benchmark.benchmark"] + args + ["--output", output_file]
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        return None

    try:
        with open(output_file) as f:
            return json.load(f)
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Run benchmark with metadata capture and results storage",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Required args for this wrapper
    parser.add_argument(
        "--server",
        required=True,
        help="Server implementation (e.g., vjepa2-server, kserve)",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model identifier (e.g., vit-l, vit-h)",
    )

    # Hardware/device identification (auto-detected if not provided)
    parser.add_argument(
        "--hardware",
        help="Hardware identifier (e.g., m2-max, g6.xlarge). Auto-detected from CPU/instance.",
    )
    parser.add_argument(
        "--device",
        help="Device type (e.g., mps, cuda, cpu). Auto-detected from server health endpoint.",
    )

    # Optional metadata
    parser.add_argument(
        "--container-image",
        help="Container image tag for metadata",
    )
    parser.add_argument(
        "--results-dir",
        default="benchmark/results",
        help="Directory to store results (default: benchmark/results)",
    )

    # Required passthrough args
    parser.add_argument(
        "--target",
        required=True,
        help="Target API URL (passed to benchmark.py)",
    )
    parser.add_argument(
        "--video",
        required=True,
        help="Video file path (passed to benchmark.py)",
    )

    # Concurrency configuration
    parser.add_argument(
        "--concurrency", "-c",
        type=int,
        help="Single concurrency level (overrides --concurrency-levels)",
    )
    parser.add_argument(
        "--concurrency-levels",
        default="1,4",
        help="Comma-separated concurrency levels to test (default: 1,4)",
    )

    # Known passthrough args with defaults
    parser.add_argument("--requests", "-n", type=int, default=20)
    parser.add_argument("--jaeger")
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--trace-delay", type=float)
    parser.add_argument("--num-frames", type=int)
    parser.add_argument("--fps", type=float)
    parser.add_argument("--mode", choices=["fixed", "saturation", "soak"])
    parser.add_argument("--inject-timestamp", action="store_true")
    parser.add_argument("--insecure", "-k", action="store_true",
                        help="Skip SSL certificate verification")

    args = parser.parse_args()

    # Determine concurrency levels to test
    if args.concurrency:
        # Single concurrency specified - only run that level
        concurrency_levels = [args.concurrency]
    else:
        # Parse comma-separated levels
        concurrency_levels = [int(c.strip()) for c in args.concurrency_levels.split(",")]

    # Build base passthrough args for benchmark.py (without concurrency)
    passthrough_base = ["--target", args.target, "--video", args.video]
    passthrough_base.extend(["--requests", str(args.requests)])
    if args.jaeger:
        passthrough_base.extend(["--jaeger", args.jaeger])
    if args.warmup:
        passthrough_base.extend(["--warmup", str(args.warmup)])
    if args.trace_delay:
        passthrough_base.extend(["--trace-delay", str(args.trace_delay)])
    if args.num_frames:
        passthrough_base.extend(["--num-frames", str(args.num_frames)])
    if args.fps:
        passthrough_base.extend(["--fps", str(args.fps)])
    if args.mode:
        passthrough_base.extend(["--mode", args.mode])
    if args.inject_timestamp:
        passthrough_base.append("--inject-timestamp")
    if args.insecure:
        passthrough_base.append("--insecure")

    # Collect metadata
    timestamp = datetime.now(timezone.utc)
    git_info = get_git_info()
    hardware_info = get_hardware_info()
    server_info = get_server_info(args.target, insecure=args.insecure)

    # Derive or apply overrides for hardware and device
    hardware_id = args.hardware or derive_hardware_id(hardware_info)
    device = args.device or server_info.get("device") or "unknown"

    print(f"Hardware: {hardware_id}")
    print(f"Device: {device}")
    print(f"Server: {args.server}")
    print(f"Model: {args.model}")
    version_str = git_info["version"] or git_info["commit"] or "unknown"
    if git_info["dirty"]:
        version_str += " (dirty)"
    print(f"Version: {version_str}")
    print(f"Concurrency levels: {concurrency_levels}")
    print(f"Requests per level: {args.requests}")
    print()

    # Results directory setup
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Build filename: {timestamp}_{hardware}_{device}_{server}_{model}_c{concurrency}.json
    def sanitize(s: str) -> str:
        return s.replace("/", "-").replace(":", "-").replace(" ", "-")

    timestamp_str = timestamp.strftime("%Y-%m-%dT%H-%M-%S")
    safe_hardware = sanitize(hardware_id)
    safe_device = sanitize(device)
    safe_server = sanitize(args.server)
    safe_model = sanitize(args.model)

    output_paths = []

    for concurrency in concurrency_levels:
        print(f"{'='*60}")
        print(f"Running benchmark with concurrency={concurrency}")
        print(f"{'='*60}")

        # Build passthrough args with this concurrency level
        passthrough = passthrough_base + ["--concurrency", str(concurrency)]

        # Run benchmark
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name

        benchmark_result = run_benchmark(passthrough, tmp_path)
        Path(tmp_path).unlink(missing_ok=True)

        if benchmark_result is None:
            print(f"Benchmark failed for concurrency={concurrency}", file=sys.stderr)
            continue

        # Extract summary metrics from benchmark result
        summary = benchmark_result.get("summary", {})
        methodology = benchmark_result.get("methodology", {})

        # Build stored result
        result = {
            "metadata": {
                "timestamp": timestamp.isoformat(),
                "version": git_info["version"],
                "commit": git_info["commit"],
                "dirty": git_info["dirty"],
                "hardware": hardware_id,
                "device": device,
                "server": args.server,
                "model": args.model,
                "video_file": Path(args.video).name,
                "concurrency": concurrency,
                "requests": args.requests,
                "environment": {
                    "platform": hardware_info["platform"],
                    "arch": hardware_info["arch"],
                    "cpu_model": hardware_info["cpu_model"],
                    "cpu_count": hardware_info["cpu_count"],
                    "memory_gb": hardware_info["memory_gb"],
                    "memory_arch": hardware_info["memory_arch"],
                    "instance_type": hardware_info["instance_type"],
                    "gpu": server_info.get("gpu"),
                    "cuda_version": None,
                    "pytorch_version": None,
                    "container_image": args.container_image,
                },
            },
            "config": benchmark_result.get("config", {}),
            "summary": {
                "mean_latency_ms": summary.get("mean_latency_ms"),
                "p95_latency_ms": None,
                "p99_latency_ms": None,
                "throughput_rps": summary.get("throughput_rps"),
                "success_rate": summary.get("success_rate"),
            },
            "methodology": methodology,
            "stages": benchmark_result.get("stages", {}),
            "raw": benchmark_result,
        }

        # Extract p95/p99 from stages if available
        stages = benchmark_result.get("stages", {})
        if stages:
            # Use the total or largest stage for percentiles
            for stage_name in ["video_inference", "clip_inference"]:
                if stage_name in stages:
                    result["summary"]["p95_latency_ms"] = stages[stage_name].get("p95_ms")
                    result["summary"]["p99_latency_ms"] = stages[stage_name].get("p99_ms")
                    break

        # Save result: {hardware}_{device}_{server}_{model}_c{concurrency}_{timestamp}.json
        # Hardware-first ordering enables natural sorting by configuration
        filename = f"{safe_hardware}_{safe_device}_{safe_server}_{safe_model}_c{concurrency}_{timestamp_str}.json"
        output_path = results_dir / filename

        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)

        output_paths.append(output_path)
        print(f"Results saved to: {output_path}")
        print()

    if not output_paths:
        print("All benchmarks failed", file=sys.stderr)
        sys.exit(1)

    print(f"{'='*60}")
    print(f"Completed {len(output_paths)} benchmark(s)")
    for path in output_paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
