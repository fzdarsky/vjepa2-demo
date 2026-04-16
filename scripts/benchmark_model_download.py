#!/usr/bin/env python3
"""Benchmark model download: OCI vs HuggingFace.

Compares cold download times and measures:
- Time to complete download
- Bytes transferred
- Volume creation overhead (OCI only)

Usage:
    python scripts/benchmark_model_download.py [--runs N] [--model vitl|vitg]
"""

import argparse
import json
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

MODELS = {
    "vitl": {
        "hf_id": "facebook/vjepa2-vitl-fpc16-256-ssv2",
        "oci_image": "quay.io/fzdarsky/vjepa2-model-vitl:latest",
        "expected_size_gb": 1.4,
    },
    "vitg": {
        "hf_id": "facebook/vjepa2-vitg-fpc64-384-ssv2",
        "oci_image": "quay.io/fzdarsky/vjepa2-model-vitg:latest",
        "expected_size_gb": 4.3,
    },
}


@dataclass
class BenchmarkResult:
    source: str
    duration_s: float
    size_bytes: int
    throughput_mbps: float
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "duration_s": round(self.duration_s, 2),
            "size_mb": round(self.size_bytes / 1024 / 1024, 1),
            "throughput_mbps": round(self.throughput_mbps, 1),
            "error": self.error,
        }


def clear_hf_cache(model_id: str) -> None:
    """Remove HuggingFace cache for a specific model."""
    cache_dir = Path.home() / ".cache/huggingface/hub"
    if not cache_dir.exists():
        return

    # HF cache uses models--org--name format
    model_cache_name = "models--" + model_id.replace("/", "--")
    model_cache_path = cache_dir / model_cache_name

    if model_cache_path.exists():
        print(f"  Clearing HF cache: {model_cache_path}")
        shutil.rmtree(model_cache_path)

    # Also clear any snapshots
    snapshots_dir = cache_dir / "hub" / model_cache_name
    if snapshots_dir.exists():
        shutil.rmtree(snapshots_dir)


def clear_podman_image(image: str) -> None:
    """Remove OCI image from local storage."""
    print(f"  Clearing podman image: {image}")
    subprocess.run(
        ["podman", "rmi", "-f", image],
        capture_output=True,
    )


def get_podman_image_size(image: str) -> int:
    """Get size of a podman image in bytes."""
    result = subprocess.run(
        ["podman", "image", "inspect", image, "--format", "{{.Size}}"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return 0
    return int(result.stdout.strip())


def benchmark_hf_cold(model_id: str) -> BenchmarkResult:
    """Download from HuggingFace with no cache."""
    from huggingface_hub import snapshot_download

    clear_hf_cache(model_id)
    download_dir = Path("/tmp/hf_benchmark_model")
    if download_dir.exists():
        shutil.rmtree(download_dir)

    print(f"  Downloading from HuggingFace: {model_id}")
    start = time.monotonic()
    try:
        snapshot_download(
            model_id,
            local_dir=str(download_dir),
            local_dir_use_symlinks=False,
        )
        duration = time.monotonic() - start

        # Calculate total size
        size = sum(
            f.stat().st_size for f in download_dir.rglob("*") if f.is_file()
        )
        throughput = (size / 1024 / 1024) / duration * 8  # Mbps

        return BenchmarkResult(
            source="hf_cold",
            duration_s=duration,
            size_bytes=size,
            throughput_mbps=throughput,
        )
    except Exception as e:
        return BenchmarkResult(
            source="hf_cold",
            duration_s=0,
            size_bytes=0,
            throughput_mbps=0,
            error=str(e),
        )
    finally:
        if download_dir.exists():
            shutil.rmtree(download_dir)


def benchmark_hf_cached(model_id: str) -> BenchmarkResult:
    """Load from HuggingFace cache (already downloaded)."""
    from huggingface_hub import snapshot_download

    download_dir = Path("/tmp/hf_benchmark_model_cached")
    if download_dir.exists():
        shutil.rmtree(download_dir)

    print(f"  Loading from HuggingFace cache: {model_id}")
    start = time.monotonic()
    try:
        # This should use cache
        snapshot_download(
            model_id,
            local_dir=str(download_dir),
            local_dir_use_symlinks=False,
        )
        duration = time.monotonic() - start

        size = sum(
            f.stat().st_size for f in download_dir.rglob("*") if f.is_file()
        )
        throughput = (size / 1024 / 1024) / duration * 8 if duration > 0 else 0

        return BenchmarkResult(
            source="hf_cached",
            duration_s=duration,
            size_bytes=size,
            throughput_mbps=throughput,
        )
    except Exception as e:
        return BenchmarkResult(
            source="hf_cached",
            duration_s=0,
            size_bytes=0,
            throughput_mbps=0,
            error=str(e),
        )
    finally:
        if download_dir.exists():
            shutil.rmtree(download_dir)


def benchmark_oci_cold(image: str) -> BenchmarkResult:
    """Pull OCI image with no local cache."""
    clear_podman_image(image)

    print(f"  Pulling OCI image: {image}")
    start = time.monotonic()
    result = subprocess.run(
        ["podman", "pull", image],
        capture_output=True,
        text=True,
    )
    duration = time.monotonic() - start

    if result.returncode != 0:
        return BenchmarkResult(
            source="oci_cold",
            duration_s=duration,
            size_bytes=0,
            throughput_mbps=0,
            error=result.stderr,
        )

    size = get_podman_image_size(image)
    throughput = (size / 1024 / 1024) / duration * 8 if duration > 0 else 0

    return BenchmarkResult(
        source="oci_cold",
        duration_s=duration,
        size_bytes=size,
        throughput_mbps=throughput,
    )


def benchmark_oci_cached(image: str) -> BenchmarkResult:
    """Pull OCI image that's already present (no-op check)."""
    print(f"  Pulling OCI image (cached): {image}")
    start = time.monotonic()
    result = subprocess.run(
        ["podman", "pull", image],
        capture_output=True,
        text=True,
    )
    duration = time.monotonic() - start

    if result.returncode != 0:
        return BenchmarkResult(
            source="oci_cached",
            duration_s=duration,
            size_bytes=0,
            throughput_mbps=0,
            error=result.stderr,
        )

    size = get_podman_image_size(image)

    return BenchmarkResult(
        source="oci_cached",
        duration_s=duration,
        size_bytes=size,
        throughput_mbps=0,  # No actual transfer
    )


def benchmark_oci_volume_create(image: str) -> BenchmarkResult:
    """Measure volume creation from OCI image."""
    vol_name = "benchmark-model-vol"

    # Ensure volume doesn't exist
    subprocess.run(
        ["podman", "volume", "rm", "-f", vol_name],
        capture_output=True,
    )

    print(f"  Creating volume from: {image}")
    start = time.monotonic()
    result = subprocess.run(
        [
            "podman", "volume", "create",
            "--driver", "image",
            "--opt", f"image={image}",
            vol_name,
        ],
        capture_output=True,
        text=True,
    )
    duration = time.monotonic() - start

    # Cleanup
    subprocess.run(["podman", "volume", "rm", vol_name], capture_output=True)

    if result.returncode != 0:
        return BenchmarkResult(
            source="oci_volume_create",
            duration_s=duration,
            size_bytes=0,
            throughput_mbps=0,
            error=result.stderr,
        )

    size = get_podman_image_size(image)

    return BenchmarkResult(
        source="oci_volume_create",
        duration_s=duration,
        size_bytes=size,
        throughput_mbps=0,  # Local operation
    )


def run_benchmarks(model_key: str, runs: int = 3) -> list[dict]:
    """Run all benchmarks for a model."""
    model = MODELS[model_key]
    results = []

    print(f"\n{'='*60}")
    print(f"Benchmarking: {model_key}")
    print(f"  HuggingFace: {model['hf_id']}")
    print(f"  OCI Image:   {model['oci_image']}")
    print(f"  Expected:    ~{model['expected_size_gb']} GB")
    print(f"{'='*60}\n")

    for run in range(runs):
        print(f"\n--- Run {run + 1}/{runs} ---")

        # Cold downloads (clear cache first)
        print("\n[HuggingFace Cold Download]")
        results.append(benchmark_hf_cold(model["hf_id"]).to_dict())

        print("\n[OCI Cold Pull]")
        results.append(benchmark_oci_cold(model["oci_image"]).to_dict())

        # Cached operations
        print("\n[HuggingFace Cached]")
        results.append(benchmark_hf_cached(model["hf_id"]).to_dict())

        print("\n[OCI Cached Pull]")
        results.append(benchmark_oci_cached(model["oci_image"]).to_dict())

        # Volume creation (OCI-specific)
        print("\n[OCI Volume Creation]")
        results.append(benchmark_oci_volume_create(model["oci_image"]).to_dict())

    return results


def summarize_results(results: list[dict]) -> dict:
    """Compute averages across runs."""
    from collections import defaultdict

    by_source = defaultdict(list)
    for r in results:
        if r["error"] is None:
            by_source[r["source"]].append(r)

    summary = {}
    for source, runs in by_source.items():
        if runs:
            summary[source] = {
                "avg_duration_s": round(
                    sum(r["duration_s"] for r in runs) / len(runs), 2
                ),
                "avg_size_mb": round(
                    sum(r["size_mb"] for r in runs) / len(runs), 1
                ),
                "avg_throughput_mbps": round(
                    sum(r["throughput_mbps"] for r in runs) / len(runs), 1
                ),
                "runs": len(runs),
            }

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark model download: OCI vs HuggingFace"
    )
    parser.add_argument(
        "--model",
        choices=["vitl", "vitg"],
        default="vitl",
        help="Model to benchmark (default: vitl)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of benchmark runs (default: 3)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file for results (JSON)",
    )
    args = parser.parse_args()

    results = run_benchmarks(args.model, args.runs)
    summary = summarize_results(results)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(json.dumps(summary, indent=2))

    output = {
        "model": args.model,
        "runs": args.runs,
        "results": results,
        "summary": summary,
    }

    if args.output:
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to: {args.output}")
    else:
        print("\n" + "-" * 60)
        print("DETAILED RESULTS")
        print("-" * 60)
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
