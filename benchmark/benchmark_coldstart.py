#!/usr/bin/env python3
"""
V-JEPA2 Cold Start Benchmark

Measures startup latency components:
  - Model loading (disk → CPU → GPU)
  - First inference (includes JIT compilation, memory allocation)
  - Warm inference (steady-state)

This helps understand the penalty for serverless/scale-to-zero deployments.

Usage:
    python -m scripts.benchmark_coldstart
    python -m scripts.benchmark_coldstart --iterations 5 --output coldstart.json
"""

import argparse
import gc
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class ColdStartResult:
    """Cold start benchmark results."""
    model_path: str
    device: str
    iterations: int
    load_times_ms: list[float] = field(default_factory=list)
    first_inference_ms: list[float] = field(default_factory=list)
    warm_inference_ms: list[float] = field(default_factory=list)

    @property
    def mean_load_ms(self) -> float:
        return sum(self.load_times_ms) / len(self.load_times_ms) if self.load_times_ms else 0

    @property
    def mean_first_inference_ms(self) -> float:
        return sum(self.first_inference_ms) / len(self.first_inference_ms) if self.first_inference_ms else 0

    @property
    def mean_warm_inference_ms(self) -> float:
        return sum(self.warm_inference_ms) / len(self.warm_inference_ms) if self.warm_inference_ms else 0

    @property
    def total_cold_start_ms(self) -> float:
        return self.mean_load_ms + self.mean_first_inference_ms

    @property
    def first_inference_overhead(self) -> float:
        """How much slower is first inference vs warm."""
        if self.mean_warm_inference_ms > 0:
            return self.mean_first_inference_ms / self.mean_warm_inference_ms
        return 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": {
                "model_path": self.model_path,
                "device": self.device,
                "iterations": self.iterations,
            },
            "results": {
                "model_load_ms": {
                    "mean": round(self.mean_load_ms, 1),
                    "samples": [round(t, 1) for t in self.load_times_ms],
                },
                "first_inference_ms": {
                    "mean": round(self.mean_first_inference_ms, 1),
                    "samples": [round(t, 1) for t in self.first_inference_ms],
                },
                "warm_inference_ms": {
                    "mean": round(self.mean_warm_inference_ms, 1),
                    "samples": [round(t, 1) for t in self.warm_inference_ms],
                },
                "total_cold_start_ms": round(self.total_cold_start_ms, 1),
                "first_inference_overhead": round(self.first_inference_overhead, 2),
            },
        }


def create_synthetic_frames(num_frames: int = 16, height: int = 224, width: int = 224) -> np.ndarray:
    """Create synthetic video frames for benchmarking."""
    return np.random.randint(0, 255, (num_frames, height, width, 3), dtype=np.uint8)


def sync_device(device: str) -> None:
    """Synchronize device."""
    import torch
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def clear_caches(device: str) -> None:
    """Clear memory caches between iterations."""
    import torch
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    elif device == "mps":
        torch.mps.empty_cache()


def run_cold_start_iteration(
    model_path: str,
    device: str,
    frames: np.ndarray,
    num_warm_runs: int = 3,
) -> tuple[float, float, float]:
    """Run one cold start cycle, return (load_ms, first_infer_ms, warm_infer_ms)."""
    import torch
    from app.model import VJepa2Model

    clear_caches(device)

    # Measure model loading
    load_start = time.perf_counter()
    model = VJepa2Model(model_path, device)
    sync_device(device)
    load_ms = (time.perf_counter() - load_start) * 1000

    # Prepare input
    inputs = model.processor(list(frames), return_tensors="pt")
    key = "pixel_values_videos" if "pixel_values_videos" in inputs else "pixel_values"
    pixel_values = inputs[key].to(device)
    sync_device(device)

    # First inference (cold)
    first_start = time.perf_counter()
    with torch.no_grad():
        _ = model.model(pixel_values)
    sync_device(device)
    first_ms = (time.perf_counter() - first_start) * 1000

    # Warm inferences
    warm_times = []
    for _ in range(num_warm_runs):
        warm_start = time.perf_counter()
        with torch.no_grad():
            _ = model.model(pixel_values)
        sync_device(device)
        warm_times.append((time.perf_counter() - warm_start) * 1000)

    warm_ms = sum(warm_times) / len(warm_times)

    # Cleanup
    del model
    clear_caches(device)

    return load_ms, first_ms, warm_ms


def main():
    parser = argparse.ArgumentParser(description="Benchmark V-JEPA2 cold start latency")
    parser.add_argument("--model", default="./model", help="Path to model directory")
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--iterations", type=int, default=3, help="Number of cold start cycles")
    parser.add_argument("--output", "-o", help="Save results to JSON")

    args = parser.parse_args()

    if not Path(args.model).exists():
        print(f"Error: Model not found: {args.model}", file=sys.stderr)
        sys.exit(1)

    from app.model import select_device
    device = select_device(args.device)

    print(f"Cold Start Benchmark")
    print(f"Model: {args.model}")
    print(f"Device: {device}")
    print(f"Iterations: {args.iterations}")

    # Create synthetic frames
    frames = create_synthetic_frames()

    result = ColdStartResult(
        model_path=args.model,
        device=device,
        iterations=args.iterations,
    )

    print(f"\nRunning {args.iterations} cold start cycles...")
    print(f"  {'Iter':>4} {'Load':>10} {'1st Infer':>12} {'Warm Infer':>12} {'Overhead':>10}")
    print(f"  {'-'*4} {'-'*10} {'-'*12} {'-'*12} {'-'*10}")

    for i in range(args.iterations):
        load_ms, first_ms, warm_ms = run_cold_start_iteration(
            args.model, device, frames
        )

        result.load_times_ms.append(load_ms)
        result.first_inference_ms.append(first_ms)
        result.warm_inference_ms.append(warm_ms)

        overhead = first_ms / warm_ms if warm_ms > 0 else 0
        print(f"  {i+1:>4} {load_ms:>9.1f}ms {first_ms:>11.1f}ms {warm_ms:>11.1f}ms {overhead:>9.1f}x")

    print("\n" + "=" * 60)
    print("COLD START RESULTS")
    print("=" * 60)

    print(f"\n  Model Load:           {result.mean_load_ms:>8.1f}ms")
    print(f"  First Inference:      {result.mean_first_inference_ms:>8.1f}ms")
    print(f"  Warm Inference:       {result.mean_warm_inference_ms:>8.1f}ms")
    print(f"  ─────────────────────────────────")
    print(f"  Total Cold Start:     {result.total_cold_start_ms:>8.1f}ms")
    print(f"  First Inference OH:   {result.first_inference_overhead:>8.1f}x slower than warm")

    # Context
    print(f"\nImplications:")
    if result.total_cold_start_ms > 5000:
        print(f"  ⚠ Cold start > 5s - consider keeping model warm")
    if result.first_inference_overhead > 2:
        print(f"  ⚠ First inference {result.first_inference_overhead:.1f}x slower - JIT compilation overhead")
    print(f"  Scale-to-zero penalty: {result.total_cold_start_ms/1000:.1f}s before first prediction")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
