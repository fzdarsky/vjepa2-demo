#!/usr/bin/env python3
"""
V-JEPA2 Whitebox Benchmark (In-Process)

Measures latency breakdown by directly timing Python function calls.
Use for local development and quick A/B testing without full server setup.

Stages measured (simplified view):
  - decode: video file → numpy frames
  - preprocess: frames → tensor (AutoVideoProcessor)
  - inference: tensor → logits (full model forward pass)
  - postprocess: logits → predictions (softmax + topk)

Note: For production benchmarking with full OTel span breakdown (input_*,
jepa_*, output_*), use the external benchmark: `python -m benchmark.benchmark`

Usage:
    python -m benchmark.benchmark_whitebox --video samples/sample.mp4
    python -m benchmark.benchmark_whitebox --video samples/sample.mp4 --iterations 50
    python -m benchmark.benchmark_whitebox --video samples/sample.mp4 --compare baseline.json
"""

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch


@dataclass
class StageStats:
    """Statistics for a single pipeline stage."""
    name: str
    samples: list[float] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.samples)

    @property
    def mean_ms(self) -> float:
        return statistics.mean(self.samples) * 1000 if self.samples else 0.0

    @property
    def std_ms(self) -> float:
        return statistics.stdev(self.samples) * 1000 if len(self.samples) > 1 else 0.0

    @property
    def min_ms(self) -> float:
        return min(self.samples) * 1000 if self.samples else 0.0

    @property
    def max_ms(self) -> float:
        return max(self.samples) * 1000 if self.samples else 0.0

    def percentile_ms(self, p: float) -> float:
        if not self.samples:
            return 0.0
        sorted_samples = sorted(self.samples)
        idx = int(len(sorted_samples) * p / 100)
        idx = min(idx, len(sorted_samples) - 1)
        return sorted_samples[idx] * 1000

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "count": self.count,
            "mean_ms": round(self.mean_ms, 3),
            "std_ms": round(self.std_ms, 3),
            "min_ms": round(self.min_ms, 3),
            "max_ms": round(self.max_ms, 3),
            "p50_ms": round(self.percentile_ms(50), 3),
            "p95_ms": round(self.percentile_ms(95), 3),
            "p99_ms": round(self.percentile_ms(99), 3),
        }


@dataclass
class BenchmarkResult:
    """Complete benchmark results."""
    model_path: str
    device: str
    video_path: str
    num_frames: int
    top_k: int
    warmup_iterations: int
    iterations: int
    stages: dict[str, StageStats] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total_ms(self) -> float:
        return sum(s.mean_ms for s in self.stages.values())

    @property
    def throughput_clips_per_sec(self) -> float:
        total_sec = self.total_ms / 1000
        return 1.0 / total_sec if total_sec > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": {
                "model_path": self.model_path,
                "device": self.device,
                "video_path": self.video_path,
                "num_frames": self.num_frames,
                "top_k": self.top_k,
                "warmup_iterations": self.warmup_iterations,
                "iterations": self.iterations,
            },
            "summary": {
                "total_mean_ms": round(self.total_ms, 3),
                "throughput_clips_per_sec": round(self.throughput_clips_per_sec, 2),
            },
            "stages": {name: stats.to_dict() for name, stats in self.stages.items()},
            "metadata": self.metadata,
        }


class PipelineBenchmark:
    """Benchmark harness for V-JEPA2 inference pipeline."""

    def __init__(
        self,
        model_path: str,
        device: str | None = None,
        num_frames: int = 16,
        top_k: int = 3,
    ):
        from app.model import VJepa2Model, select_device
        from app.video import decode_video

        self.model_path = model_path
        self.device = select_device(device)
        self.num_frames = num_frames
        self.top_k = top_k

        # Import here to avoid loading model during arg parsing
        self._decode_video = decode_video

        print(f"Loading model from {model_path}...")
        print(f"Device: {self.device}")
        self.model = VJepa2Model(model_path, self.device)
        print("Model loaded.")

    def _bench_decode(self, video_path: str) -> tuple[np.ndarray, float]:
        """Benchmark video decoding stage."""
        start = time.perf_counter()
        frames = self._decode_video(video_path, self.num_frames)
        elapsed = time.perf_counter() - start
        return frames, elapsed

    def _bench_preprocess(self, frames: np.ndarray) -> tuple[torch.Tensor, float]:
        """Benchmark preprocessing stage."""
        start = time.perf_counter()
        inputs = self.model.processor(list(frames), return_tensors="pt")
        key = "pixel_values_videos" if "pixel_values_videos" in inputs else "pixel_values"
        pixel_values = inputs[key].to(self.device)

        self._sync_device()
        elapsed = time.perf_counter() - start
        return pixel_values, elapsed

    def _sync_device(self) -> None:
        """Synchronize device to ensure accurate timing."""
        if self.device == "cuda":
            torch.cuda.synchronize()
        elif self.device == "mps":
            torch.mps.synchronize()

    def _bench_inference(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, float]:
        """Benchmark model inference stage."""
        self._sync_device()

        start = time.perf_counter()
        with torch.no_grad():
            outputs = self.model.model(pixel_values)

        self._sync_device()
        elapsed = time.perf_counter() - start
        return outputs.logits, elapsed

    def _bench_postprocess(self, logits: torch.Tensor) -> tuple[list, float]:
        """Benchmark postprocessing stage."""
        self._sync_device()

        start = time.perf_counter()
        probs = torch.softmax(logits, dim=-1)[0]
        top_scores, top_indices = torch.topk(probs, k=self.top_k)
        predictions = [
            {"label": self.model.id2label[idx.item()], "score": round(score.item(), 6)}
            for score, idx in zip(top_scores, top_indices)
        ]

        self._sync_device()
        elapsed = time.perf_counter() - start
        return predictions, elapsed

    def run(
        self,
        video_path: str,
        warmup: int = 5,
        iterations: int = 20,
    ) -> BenchmarkResult:
        """Run the complete benchmark."""
        result = BenchmarkResult(
            model_path=self.model_path,
            device=self.device,
            video_path=video_path,
            num_frames=self.num_frames,
            top_k=self.top_k,
            warmup_iterations=warmup,
            iterations=iterations,
        )

        # Initialize stage stats
        for stage in ["decode", "preprocess", "inference", "postprocess"]:
            result.stages[stage] = StageStats(name=stage)

        # Collect metadata
        result.metadata["torch_version"] = torch.__version__
        result.metadata["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            result.metadata["cuda_device"] = torch.cuda.get_device_name(0)
            result.metadata["cuda_memory_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1e9, 2
            )

        total_runs = warmup + iterations

        print(f"\nRunning {warmup} warmup + {iterations} timed iterations...")

        for i in range(total_runs):
            is_warmup = i < warmup
            prefix = "warmup" if is_warmup else "iter"
            iter_num = i if is_warmup else i - warmup

            # Run each stage
            frames, t_decode = self._bench_decode(video_path)
            pixel_values, t_preprocess = self._bench_preprocess(frames)
            logits, t_inference = self._bench_inference(pixel_values)
            predictions, t_postprocess = self._bench_postprocess(logits)

            total = (t_decode + t_preprocess + t_inference + t_postprocess) * 1000

            if not is_warmup:
                result.stages["decode"].samples.append(t_decode)
                result.stages["preprocess"].samples.append(t_preprocess)
                result.stages["inference"].samples.append(t_inference)
                result.stages["postprocess"].samples.append(t_postprocess)

            # Progress output
            print(
                f"  [{prefix} {iter_num:2d}] "
                f"decode={t_decode*1000:6.1f}ms "
                f"preproc={t_preprocess*1000:6.1f}ms "
                f"infer={t_inference*1000:6.1f}ms "
                f"postproc={t_postprocess*1000:5.2f}ms "
                f"total={total:7.1f}ms"
            )

        return result


def print_results(result: BenchmarkResult) -> None:
    """Print formatted benchmark results."""
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)

    print(f"\nConfig:")
    print(f"  Model:      {result.model_path}")
    print(f"  Device:     {result.device}")
    print(f"  Video:      {result.video_path}")
    print(f"  Frames:     {result.num_frames}")
    print(f"  Iterations: {result.iterations} (after {result.warmup_iterations} warmup)")

    print(f"\nLatency Breakdown (ms):")
    print(f"  {'Stage':<12} {'Mean':>8} {'Std':>8} {'P50':>8} {'P95':>8} {'P99':>8}")
    print(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    for stage in ["decode", "preprocess", "inference", "postprocess"]:
        stats = result.stages[stage]
        print(
            f"  {stage:<12} "
            f"{stats.mean_ms:>8.2f} "
            f"{stats.std_ms:>8.2f} "
            f"{stats.percentile_ms(50):>8.2f} "
            f"{stats.percentile_ms(95):>8.2f} "
            f"{stats.percentile_ms(99):>8.2f}"
        )

    print(f"  {'-'*12} {'-'*8}")
    print(f"  {'TOTAL':<12} {result.total_ms:>8.2f}")

    print(f"\nThroughput: {result.throughput_clips_per_sec:.2f} clips/sec")

    # Show percentage breakdown
    print(f"\nLatency Distribution:")
    for stage in ["decode", "preprocess", "inference", "postprocess"]:
        pct = (result.stages[stage].mean_ms / result.total_ms * 100) if result.total_ms > 0 else 0
        bar_len = int(pct / 2)
        bar = "█" * bar_len
        print(f"  {stage:<12} {bar:<50} {pct:5.1f}%")


def compare_results(current: BenchmarkResult, baseline_path: str) -> None:
    """Compare current results against a baseline."""
    with open(baseline_path) as f:
        baseline_data = json.load(f)

    print("\n" + "=" * 70)
    print("COMPARISON vs BASELINE")
    print("=" * 70)
    print(f"Baseline: {baseline_path}")

    print(f"\n  {'Stage':<12} {'Baseline':>10} {'Current':>10} {'Delta':>10} {'Change':>10}")
    print(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    baseline_total = 0
    current_total = 0

    for stage in ["decode", "preprocess", "inference", "postprocess"]:
        baseline_ms = baseline_data["stages"][stage]["mean_ms"]
        current_ms = current.stages[stage].mean_ms
        delta = current_ms - baseline_ms
        pct_change = (delta / baseline_ms * 100) if baseline_ms > 0 else 0

        baseline_total += baseline_ms
        current_total += current_ms

        indicator = "↓" if delta < 0 else "↑" if delta > 0 else "="
        print(
            f"  {stage:<12} "
            f"{baseline_ms:>9.2f}ms "
            f"{current_ms:>9.2f}ms "
            f"{delta:>+9.2f}ms "
            f"{indicator} {pct_change:>+6.1f}%"
        )

    delta_total = current_total - baseline_total
    pct_total = (delta_total / baseline_total * 100) if baseline_total > 0 else 0
    indicator = "↓" if delta_total < 0 else "↑" if delta_total > 0 else "="

    print(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    print(
        f"  {'TOTAL':<12} "
        f"{baseline_total:>9.2f}ms "
        f"{current_total:>9.2f}ms "
        f"{delta_total:>+9.2f}ms "
        f"{indicator} {pct_total:>+6.1f}%"
    )

    if delta_total < 0:
        print(f"\n  ✓ Current is {abs(pct_total):.1f}% faster than baseline")
    elif delta_total > 0:
        print(f"\n  ✗ Current is {pct_total:.1f}% slower than baseline")
    else:
        print(f"\n  = No change from baseline")


def main():
    parser = argparse.ArgumentParser(
        description="Whitebox benchmark for V-JEPA2 inference pipeline (in-process timing)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic benchmark
  python -m benchmark.benchmark_whitebox --video samples/sample.mp4

  # Save results for later comparison
  python -m benchmark.benchmark_whitebox --video samples/sample.mp4 --output baseline.json

  # Compare against baseline
  python -m benchmark.benchmark_whitebox --video samples/sample.mp4 --compare baseline.json

  # More iterations for statistical significance
  python -m benchmark.benchmark_whitebox --video samples/sample.mp4 --iterations 100 --warmup 10
        """,
    )
    parser.add_argument(
        "--video",
        required=True,
        help="Path to video file for benchmarking",
    )
    parser.add_argument(
        "--model",
        default="./model",
        help="Path to model directory (default: ./model)",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "mps"],
        help="Device to use (default: auto-detect)",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=16,
        help="Number of frames per clip (default: 16)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of top predictions (default: 3)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Number of warmup iterations (default: 5)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=20,
        help="Number of timed iterations (default: 20)",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Save results to JSON file",
    )
    parser.add_argument(
        "--compare",
        "-c",
        help="Compare against baseline JSON file",
    )

    args = parser.parse_args()

    # Validate video path
    if not Path(args.video).exists():
        print(f"Error: Video file not found: {args.video}", file=sys.stderr)
        sys.exit(1)

    # Validate model path
    if not Path(args.model).exists():
        print(f"Error: Model directory not found: {args.model}", file=sys.stderr)
        sys.exit(1)

    # Run benchmark
    benchmark = PipelineBenchmark(
        model_path=args.model,
        device=args.device,
        num_frames=args.num_frames,
        top_k=args.top_k,
    )

    result = benchmark.run(
        video_path=args.video,
        warmup=args.warmup,
        iterations=args.iterations,
    )

    # Print results
    print_results(result)

    # Compare if baseline provided
    if args.compare:
        if not Path(args.compare).exists():
            print(f"\nWarning: Baseline file not found: {args.compare}", file=sys.stderr)
        else:
            compare_results(result, args.compare)

    # Save results
    if args.output:
        with open(args.output, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
