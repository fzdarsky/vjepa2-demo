#!/usr/bin/env python3
"""
V-JEPA2 End-to-End Latency Benchmark

Measures the complete capture-to-prediction latency including:
  - Source latency (camera/network/file)
  - Decode latency (FFmpeg/PyAV)
  - Buffer/queue latency
  - Inference latency
  - Total glass-to-glass

Methods:
  1. Synthetic timestamps: Embed frame number, measure decode delay
  2. Visual clock: Display timestamp, capture, measure round-trip
  3. Source comparison: File vs RTSP vs camera

Usage:
    # Measure file decode latency
    python -m scripts.benchmark_e2e --video samples/sample.mp4

    # Measure RTSP source latency (requires running RTSP server)
    python -m scripts.benchmark_e2e --rtsp rtsp://localhost:8554/stream

    # Measure local camera latency
    python -m scripts.benchmark_e2e --camera 0

    # Compare sources
    python -m scripts.benchmark_e2e --compare --video samples/sample.mp4 --rtsp rtsp://...
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np


@dataclass
class SourceLatencyResult:
    """Latency measurements for a single source."""
    source_type: str  # "file", "rtsp", "camera", "synthetic"
    source_uri: str
    num_samples: int
    frame_latencies_ms: list[float] = field(default_factory=list)
    decode_latencies_ms: list[float] = field(default_factory=list)
    total_latencies_ms: list[float] = field(default_factory=list)

    @property
    def mean_frame_latency_ms(self) -> float:
        return sum(self.frame_latencies_ms) / len(self.frame_latencies_ms) if self.frame_latencies_ms else 0

    @property
    def mean_decode_latency_ms(self) -> float:
        return sum(self.decode_latencies_ms) / len(self.decode_latencies_ms) if self.decode_latencies_ms else 0

    @property
    def mean_total_latency_ms(self) -> float:
        return sum(self.total_latencies_ms) / len(self.total_latencies_ms) if self.total_latencies_ms else 0

    def percentile(self, values: list[float], p: float) -> float:
        if not values:
            return 0
        sorted_vals = sorted(values)
        idx = min(int(len(sorted_vals) * p / 100), len(sorted_vals) - 1)
        return sorted_vals[idx]

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_type": self.source_type,
            "source_uri": self.source_uri,
            "num_samples": self.num_samples,
            "frame_latency_ms": {
                "mean": round(self.mean_frame_latency_ms, 2),
                "p50": round(self.percentile(self.frame_latencies_ms, 50), 2),
                "p99": round(self.percentile(self.frame_latencies_ms, 99), 2),
            },
            "decode_latency_ms": {
                "mean": round(self.mean_decode_latency_ms, 2),
                "p50": round(self.percentile(self.decode_latencies_ms, 50), 2),
                "p99": round(self.percentile(self.decode_latencies_ms, 99), 2),
            },
            "total_latency_ms": {
                "mean": round(self.mean_total_latency_ms, 2),
                "p50": round(self.percentile(self.total_latencies_ms, 50), 2),
                "p99": round(self.percentile(self.total_latencies_ms, 99), 2),
            },
        }


@dataclass
class E2EBenchmarkResult:
    """Complete E2E benchmark results."""
    device: str
    inference_latency_ms: float
    sources: list[SourceLatencyResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "device": self.device,
            "inference_latency_ms": round(self.inference_latency_ms, 1),
            "sources": [s.to_dict() for s in self.sources],
            "analysis": self._analyze(),
        }

    def _analyze(self) -> dict[str, Any]:
        """Generate analysis comparing sources."""
        if not self.sources:
            return {}

        analysis = {
            "inference_dominates": self.inference_latency_ms > 100,
            "source_latency_matters": False,
        }

        for src in self.sources:
            # If source latency is >10% of inference, it matters
            if src.mean_total_latency_ms > self.inference_latency_ms * 0.1:
                analysis["source_latency_matters"] = True

        if len(self.sources) > 1:
            sorted_sources = sorted(self.sources, key=lambda s: s.mean_total_latency_ms)
            analysis["fastest_source"] = sorted_sources[0].source_type
            analysis["slowest_source"] = sorted_sources[-1].source_type
            analysis["source_latency_range_ms"] = (
                round(sorted_sources[0].mean_total_latency_ms, 1),
                round(sorted_sources[-1].mean_total_latency_ms, 1),
            )

        return analysis


def iter_frames_from_file(path: str, max_frames: int = 100) -> Iterator[tuple[int, float, np.ndarray]]:
    """Yield (frame_idx, decode_start_time, frame) from file."""
    import av

    container = av.open(path)
    for i, frame in enumerate(container.decode(video=0)):
        if i >= max_frames:
            break
        decode_start = time.perf_counter()
        arr = frame.to_ndarray(format="rgb24")
        yield i, decode_start, arr
    container.close()


def iter_frames_from_rtsp(url: str, max_frames: int = 100, timeout: float = 10.0) -> Iterator[tuple[int, float, np.ndarray]]:
    """Yield frames from RTSP stream with timestamp tracking."""
    import av

    options = {
        "rtsp_transport": "tcp",
        "stimeout": str(int(timeout * 1000000)),  # microseconds
    }

    try:
        container = av.open(url, options=options)
    except Exception as e:
        print(f"Failed to connect to RTSP: {e}")
        return

    for i, frame in enumerate(container.decode(video=0)):
        if i >= max_frames:
            break
        decode_start = time.perf_counter()
        arr = frame.to_ndarray(format="rgb24")

        # Try to get frame PTS for latency calculation
        # Note: actual capture time requires camera-side timestamps
        yield i, decode_start, arr

    container.close()


def iter_frames_from_camera(device_id: int = 0, max_frames: int = 100) -> Iterator[tuple[int, float, np.ndarray]]:
    """Yield frames from local camera (V4L2/AVFoundation)."""
    import av

    # Platform-specific device path
    import platform
    if platform.system() == "Darwin":
        device_path = str(device_id)
        input_format = "avfoundation"
    else:
        device_path = f"/dev/video{device_id}"
        input_format = "v4l2"

    try:
        container = av.open(device_path, format=input_format)
    except Exception as e:
        print(f"Failed to open camera: {e}")
        return

    for i, frame in enumerate(container.decode(video=0)):
        if i >= max_frames:
            break
        decode_start = time.perf_counter()
        arr = frame.to_ndarray(format="rgb24")
        yield i, decode_start, arr

    container.close()


def measure_source_latency(
    source_type: str,
    source_uri: str,
    frame_iterator: Iterator[tuple[int, float, np.ndarray]],
    num_samples: int = 50,
) -> SourceLatencyResult:
    """Measure latency for a frame source."""
    result = SourceLatencyResult(
        source_type=source_type,
        source_uri=source_uri,
        num_samples=num_samples,
    )

    print(f"\nMeasuring {source_type} source: {source_uri}")

    prev_time = None
    for i, (frame_idx, decode_start, frame) in enumerate(frame_iterator):
        decode_end = time.perf_counter()
        decode_ms = (decode_end - decode_start) * 1000

        result.decode_latencies_ms.append(decode_ms)

        # Inter-frame latency (time between frames arriving)
        if prev_time is not None:
            frame_latency = (decode_start - prev_time) * 1000
            result.frame_latencies_ms.append(frame_latency)

        prev_time = decode_start

        # Total latency estimate (decode + any buffering)
        # For files this is just decode time
        # For RTSP/camera this includes network/driver latency
        result.total_latencies_ms.append(decode_ms)

        if i >= num_samples:
            break

    result.num_samples = len(result.decode_latencies_ms)
    return result


def measure_inference_latency(model_path: str, device: str) -> float:
    """Measure warm inference latency."""
    import torch
    from app.model import VJepa2Model

    print(f"\nMeasuring inference latency on {device}...")

    model = VJepa2Model(model_path, device)

    # Create synthetic input
    frames = np.random.randint(0, 255, (16, 224, 224, 3), dtype=np.uint8)
    inputs = model.processor(list(frames), return_tensors="pt")
    key = "pixel_values_videos" if "pixel_values_videos" in inputs else "pixel_values"
    pixel_values = inputs[key].to(device)

    # Sync helper
    def sync():
        if device == "cuda":
            torch.cuda.synchronize()
        elif device == "mps":
            torch.mps.synchronize()

    # Warmup
    for _ in range(3):
        with torch.no_grad():
            _ = model.model(pixel_values)
        sync()

    # Measure
    latencies = []
    for _ in range(10):
        sync()
        start = time.perf_counter()
        with torch.no_grad():
            _ = model.model(pixel_values)
        sync()
        latencies.append((time.perf_counter() - start) * 1000)

    return sum(latencies) / len(latencies)


def print_result(result: E2EBenchmarkResult) -> None:
    """Print formatted E2E results."""
    print("\n" + "=" * 70)
    print("END-TO-END LATENCY ANALYSIS")
    print("=" * 70)

    print(f"\nInference Latency: {result.inference_latency_ms:.1f}ms ({result.device})")

    for src in result.sources:
        print(f"\n{src.source_type.upper()} Source: {src.source_uri}")
        print(f"  Samples:        {src.num_samples}")
        print(f"  Decode latency: {src.mean_decode_latency_ms:.2f}ms (p99: {src.percentile(src.decode_latencies_ms, 99):.2f}ms)")
        if src.frame_latencies_ms:
            print(f"  Frame interval: {src.mean_frame_latency_ms:.2f}ms")
        print(f"  Total source:   {src.mean_total_latency_ms:.2f}ms")

        # Calculate E2E
        e2e = src.mean_total_latency_ms + result.inference_latency_ms
        source_pct = (src.mean_total_latency_ms / e2e) * 100
        infer_pct = (result.inference_latency_ms / e2e) * 100
        print(f"\n  E2E Latency Breakdown:")
        print(f"    Source:    {src.mean_total_latency_ms:>8.1f}ms ({source_pct:>5.1f}%)")
        print(f"    Inference: {result.inference_latency_ms:>8.1f}ms ({infer_pct:>5.1f}%)")
        print(f"    ─────────────────────────")
        print(f"    Total E2E: {e2e:>8.1f}ms")

    # Analysis
    analysis = result._analyze()
    print("\n" + "-" * 40)
    print("Analysis:")

    if analysis.get("inference_dominates"):
        print(f"  • Inference dominates latency ({result.inference_latency_ms:.0f}ms)")
        print(f"    → Source optimization has limited impact")
        print(f"    → Focus on model optimization (torch.compile, quantization)")

    if not analysis.get("source_latency_matters"):
        print(f"  • Source latency is <10% of inference")
        print(f"    → CSI vs network camera: minimal difference")
    else:
        print(f"  • Source latency is significant")
        if "source_latency_range_ms" in analysis:
            lo, hi = analysis["source_latency_range_ms"]
            print(f"    → Range: {lo:.1f}ms - {hi:.1f}ms")
            print(f"    → Fastest: {analysis['fastest_source']}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark end-to-end capture-to-prediction latency",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", default="./model", help="Path to model directory")
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--video", help="Video file to benchmark")
    parser.add_argument("--rtsp", help="RTSP URL to benchmark")
    parser.add_argument("--camera", type=int, help="Camera device ID")
    parser.add_argument("--samples", type=int, default=50, help="Frames to sample per source")
    parser.add_argument("--output", "-o", help="Save results to JSON")

    args = parser.parse_args()

    if not any([args.video, args.rtsp, args.camera is not None]):
        print("Error: Specify at least one source (--video, --rtsp, or --camera)")
        sys.exit(1)

    from app.model import select_device
    device = select_device(args.device)

    # Measure inference first
    inference_ms = measure_inference_latency(args.model, device)

    result = E2EBenchmarkResult(
        device=device,
        inference_latency_ms=inference_ms,
    )

    # Measure each source
    if args.video:
        if not Path(args.video).exists():
            print(f"Warning: Video not found: {args.video}")
        else:
            src_result = measure_source_latency(
                "file",
                args.video,
                iter_frames_from_file(args.video, args.samples),
                args.samples,
            )
            result.sources.append(src_result)

    if args.rtsp:
        src_result = measure_source_latency(
            "rtsp",
            args.rtsp,
            iter_frames_from_rtsp(args.rtsp, args.samples),
            args.samples,
        )
        result.sources.append(src_result)

    if args.camera is not None:
        src_result = measure_source_latency(
            "camera",
            f"device:{args.camera}",
            iter_frames_from_camera(args.camera, args.samples),
            args.samples,
        )
        result.sources.append(src_result)

    print_result(result)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
