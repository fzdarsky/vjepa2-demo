#!/usr/bin/env python3
"""
V-JEPA2 Stream Benchmark

Measures throughput and realtime performance when processing a continuous
stream of clips, with optional pipeline parallelism.

Key metrics:
  - Throughput (clips/sec, effective FPS)
  - Realtime ratio (can we keep up with source?)
  - Queue depth over time (are we falling behind?)
  - Pipeline efficiency (overlap vs sequential)

Usage:
    # Sequential processing (baseline)
    python -m scripts.benchmark_stream --video samples/hands.mp4

    # Pipelined processing (decode overlaps with inference)
    python -m scripts.benchmark_stream --video samples/hands.mp4 --pipeline

    # Compare sequential vs pipelined
    python -m scripts.benchmark_stream --video samples/hands.mp4 --compare-modes
"""

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class StreamBenchmarkResult:
    """Results from stream benchmark."""
    mode: str  # "sequential" or "pipelined"
    video_path: str
    model_path: str
    device: str
    num_frames: int
    stride: int
    source_fps: float
    total_clips: int
    total_time_sec: float
    clip_latencies_ms: list[float] = field(default_factory=list)
    queue_depths: list[int] = field(default_factory=list)  # for pipelined mode

    @property
    def throughput_clips_per_sec(self) -> float:
        return self.total_clips / self.total_time_sec if self.total_time_sec > 0 else 0

    @property
    def effective_fps(self) -> float:
        """Effective source FPS we can sustain."""
        return self.throughput_clips_per_sec * self.stride

    @property
    def rt_ratio(self) -> float:
        """Realtime ratio: <1 means keeping up, >1 means falling behind."""
        clip_duration_sec = self.stride / self.source_fps
        avg_processing_sec = (sum(self.clip_latencies_ms) / len(self.clip_latencies_ms) / 1000
                              if self.clip_latencies_ms else 0)
        return avg_processing_sec / clip_duration_sec if clip_duration_sec > 0 else 0

    @property
    def can_realtime(self) -> bool:
        return self.rt_ratio <= 1.0

    @property
    def mean_latency_ms(self) -> float:
        return sum(self.clip_latencies_ms) / len(self.clip_latencies_ms) if self.clip_latencies_ms else 0

    @property
    def max_queue_depth(self) -> int:
        return max(self.queue_depths) if self.queue_depths else 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "config": {
                "video_path": self.video_path,
                "model_path": self.model_path,
                "device": self.device,
                "num_frames": self.num_frames,
                "stride": self.stride,
                "source_fps": self.source_fps,
            },
            "results": {
                "total_clips": self.total_clips,
                "total_time_sec": round(self.total_time_sec, 3),
                "throughput_clips_per_sec": round(self.throughput_clips_per_sec, 2),
                "effective_fps": round(self.effective_fps, 1),
                "rt_ratio": round(self.rt_ratio, 3),
                "can_realtime": self.can_realtime,
                "mean_latency_ms": round(self.mean_latency_ms, 2),
                "max_queue_depth": self.max_queue_depth,
            },
        }


class StreamBenchmark:
    """Benchmark stream processing throughput."""

    def __init__(
        self,
        model_path: str,
        device: str | None = None,
        num_frames: int = 16,
        stride: int = 16,
        top_k: int = 3,
    ):
        from app.model import VJepa2Model, select_device

        self.model_path = model_path
        self.device = select_device(device)
        self.num_frames = num_frames
        self.stride = stride
        self.top_k = top_k

        print(f"Loading model from {model_path}...")
        print(f"Device: {self.device}")
        self.model = VJepa2Model(model_path, self.device)
        print("Model loaded.")

    def _get_source_fps(self, video_path: str) -> float:
        """Extract FPS from video file."""
        import av
        container = av.open(video_path)
        stream = container.streams.video[0]
        fps = float(stream.average_rate) if stream.average_rate else 30.0
        container.close()
        return fps

    def _sync_device(self) -> None:
        """Synchronize device for accurate timing."""
        import torch
        if self.device == "cuda":
            torch.cuda.synchronize()
        elif self.device == "mps":
            torch.mps.synchronize()

    def run_sequential(self, video_path: str, max_clips: int | None = None) -> StreamBenchmarkResult:
        """Process clips sequentially (no overlap)."""
        from app.video import iter_clips
        import torch

        source_fps = self._get_source_fps(video_path)
        result = StreamBenchmarkResult(
            mode="sequential",
            video_path=video_path,
            model_path=self.model_path,
            device=self.device,
            num_frames=self.num_frames,
            stride=self.stride,
            source_fps=source_fps,
            total_clips=0,
            total_time_sec=0,
        )

        print(f"\nSequential mode: processing clips from {video_path}")
        print(f"Source FPS: {source_fps:.1f}, stride: {self.stride} frames")

        clips = list(iter_clips(video_path, self.num_frames, self.stride))
        if max_clips:
            clips = clips[:max_clips]

        # Warmup
        print("Warming up...")
        if clips:
            self._process_clip(clips[0].frames)

        print(f"Processing {len(clips)} clips...")
        start_total = time.perf_counter()

        for i, clip in enumerate(clips):
            clip_start = time.perf_counter()
            self._process_clip(clip.frames)
            self._sync_device()
            clip_latency = (time.perf_counter() - clip_start) * 1000

            result.clip_latencies_ms.append(clip_latency)
            result.total_clips += 1

            if (i + 1) % 10 == 0 or i == len(clips) - 1:
                print(f"  [{i+1}/{len(clips)}] latency={clip_latency:.1f}ms")

        result.total_time_sec = time.perf_counter() - start_total
        return result

    def _process_clip(self, frames: np.ndarray) -> list:
        """Run full inference on a clip."""
        import torch

        # Preprocess
        inputs = self.model.processor(list(frames), return_tensors="pt")
        key = "pixel_values_videos" if "pixel_values_videos" in inputs else "pixel_values"
        pixel_values = inputs[key].to(self.device)

        # Inference
        with torch.no_grad():
            outputs = self.model.model(pixel_values)

        # Postprocess
        probs = torch.softmax(outputs.logits, dim=-1)[0]
        top_scores, top_indices = torch.topk(probs, k=self.top_k)

        return [
            {"label": self.model.id2label[idx.item()], "score": score.item()}
            for score, idx in zip(top_scores, top_indices)
        ]

    def run_pipelined(self, video_path: str, max_clips: int | None = None) -> StreamBenchmarkResult:
        """Process clips with pipeline parallelism (decode overlaps inference)."""
        return asyncio.run(self._run_pipelined_async(video_path, max_clips))

    async def _run_pipelined_async(
        self, video_path: str, max_clips: int | None = None
    ) -> StreamBenchmarkResult:
        """Async pipelined processing."""
        from app.video import iter_clips
        import torch

        source_fps = self._get_source_fps(video_path)
        result = StreamBenchmarkResult(
            mode="pipelined",
            video_path=video_path,
            model_path=self.model_path,
            device=self.device,
            num_frames=self.num_frames,
            stride=self.stride,
            source_fps=source_fps,
            total_clips=0,
            total_time_sec=0,
        )

        print(f"\nPipelined mode: processing clips from {video_path}")
        print(f"Source FPS: {source_fps:.1f}, stride: {self.stride} frames")

        clips = list(iter_clips(video_path, self.num_frames, self.stride))
        if max_clips:
            clips = clips[:max_clips]

        # Warmup
        print("Warming up...")
        if clips:
            self._process_clip(clips[0].frames)
            self._sync_device()

        print(f"Processing {len(clips)} clips with pipeline overlap...")

        # Use a queue for producer-consumer pattern
        queue: asyncio.Queue = asyncio.Queue(maxsize=4)
        clip_start_times: dict[int, float] = {}

        async def producer():
            """Feed clips into the queue."""
            for i, clip in enumerate(clips):
                clip_start_times[i] = time.perf_counter()
                await queue.put((i, clip))
                result.queue_depths.append(queue.qsize())
            await queue.put(None)  # Sentinel

        async def consumer():
            """Process clips from the queue."""
            while True:
                item = await queue.get()
                if item is None:
                    break

                idx, clip = item
                # Run inference in thread pool to not block event loop
                await asyncio.to_thread(self._process_clip, clip.frames)
                self._sync_device()

                clip_latency = (time.perf_counter() - clip_start_times[idx]) * 1000
                result.clip_latencies_ms.append(clip_latency)
                result.total_clips += 1

                if (idx + 1) % 10 == 0 or idx == len(clips) - 1:
                    print(f"  [{idx+1}/{len(clips)}] latency={clip_latency:.1f}ms queue={queue.qsize()}")

        start_total = time.perf_counter()
        await asyncio.gather(producer(), consumer())
        result.total_time_sec = time.perf_counter() - start_total

        return result


def print_result(result: StreamBenchmarkResult) -> None:
    """Print formatted results."""
    print("\n" + "=" * 70)
    print(f"STREAM BENCHMARK RESULTS ({result.mode.upper()})")
    print("=" * 70)

    print(f"\nConfig:")
    print(f"  Video:      {result.video_path}")
    print(f"  Source FPS: {result.source_fps:.1f}")
    print(f"  Clip size:  {result.num_frames} frames, stride {result.stride}")
    print(f"  Device:     {result.device}")

    print(f"\nThroughput:")
    print(f"  Total clips:     {result.total_clips}")
    print(f"  Total time:      {result.total_time_sec:.2f}s")
    print(f"  Clips/sec:       {result.throughput_clips_per_sec:.2f}")
    print(f"  Effective FPS:   {result.effective_fps:.1f}")

    print(f"\nRealtime Analysis:")
    print(f"  Mean latency:    {result.mean_latency_ms:.1f}ms")
    print(f"  RT ratio:        {result.rt_ratio:.2f}x")

    clip_duration_ms = result.stride / result.source_fps * 1000
    if result.can_realtime:
        print(f"  Status:          ✓ CAN keep up with realtime")
        headroom = (1 - result.rt_ratio) * 100
        print(f"  Headroom:        {headroom:.1f}% spare capacity")
    else:
        print(f"  Status:          ✗ CANNOT keep up with realtime")
        shortfall = (result.rt_ratio - 1) * 100
        print(f"  Shortfall:       {shortfall:.1f}% too slow")
        needed_speedup = result.mean_latency_ms / clip_duration_ms
        print(f"  Need {needed_speedup:.1f}x speedup or stride >= {int(result.stride * result.rt_ratio)}")

    if result.queue_depths:
        print(f"\nQueue Stats (pipelined):")
        print(f"  Max depth:       {result.max_queue_depth}")
        avg_depth = sum(result.queue_depths) / len(result.queue_depths)
        print(f"  Avg depth:       {avg_depth:.1f}")


def compare_modes(seq: StreamBenchmarkResult, pipe: StreamBenchmarkResult) -> None:
    """Compare sequential vs pipelined results."""
    print("\n" + "=" * 70)
    print("MODE COMPARISON")
    print("=" * 70)

    print(f"\n  {'Metric':<25} {'Sequential':>12} {'Pipelined':>12} {'Speedup':>10}")
    print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*10}")

    metrics = [
        ("Total time (s)", seq.total_time_sec, pipe.total_time_sec),
        ("Throughput (clips/s)", seq.throughput_clips_per_sec, pipe.throughput_clips_per_sec),
        ("Effective FPS", seq.effective_fps, pipe.effective_fps),
        ("Mean latency (ms)", seq.mean_latency_ms, pipe.mean_latency_ms),
        ("RT ratio", seq.rt_ratio, pipe.rt_ratio),
    ]

    for name, seq_val, pipe_val in metrics:
        if "latency" in name.lower() or "ratio" in name.lower() or "time" in name.lower():
            # Lower is better
            speedup = seq_val / pipe_val if pipe_val > 0 else 0
        else:
            # Higher is better
            speedup = pipe_val / seq_val if seq_val > 0 else 0

        print(f"  {name:<25} {seq_val:>12.2f} {pipe_val:>12.2f} {speedup:>9.2f}x")

    print(f"\n  Pipeline benefit: ", end="")
    if pipe.throughput_clips_per_sec > seq.throughput_clips_per_sec:
        improvement = (pipe.throughput_clips_per_sec / seq.throughput_clips_per_sec - 1) * 100
        print(f"✓ {improvement:.1f}% higher throughput")
    else:
        print("✗ No improvement (inference-bound)")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark V-JEPA2 stream processing throughput",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--model", default="./model", help="Path to model directory")
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--max-clips", type=int, help="Limit number of clips to process")
    parser.add_argument("--pipeline", action="store_true", help="Use pipelined processing")
    parser.add_argument("--compare-modes", action="store_true", help="Compare sequential vs pipelined")
    parser.add_argument("--output", "-o", help="Save results to JSON")

    args = parser.parse_args()

    if not Path(args.video).exists():
        print(f"Error: Video not found: {args.video}", file=sys.stderr)
        sys.exit(1)

    benchmark = StreamBenchmark(
        model_path=args.model,
        device=args.device,
        num_frames=args.num_frames,
        stride=args.stride,
    )

    results = []

    if args.compare_modes:
        seq_result = benchmark.run_sequential(args.video, args.max_clips)
        print_result(seq_result)
        results.append(seq_result)

        pipe_result = benchmark.run_pipelined(args.video, args.max_clips)
        print_result(pipe_result)
        results.append(pipe_result)

        compare_modes(seq_result, pipe_result)
    elif args.pipeline:
        result = benchmark.run_pipelined(args.video, args.max_clips)
        print_result(result)
        results.append(result)
    else:
        result = benchmark.run_sequential(args.video, args.max_clips)
        print_result(result)
        results.append(result)

    if args.output:
        output_data = [r.to_dict() for r in results]
        with open(args.output, "w") as f:
            json.dump(output_data if len(output_data) > 1 else output_data[0], f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
