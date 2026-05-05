#!/usr/bin/env python3
"""
V-JEPA2 Session-Based Streaming Benchmark

Benchmarks the vLLM-Omni session-based World Model API which uses SSE streaming
for continuous prediction delivery.

API Flow:
    1. POST /v1/world/session - Create session with video source
    2. GET /v1/world/session/{id}/predictions - SSE stream of predictions
    3. DELETE /v1/world/session/{id} - Cleanup

Key metrics (JWMSP):
    - Per-prediction latency (time between predictions)
    - L_comp (compute latency from trace spans)
    - RTF (realtime factor)
    - Throughput (predictions/sec)

Usage:
    # Basic streaming benchmark
    python -m benchmark.benchmark_session \\
        --target http://localhost:8080 \\
        --video /path/to/video.mp4

    # With Jaeger trace collection
    python -m benchmark.benchmark_session \\
        --target http://localhost:8080 \\
        --video /path/to/video.mp4 \\
        --jaeger http://localhost:16686

    # Limit predictions for quick test
    python -m benchmark.benchmark_session \\
        --target http://localhost:8080 \\
        --video /path/to/video.mp4 \\
        --max-predictions 10
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from benchmark.jaeger_client import JaegerClient
from benchmark.metrics import (
    JWMSP_SPANS,
    JWSMPMetrics,
    StageMetrics,
    aggregate_span_durations,
    compute_jwmsp_metrics,
    format_latency_breakdown,
    format_latency_table,
)


@dataclass
class Prediction:
    """Single prediction from SSE stream."""

    sequence_num: int
    frame_range: tuple[int, int]
    timestamp_ns: int
    receive_time: float  # time.perf_counter() when received
    data: dict[str, Any]


@dataclass
class SessionBenchmarkConfig:
    """Configuration for session-based streaming benchmark."""

    target_url: str
    video_path: str
    source_type: str = "file"  # file, rtsp, v4l2, csi
    jaeger_url: str | None = None
    max_predictions: int | None = None  # None = process entire video
    num_frames: int = 16
    stride: int = 8
    source_fps: float = 30.0
    trace_flush_delay: float = 2.0
    session_timeout: float = 300.0  # max session duration
    insecure: bool = False


@dataclass
class SessionBenchmarkResult:
    """Results from session-based streaming benchmark."""

    config: SessionBenchmarkConfig
    session_id: str | None = None
    predictions: list[Prediction] = field(default_factory=list)
    stage_metrics: dict[str, StageMetrics] = field(default_factory=dict)
    jwmsp: JWSMPMetrics | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def prediction_count(self) -> int:
        return len(self.predictions)

    @property
    def inter_prediction_latencies_ms(self) -> list[float]:
        """Time between consecutive predictions."""
        if len(self.predictions) < 2:
            return []
        latencies = []
        for i in range(1, len(self.predictions)):
            delta = (
                self.predictions[i].receive_time - self.predictions[i - 1].receive_time
            ) * 1000
            latencies.append(delta)
        return latencies

    @property
    def mean_inter_prediction_ms(self) -> float:
        lats = self.inter_prediction_latencies_ms
        return sum(lats) / len(lats) if lats else 0.0

    @property
    def throughput_predictions_per_sec(self) -> float:
        if not self.predictions or not self.start_time or not self.end_time:
            return 0.0
        duration = (self.end_time - self.start_time).total_seconds()
        return self.prediction_count / duration if duration > 0 else 0.0

    @property
    def effective_fps(self) -> float:
        """Effective source FPS we can sustain."""
        return self.throughput_predictions_per_sec * self.config.stride

    @property
    def rt_ratio(self) -> float:
        """Realtime ratio: <1 means keeping up, >1 means falling behind."""
        clip_duration_ms = (self.config.stride / self.config.source_fps) * 1000
        return self.mean_inter_prediction_ms / clip_duration_ms if clip_duration_ms > 0 else 0

    @property
    def can_realtime(self) -> bool:
        return self.rt_ratio <= 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": {
                "target_url": self.config.target_url,
                "video_path": self.config.video_path,
                "source_type": self.config.source_type,
                "num_frames": self.config.num_frames,
                "stride": self.config.stride,
                "source_fps": self.config.source_fps,
            },
            "session": {
                "session_id": self.session_id,
                "start_time": (
                    self.start_time.isoformat() if self.start_time else None
                ),
                "end_time": self.end_time.isoformat() if self.end_time else None,
                "error": self.error,
            },
            "summary": {
                "prediction_count": self.prediction_count,
                "throughput_predictions_per_sec": round(
                    self.throughput_predictions_per_sec, 3
                ),
                "effective_fps": round(self.effective_fps, 1),
                "mean_inter_prediction_ms": round(self.mean_inter_prediction_ms, 3),
                "rt_ratio": round(self.rt_ratio, 3),
                "can_realtime": self.can_realtime,
            },
            "inter_prediction_latencies_ms": [
                round(lat, 3) for lat in self.inter_prediction_latencies_ms
            ],
            "stages": {
                name: metrics.to_dict()
                for name, metrics in self.stage_metrics.items()
            },
            "methodology": self.jwmsp.to_dict() if self.jwmsp else {},
            "metadata": self.metadata,
        }


class SessionClient:
    """Client for vLLM-Omni World Model Session API."""

    def __init__(
        self,
        target_url: str,
        timeout: float = 300.0,
        insecure: bool = False,
    ):
        self.target_url = target_url.rstrip("/")
        self.timeout = timeout
        self.insecure = insecure
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "SessionClient":
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.timeout, connect=10.0),
            verify=not self.insecure,
        )
        return self

    async def __aexit__(self, *args: Any) -> None:
        if self._client:
            await self._client.aclose()

    async def create_session(
        self,
        source_type: str,
        source_uri: str,
        num_frames: int = 16,
        stride: int = 8,
    ) -> str:
        """Create a new session and return session_id."""
        if not self._client:
            raise RuntimeError("SessionClient not initialized")

        url = f"{self.target_url}/v1/world/session"
        payload = {
            "source": {
                "type": source_type,
                "uri": source_uri,
            },
            "config": {
                "num_frames": num_frames,
                "stride": stride,
            },
        }

        response = await self._client.post(url, json=payload)
        response.raise_for_status()
        data = response.json()
        return data["session_id"]

    async def stream_predictions(
        self,
        session_id: str,
        on_prediction: Any = None,
        max_predictions: int | None = None,
    ) -> list[Prediction]:
        """Stream predictions via SSE and return all received predictions."""
        if not self._client:
            raise RuntimeError("SessionClient not initialized")

        url = f"{self.target_url}/v1/world/session/{session_id}/predictions"
        predictions: list[Prediction] = []
        sequence_num = 0

        async with self._client.stream(
            "GET",
            url,
            headers={"Accept": "text/event-stream"},
        ) as response:
            response.raise_for_status()

            async for line in response.aiter_lines():
                # SSE format: "data: {...json...}"
                if not line.startswith("data: "):
                    continue

                json_data = line[6:]  # Strip "data: " prefix
                if json_data == "[DONE]":
                    break

                receive_time = time.perf_counter()
                data = json.loads(json_data)

                prediction = Prediction(
                    sequence_num=sequence_num,
                    frame_range=tuple(data.get("frame_range", [0, 0])),
                    timestamp_ns=data.get("timestamp_ns", 0),
                    receive_time=receive_time,
                    data=data,
                )
                predictions.append(prediction)
                sequence_num += 1

                if on_prediction:
                    on_prediction(prediction)

                if max_predictions and sequence_num >= max_predictions:
                    break

        return predictions

    async def delete_session(self, session_id: str) -> None:
        """Delete/stop a session."""
        if not self._client:
            raise RuntimeError("SessionClient not initialized")

        url = f"{self.target_url}/v1/world/session/{session_id}"
        response = await self._client.delete(url)
        response.raise_for_status()

    async def get_session_status(self, session_id: str) -> dict[str, Any]:
        """Get session status (for debugging/reconnection)."""
        if not self._client:
            raise RuntimeError("SessionClient not initialized")

        url = f"{self.target_url}/v1/world/session/{session_id}"
        response = await self._client.get(url)
        response.raise_for_status()
        return response.json()


async def run_session_benchmark(
    config: SessionBenchmarkConfig,
) -> SessionBenchmarkResult:
    """Run the session-based streaming benchmark.

    1. Create session with video source
    2. Stream predictions via SSE
    3. Wait for traces to flush
    4. Query Jaeger for span timing
    5. Compute JWMSP metrics
    """
    result = SessionBenchmarkResult(config=config)

    # Validate inputs
    if config.source_type == "file" and not Path(config.video_path).exists():
        raise FileNotFoundError(f"Video not found: {config.video_path}")

    # Build source URI
    if config.source_type == "file":
        source_uri = f"file://{Path(config.video_path).absolute()}"
    else:
        source_uri = config.video_path  # Assume it's already a URI

    print(f"Target: {config.target_url}")
    print(f"Source: {config.source_type}:{source_uri}")
    print(f"Clip config: {config.num_frames} frames, stride {config.stride}")

    # Check Jaeger availability
    jaeger: JaegerClient | None = None
    if config.jaeger_url:
        jaeger = JaegerClient(config.jaeger_url)
        if jaeger.health_check():
            print(f"Jaeger: {config.jaeger_url} (connected)")
        else:
            print(f"Jaeger: {config.jaeger_url} (not reachable, skipping traces)")
            jaeger = None

    async with SessionClient(
        config.target_url,
        timeout=config.session_timeout,
        insecure=config.insecure,
    ) as client:
        # Create session
        print(f"\nCreating session...")
        try:
            result.session_id = await client.create_session(
                source_type=config.source_type,
                source_uri=source_uri,
                num_frames=config.num_frames,
                stride=config.stride,
            )
            print(f"Session created: {result.session_id}")
        except httpx.HTTPStatusError as e:
            result.error = f"Failed to create session: {e.response.status_code}"
            print(f"Error: {result.error}")
            return result

        # Stream predictions
        print(f"\nStreaming predictions...")
        result.start_time = datetime.now(timezone.utc)

        def on_prediction(pred: Prediction) -> None:
            frame_start, frame_end = pred.frame_range
            data = pred.data
            # Print preview of prediction data
            if "predictions" in data and data["predictions"]:
                top = data["predictions"][0]
                label = top.get("label", "?")
                score = top.get("score", 0.0)
                print(
                    f"  [pred {pred.sequence_num:3d}] "
                    f"frames [{frame_start:4d}-{frame_end:4d}] "
                    f"top: {label} ({score:.2%})"
                )
            else:
                print(f"  [pred {pred.sequence_num:3d}] frames [{frame_start:4d}-{frame_end:4d}]")

        try:
            result.predictions = await client.stream_predictions(
                result.session_id,
                on_prediction=on_prediction,
                max_predictions=config.max_predictions,
            )
        except Exception as e:
            result.error = f"Streaming error: {e}"
            print(f"Error: {result.error}")

        result.end_time = datetime.now(timezone.utc)

        # Cleanup session
        print(f"\nCleaning up session...")
        try:
            await client.delete_session(result.session_id)
            print("Session deleted.")
        except Exception as e:
            print(f"Warning: Failed to delete session: {e}")

    # Collect traces from Jaeger
    if jaeger and result.predictions:
        print(f"\nWaiting {config.trace_flush_delay}s for traces to flush...")
        await asyncio.sleep(config.trace_flush_delay)

        print("Collecting traces from Jaeger...")
        traces = jaeger.wait_for_traces(
            service="vllm-omni-jepa",  # Updated service name for vLLM-Omni
            start_time=result.start_time,
            expected_count=result.prediction_count,
            timeout=30.0,
        )
        print(f"  Found {len(traces)} traces")

        if traces:
            durations = jaeger.extract_span_durations(traces, JWMSP_SPANS)
            result.stage_metrics = aggregate_span_durations(durations)
            result.jwmsp = compute_jwmsp_metrics(
                result.stage_metrics,
                num_frames=config.num_frames,
                source_fps=config.source_fps,
            )

        jaeger.close()

    return result


def print_results(result: SessionBenchmarkResult) -> None:
    """Print formatted benchmark results."""
    print("\n" + "=" * 70)
    print("SESSION STREAMING BENCHMARK RESULTS")
    print("=" * 70)

    print(f"\nSession:")
    print(f"  Session ID:   {result.session_id}")
    if result.error:
        print(f"  Error:        {result.error}")
    else:
        print(f"  Status:       Completed")

    print(f"\nStreaming Metrics:")
    print(f"  Predictions:     {result.prediction_count}")
    print(f"  Throughput:      {result.throughput_predictions_per_sec:.2f} predictions/sec")
    print(f"  Effective FPS:   {result.effective_fps:.1f}")

    if result.inter_prediction_latencies_ms:
        lats = result.inter_prediction_latencies_ms
        print(f"\nInter-Prediction Latency:")
        print(f"  Mean:            {result.mean_inter_prediction_ms:.1f}ms")
        print(f"  Min:             {min(lats):.1f}ms")
        print(f"  Max:             {max(lats):.1f}ms")
        if len(lats) > 1:
            import statistics
            print(f"  Std:             {statistics.stdev(lats):.1f}ms")

    print(f"\nRealtime Analysis:")
    clip_duration_ms = (result.config.stride / result.config.source_fps) * 1000
    print(f"  Clip duration:   {clip_duration_ms:.1f}ms (stride={result.config.stride} @ {result.config.source_fps}fps)")
    print(f"  RT ratio:        {result.rt_ratio:.2f}x")

    if result.can_realtime:
        headroom = (1 - result.rt_ratio) * 100
        print(f"  Status:          REALTIME CAPABLE")
        print(f"  Headroom:        {headroom:.1f}% spare capacity")
    else:
        shortfall = (result.rt_ratio - 1) * 100
        print(f"  Status:          BELOW REALTIME")
        print(f"  Shortfall:       {shortfall:.1f}% too slow")

    if result.stage_metrics:
        print(f"\nPipeline Latency Breakdown:")
        print(format_latency_table(result.stage_metrics))

        print(f"\nLatency Distribution:")
        print(format_latency_breakdown(result.stage_metrics))

    if result.jwmsp:
        print(f"\nJWSMP Methodology Metrics:")
        print(f"  L_comp (compute):  {result.jwmsp.l_comp_ms:.1f}ms")
        print(f"  L_algo (video):    {result.jwmsp.l_algo_ms:.1f}ms")
        print(f"  L_sys (estimated): {result.jwmsp.l_sys_ms:.1f}ms")
        print(f"  RTF:               {result.jwmsp.rtf:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Session-based streaming benchmark for vLLM-Omni World Model API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic streaming benchmark
  python -m benchmark.benchmark_session \\
      --target http://localhost:8080 \\
      --video /path/to/video.mp4

  # With Jaeger trace collection
  python -m benchmark.benchmark_session \\
      --target http://localhost:8080 \\
      --video /path/to/video.mp4 \\
      --jaeger http://localhost:16686

  # RTSP source (streaming camera)
  python -m benchmark.benchmark_session \\
      --target http://localhost:8080 \\
      --video rtsp://camera.local/stream \\
      --source-type rtsp
        """,
    )

    parser.add_argument(
        "--target",
        required=True,
        help="Target API URL (e.g., http://localhost:8080)",
    )
    parser.add_argument(
        "--video",
        required=True,
        help="Path to video file or streaming URI",
    )
    parser.add_argument(
        "--source-type",
        choices=["file", "rtsp", "v4l2", "csi"],
        default="file",
        help="Source type (default: file)",
    )
    parser.add_argument(
        "--jaeger",
        help="Jaeger Query API URL for trace collection",
    )
    parser.add_argument(
        "--max-predictions",
        type=int,
        help="Maximum predictions to collect (default: all)",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=16,
        help="Frames per clip (default: 16)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=8,
        help="Frame stride between clips (default: 8)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Source video FPS for RTF calculation (default: 30.0)",
    )
    parser.add_argument(
        "--trace-delay",
        type=float,
        default=2.0,
        help="Seconds to wait for traces to flush (default: 2.0)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="Session timeout in seconds (default: 300)",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Save results to JSON file",
    )
    parser.add_argument(
        "--insecure",
        "-k",
        action="store_true",
        help="Skip SSL certificate verification",
    )

    args = parser.parse_args()

    config = SessionBenchmarkConfig(
        target_url=args.target,
        video_path=args.video,
        source_type=args.source_type,
        jaeger_url=args.jaeger,
        max_predictions=args.max_predictions,
        num_frames=args.num_frames,
        stride=args.stride,
        source_fps=args.fps,
        trace_flush_delay=args.trace_delay,
        session_timeout=args.timeout,
        insecure=args.insecure,
    )

    try:
        result = asyncio.run(run_session_benchmark(config))
        print_results(result)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except httpx.ConnectError:
        print(f"Error: Cannot connect to {args.target}", file=sys.stderr)
        sys.exit(1)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
