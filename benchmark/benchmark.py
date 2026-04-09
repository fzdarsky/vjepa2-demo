#!/usr/bin/env python3
"""
V-JEPA2 External Benchmark (Load Generator + Trace Collection)

Sends requests to the V-JEPA2 HTTP API and collects timing from OTel traces
via Jaeger. Works with any deployment: local, Podman, Kubernetes.

This is the primary benchmark for production testing. For in-process
development testing, use benchmark_whitebox.py instead.

Usage:
    # Basic benchmark against local server
    python -m benchmark.benchmark \\
        --target http://localhost:8000 \\
        --video samples/sample.mp4

    # With Jaeger trace collection
    python -m benchmark.benchmark \\
        --target http://localhost:8000 \\
        --video samples/sample.mp4 \\
        --jaeger http://localhost:16686

    # Concurrent load test
    python -m benchmark.benchmark \\
        --target http://localhost:8000 \\
        --video samples/sample.mp4 \\
        --requests 100 \\
        --concurrency 4
"""

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
class RequestResult:
    """Result of a single HTTP request."""

    success: bool
    status_code: int
    latency_ms: float
    error: str | None = None
    response_body: dict | None = None


@dataclass
class BenchmarkConfig:
    """Benchmark configuration."""

    target_url: str
    video_path: str
    num_requests: int = 50
    concurrency: int = 1
    jaeger_url: str | None = None
    warmup_requests: int = 5
    trace_flush_delay: float = 2.0  # seconds to wait for traces
    num_frames: int = 16
    source_fps: float = 30.0


@dataclass
class BenchmarkResult:
    """Complete benchmark results."""

    config: BenchmarkConfig
    request_results: list[RequestResult] = field(default_factory=list)
    stage_metrics: dict[str, StageMetrics] = field(default_factory=dict)
    jwmsp: JWSMPMetrics | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def success_count(self) -> int:
        return sum(1 for r in self.request_results if r.success)

    @property
    def error_count(self) -> int:
        return sum(1 for r in self.request_results if not r.success)

    @property
    def success_rate(self) -> float:
        if not self.request_results:
            return 0.0
        return self.success_count / len(self.request_results)

    @property
    def mean_latency_ms(self) -> float:
        latencies = [r.latency_ms for r in self.request_results if r.success]
        return sum(latencies) / len(latencies) if latencies else 0.0

    @property
    def throughput_rps(self) -> float:
        if not self.start_time or not self.end_time:
            return 0.0
        duration = (self.end_time - self.start_time).total_seconds()
        if duration <= 0:
            return 0.0
        return self.success_count / duration

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": {
                "target_url": self.config.target_url,
                "video_path": self.config.video_path,
                "num_requests": self.config.num_requests,
                "concurrency": self.config.concurrency,
                "warmup_requests": self.config.warmup_requests,
                "num_frames": self.config.num_frames,
                "source_fps": self.config.source_fps,
            },
            "summary": {
                "total_requests": len(self.request_results),
                "successful": self.success_count,
                "failed": self.error_count,
                "success_rate": round(self.success_rate, 4),
                "mean_latency_ms": round(self.mean_latency_ms, 3),
                "throughput_rps": round(self.throughput_rps, 3),
            },
            "stages": {
                name: metrics.to_dict()
                for name, metrics in self.stage_metrics.items()
            },
            "methodology": self.jwmsp.to_dict() if self.jwmsp else {},
            "metadata": self.metadata,
        }


class LoadGenerator:
    """Async HTTP load generator for V-JEPA2 API."""

    def __init__(self, target_url: str, timeout: float = 120.0):
        self.target_url = target_url.rstrip("/")
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "LoadGenerator":
        self._client = httpx.AsyncClient(timeout=self.timeout)
        return self

    async def __aexit__(self, *args: Any) -> None:
        if self._client:
            await self._client.aclose()

    async def send_request(self, video_path: str) -> RequestResult:
        """Send a single inference request."""
        if not self._client:
            raise RuntimeError("LoadGenerator not initialized")

        url = f"{self.target_url}/v2/models/vjepa2/infer"
        start = time.perf_counter()

        try:
            with open(video_path, "rb") as f:
                files = {"file": (Path(video_path).name, f, "video/mp4")}
                response = await self._client.post(url, files=files)

            latency_ms = (time.perf_counter() - start) * 1000

            if response.status_code == 200:
                return RequestResult(
                    success=True,
                    status_code=response.status_code,
                    latency_ms=latency_ms,
                    response_body=response.json(),
                )
            else:
                return RequestResult(
                    success=False,
                    status_code=response.status_code,
                    latency_ms=latency_ms,
                    error=response.text[:500],
                )

        except Exception as e:
            latency_ms = (time.perf_counter() - start) * 1000
            return RequestResult(
                success=False,
                status_code=0,
                latency_ms=latency_ms,
                error=str(e),
            )

    async def run_concurrent(
        self,
        video_path: str,
        num_requests: int,
        concurrency: int,
        on_complete: Any = None,
    ) -> list[RequestResult]:
        """Run concurrent requests with bounded concurrency."""
        semaphore = asyncio.Semaphore(concurrency)
        results: list[RequestResult] = []

        async def bounded_request(idx: int) -> RequestResult:
            async with semaphore:
                result = await self.send_request(video_path)
                if on_complete:
                    on_complete(idx, result)
                return result

        tasks = [bounded_request(i) for i in range(num_requests)]
        results = await asyncio.gather(*tasks)
        return list(results)


async def run_benchmark(config: BenchmarkConfig) -> BenchmarkResult:
    """Run the complete benchmark.

    1. Send warmup requests (not counted)
    2. Send timed requests with concurrency
    3. Wait for traces to flush
    4. Query Jaeger for span timing
    5. Compute JWMSP metrics
    """
    result = BenchmarkResult(config=config)

    # Validate inputs
    if not Path(config.video_path).exists():
        raise FileNotFoundError(f"Video not found: {config.video_path}")

    print(f"Target: {config.target_url}")
    print(f"Video: {config.video_path}")
    print(f"Requests: {config.num_requests} (concurrency={config.concurrency})")

    # Check Jaeger availability
    jaeger: JaegerClient | None = None
    if config.jaeger_url:
        jaeger = JaegerClient(config.jaeger_url)
        if jaeger.health_check():
            print(f"Jaeger: {config.jaeger_url} (connected)")
        else:
            print(f"Jaeger: {config.jaeger_url} (not reachable, skipping traces)")
            jaeger = None

    async with LoadGenerator(config.target_url) as loader:
        # Warmup
        if config.warmup_requests > 0:
            print(f"\nWarmup: {config.warmup_requests} requests...")
            for i in range(config.warmup_requests):
                r = await loader.send_request(config.video_path)
                status = "OK" if r.success else f"ERR({r.status_code})"
                print(f"  [warmup {i}] {r.latency_ms:.1f}ms {status}")

        # Timed requests
        print(f"\nBenchmark: {config.num_requests} requests...")
        result.start_time = datetime.now(timezone.utc)

        def on_complete(idx: int, r: RequestResult) -> None:
            status = "OK" if r.success else f"ERR({r.status_code})"
            print(f"  [req {idx:3d}] {r.latency_ms:7.1f}ms {status}")

        result.request_results = await loader.run_concurrent(
            config.video_path,
            config.num_requests,
            config.concurrency,
            on_complete,
        )

        result.end_time = datetime.now(timezone.utc)

    # Collect traces from Jaeger
    if jaeger:
        print(f"\nWaiting {config.trace_flush_delay}s for traces to flush...")
        await asyncio.sleep(config.trace_flush_delay)

        print("Collecting traces from Jaeger...")
        traces = jaeger.wait_for_traces(
            service="vjepa2-server",
            start_time=result.start_time,
            expected_count=config.num_requests,
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


def print_results(result: BenchmarkResult) -> None:
    """Print formatted benchmark results."""
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)

    print(f"\nHTTP Metrics:")
    print(f"  Requests:    {len(result.request_results)} total")
    print(f"  Successful:  {result.success_count} ({result.success_rate*100:.1f}%)")
    print(f"  Failed:      {result.error_count}")
    print(f"  Mean Latency: {result.mean_latency_ms:.1f}ms")
    print(f"  Throughput:   {result.throughput_rps:.2f} req/sec")

    if result.stage_metrics:
        print(f"\nPipeline Latency Breakdown:")
        print(format_latency_table(result.stage_metrics))

        print(f"\nLatency Distribution:")
        print(format_latency_breakdown(result.stage_metrics))

    if result.jwmsp:
        print(f"\nJWSMP Methodology Metrics:")
        print(f"  L_comp (compute):  {result.jwmsp.l_comp_ms:.1f}ms")
        print(f"  L_algo (video):    {result.jwmsp.l_algo_ms:.1f}ms")
        print(f"  L_sys (total lag): {result.jwmsp.l_sys_ms:.1f}ms")
        print(f"  RTF:               {result.jwmsp.rtf:.2f}")
        if result.jwmsp.rtf >= 1.0:
            print(f"  Status:            Real-time capable")
        else:
            print(f"  Status:            Below real-time ({1/result.jwmsp.rtf:.1f}x slower)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="External benchmark for V-JEPA2 inference API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic benchmark
  python -m benchmark.benchmark --target http://localhost:8000 --video samples/sample.mp4

  # With Jaeger trace collection
  python -m benchmark.benchmark \\
      --target http://localhost:8000 \\
      --video samples/sample.mp4 \\
      --jaeger http://localhost:16686

  # High concurrency load test
  python -m benchmark.benchmark \\
      --target http://localhost:8000 \\
      --video samples/sample.mp4 \\
      --requests 100 --concurrency 8
        """,
    )

    parser.add_argument(
        "--target",
        required=True,
        help="Target API URL (e.g., http://localhost:8000)",
    )
    parser.add_argument(
        "--video",
        required=True,
        help="Path to video file for testing",
    )
    parser.add_argument(
        "--requests",
        "-n",
        type=int,
        default=50,
        help="Number of requests to send (default: 50)",
    )
    parser.add_argument(
        "--concurrency",
        "-c",
        type=int,
        default=1,
        help="Concurrent requests (default: 1)",
    )
    parser.add_argument(
        "--jaeger",
        help="Jaeger Query API URL for trace collection",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Warmup requests before timing (default: 5)",
    )
    parser.add_argument(
        "--trace-delay",
        type=float,
        default=2.0,
        help="Seconds to wait for traces to flush (default: 2.0)",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=16,
        help="Frames per clip for RTF calculation (default: 16)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Source video FPS for RTF calculation (default: 30.0)",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Save results to JSON file",
    )

    args = parser.parse_args()

    config = BenchmarkConfig(
        target_url=args.target,
        video_path=args.video,
        num_requests=args.requests,
        concurrency=args.concurrency,
        jaeger_url=args.jaeger,
        warmup_requests=args.warmup,
        trace_flush_delay=args.trace_delay,
        num_frames=args.num_frames,
        source_fps=args.fps,
    )

    try:
        result = asyncio.run(run_benchmark(config))
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except httpx.ConnectError:
        print(f"Error: Cannot connect to {args.target}", file=sys.stderr)
        sys.exit(1)

    print_results(result)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
