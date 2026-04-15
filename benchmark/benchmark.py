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
    # Saturation test settings
    concurrency_levels: list[int] = field(default_factory=lambda: [1, 2, 4, 8, 16])
    requests_per_level: int = 20
    # Soak test settings
    duration_seconds: float = 300.0  # 5 minutes default
    report_interval: float = 30.0  # report every 30s
    # Timestamp injection for true L_sys
    inject_timestamp: bool = False
    # SSL verification
    insecure: bool = False


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


@dataclass
class SaturationPoint:
    """Results at a single concurrency level."""

    concurrency: int
    throughput_rps: float
    mean_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    error_rate: float


@dataclass
class SaturationResult:
    """Results from saturation test."""

    config: BenchmarkConfig
    points: list[SaturationPoint] = field(default_factory=list)
    saturation_concurrency: int | None = None  # where latency spikes
    peak_throughput_rps: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": "saturation",
            "config": {
                "target_url": self.config.target_url,
                "video_path": self.config.video_path,
                "concurrency_levels": self.config.concurrency_levels,
                "requests_per_level": self.config.requests_per_level,
            },
            "points": [
                {
                    "concurrency": p.concurrency,
                    "throughput_rps": round(p.throughput_rps, 3),
                    "mean_latency_ms": round(p.mean_latency_ms, 3),
                    "p95_latency_ms": round(p.p95_latency_ms, 3),
                    "p99_latency_ms": round(p.p99_latency_ms, 3),
                    "error_rate": round(p.error_rate, 4),
                }
                for p in self.points
            ],
            "saturation_concurrency": self.saturation_concurrency,
            "peak_throughput_rps": round(self.peak_throughput_rps, 3),
        }


@dataclass
class SoakInterval:
    """Metrics for a single reporting interval during soak test."""

    interval_num: int
    elapsed_seconds: float
    requests_completed: int
    throughput_rps: float
    mean_latency_ms: float
    p95_latency_ms: float
    error_rate: float
    rtf: float | None = None


@dataclass
class SoakResult:
    """Results from soak test."""

    config: BenchmarkConfig
    intervals: list[SoakInterval] = field(default_factory=list)
    total_requests: int = 0
    total_errors: int = 0
    start_time: datetime | None = None
    end_time: datetime | None = None
    # Drift metrics
    rtf_drift: float = 0.0  # change in RTF over test duration
    latency_jitter_ms: float = 0.0  # std of interval mean latencies

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": "soak",
            "config": {
                "target_url": self.config.target_url,
                "video_path": self.config.video_path,
                "duration_seconds": self.config.duration_seconds,
                "report_interval": self.config.report_interval,
                "concurrency": self.config.concurrency,
            },
            "summary": {
                "total_requests": self.total_requests,
                "total_errors": self.total_errors,
                "duration_seconds": (
                    (self.end_time - self.start_time).total_seconds()
                    if self.start_time and self.end_time
                    else 0
                ),
                "rtf_drift": round(self.rtf_drift, 4),
                "latency_jitter_ms": round(self.latency_jitter_ms, 3),
            },
            "intervals": [
                {
                    "interval": i.interval_num,
                    "elapsed_s": round(i.elapsed_seconds, 1),
                    "requests": i.requests_completed,
                    "throughput_rps": round(i.throughput_rps, 3),
                    "mean_latency_ms": round(i.mean_latency_ms, 3),
                    "p95_latency_ms": round(i.p95_latency_ms, 3),
                    "error_rate": round(i.error_rate, 4),
                    "rtf": round(i.rtf, 3) if i.rtf else None,
                }
                for i in self.intervals
            ],
        }


class LoadGenerator:
    """Async HTTP load generator for V-JEPA2 API."""

    def __init__(self, target_url: str, timeout: float = 120.0, insecure: bool = False):
        self.target_url = target_url.rstrip("/")
        self.timeout = timeout
        self.insecure = insecure
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "LoadGenerator":
        self._client = httpx.AsyncClient(
            timeout=self.timeout,
            verify=not self.insecure,  # Skip SSL verification if insecure=True
        )
        return self

    async def __aexit__(self, *args: Any) -> None:
        if self._client:
            await self._client.aclose()

    async def send_request(
        self, video_path: str, inject_timestamp: bool = False
    ) -> RequestResult:
        """Send a single inference request.

        Args:
            video_path: Path to video file
            inject_timestamp: If True, include observation timestamp for L_sys calculation
        """
        if not self._client:
            raise RuntimeError("LoadGenerator not initialized")

        url = f"{self.target_url}/v2/models/vjepa2/infer"
        start = time.perf_counter()

        try:
            with open(video_path, "rb") as f:
                files = {"file": (Path(video_path).name, f, "video/mp4")}
                # Include observation timestamp for true L_sys measurement
                data = {}
                if inject_timestamp:
                    # Milliseconds since epoch
                    data["obs_timestamp_ms"] = str(int(time.time() * 1000))
                response = await self._client.post(url, files=files, data=data)

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
        inject_timestamp: bool = False,
    ) -> list[RequestResult]:
        """Run concurrent requests with bounded concurrency."""
        semaphore = asyncio.Semaphore(concurrency)
        results: list[RequestResult] = []

        async def bounded_request(idx: int) -> RequestResult:
            async with semaphore:
                result = await self.send_request(video_path, inject_timestamp)
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
    if config.inject_timestamp:
        print(f"Timestamp injection: enabled (for true L_sys measurement)")

    # Check Jaeger availability
    jaeger: JaegerClient | None = None
    if config.jaeger_url:
        jaeger = JaegerClient(config.jaeger_url)
        if jaeger.health_check():
            print(f"Jaeger: {config.jaeger_url} (connected)")
        else:
            print(f"Jaeger: {config.jaeger_url} (not reachable, skipping traces)")
            jaeger = None

    async with LoadGenerator(config.target_url, insecure=config.insecure) as loader:
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
            inject_timestamp=config.inject_timestamp,
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

            # Extract true L_sys if timestamps were injected
            if config.inject_timestamp:
                l_sys_values = jaeger.extract_l_sys(traces)
                if l_sys_values:
                    import statistics
                    result.metadata["l_sys_true"] = {
                        "count": len(l_sys_values),
                        "mean_ms": round(statistics.mean(l_sys_values), 3),
                        "min_ms": round(min(l_sys_values), 3),
                        "max_ms": round(max(l_sys_values), 3),
                        "std_ms": round(statistics.stdev(l_sys_values), 3) if len(l_sys_values) > 1 else 0,
                    }

        jaeger.close()

    return result


async def run_saturation_test(config: BenchmarkConfig) -> SaturationResult:
    """Run saturation test: ramp up concurrency to find throughput ceiling.

    Tests each concurrency level and measures throughput vs latency tradeoff.
    Identifies the saturation point where latency starts spiking.
    """
    result = SaturationResult(config=config)

    if not Path(config.video_path).exists():
        raise FileNotFoundError(f"Video not found: {config.video_path}")

    print(f"Target: {config.target_url}")
    print(f"Video: {config.video_path}")
    print(f"Concurrency levels: {config.concurrency_levels}")
    print(f"Requests per level: {config.requests_per_level}")

    async with LoadGenerator(config.target_url, insecure=config.insecure) as loader:
        # Warmup
        if config.warmup_requests > 0:
            print(f"\nWarmup: {config.warmup_requests} requests...")
            for _ in range(config.warmup_requests):
                await loader.send_request(config.video_path)

        print(f"\nSaturation test:")
        print(f"{'Conc':>6} {'Throughput':>12} {'Mean':>10} {'P95':>10} {'P99':>10} {'Errors':>8}")
        print("-" * 62)

        prev_mean_latency = 0.0
        for concurrency in config.concurrency_levels:
            start = time.perf_counter()
            results = await loader.run_concurrent(
                config.video_path,
                config.requests_per_level,
                concurrency,
            )
            elapsed = time.perf_counter() - start

            # Calculate metrics
            successful = [r for r in results if r.success]
            latencies = sorted([r.latency_ms for r in successful])

            if not latencies:
                print(f"{concurrency:>6} {'N/A':>12} {'N/A':>10} {'N/A':>10} {'N/A':>10} {len(results):>8}")
                continue

            mean_lat = sum(latencies) / len(latencies)
            p95_idx = min(int(len(latencies) * 0.95), len(latencies) - 1)
            p99_idx = min(int(len(latencies) * 0.99), len(latencies) - 1)
            p95_lat = latencies[p95_idx]
            p99_lat = latencies[p99_idx]
            throughput = len(successful) / elapsed
            error_rate = (len(results) - len(successful)) / len(results)

            point = SaturationPoint(
                concurrency=concurrency,
                throughput_rps=throughput,
                mean_latency_ms=mean_lat,
                p95_latency_ms=p95_lat,
                p99_latency_ms=p99_lat,
                error_rate=error_rate,
            )
            result.points.append(point)

            # Track peak throughput
            if throughput > result.peak_throughput_rps:
                result.peak_throughput_rps = throughput

            # Detect saturation (latency doubles from previous level)
            if (
                result.saturation_concurrency is None
                and prev_mean_latency > 0
                and mean_lat > prev_mean_latency * 2
            ):
                result.saturation_concurrency = concurrency

            prev_mean_latency = mean_lat

            print(
                f"{concurrency:>6} {throughput:>10.2f}/s {mean_lat:>9.1f}ms "
                f"{p95_lat:>9.1f}ms {p99_lat:>9.1f}ms {error_rate*100:>7.1f}%"
            )

    return result


async def run_soak_test(config: BenchmarkConfig) -> SoakResult:
    """Run soak test: sustained load over time to detect drift.

    Runs continuously for the specified duration, reporting metrics
    at regular intervals. Tracks RTF drift and latency jitter.
    """
    result = SoakResult(config=config)

    if not Path(config.video_path).exists():
        raise FileNotFoundError(f"Video not found: {config.video_path}")

    print(f"Target: {config.target_url}")
    print(f"Video: {config.video_path}")
    print(f"Duration: {config.duration_seconds}s")
    print(f"Report interval: {config.report_interval}s")
    print(f"Concurrency: {config.concurrency}")

    # Calculate L_algo for RTF
    l_algo_ms = (config.num_frames / config.source_fps) * 1000

    async with LoadGenerator(config.target_url, insecure=config.insecure) as loader:
        # Warmup
        if config.warmup_requests > 0:
            print(f"\nWarmup: {config.warmup_requests} requests...")
            for _ in range(config.warmup_requests):
                await loader.send_request(config.video_path)

        print(f"\nSoak test:")
        print(f"{'Interval':>8} {'Elapsed':>10} {'Reqs':>8} {'Throughput':>12} {'Mean':>10} {'P95':>10} {'RTF':>8}")
        print("-" * 76)

        result.start_time = datetime.now(timezone.utc)
        test_start = time.perf_counter()
        interval_num = 0
        interval_latencies: list[float] = []
        interval_errors = 0
        interval_start = test_start

        # Continuous request loop with reporting
        while True:
            elapsed = time.perf_counter() - test_start
            if elapsed >= config.duration_seconds:
                break

            # Send batch of concurrent requests
            batch_results = await loader.run_concurrent(
                config.video_path,
                config.concurrency,  # one batch per concurrency level
                config.concurrency,
            )

            for r in batch_results:
                result.total_requests += 1
                if r.success:
                    interval_latencies.append(r.latency_ms)
                else:
                    result.total_errors += 1
                    interval_errors += 1

            # Check if we should report
            interval_elapsed = time.perf_counter() - interval_start
            if interval_elapsed >= config.report_interval:
                interval_num += 1
                total_elapsed = time.perf_counter() - test_start

                if interval_latencies:
                    sorted_lats = sorted(interval_latencies)
                    mean_lat = sum(sorted_lats) / len(sorted_lats)
                    p95_idx = min(int(len(sorted_lats) * 0.95), len(sorted_lats) - 1)
                    p95_lat = sorted_lats[p95_idx]
                    throughput = len(interval_latencies) / interval_elapsed
                    error_rate = interval_errors / (len(interval_latencies) + interval_errors)
                    rtf = l_algo_ms / mean_lat if mean_lat > 0 else None

                    interval = SoakInterval(
                        interval_num=interval_num,
                        elapsed_seconds=total_elapsed,
                        requests_completed=len(interval_latencies),
                        throughput_rps=throughput,
                        mean_latency_ms=mean_lat,
                        p95_latency_ms=p95_lat,
                        error_rate=error_rate,
                        rtf=rtf,
                    )
                    result.intervals.append(interval)

                    print(
                        f"{interval_num:>8} {total_elapsed:>9.1f}s {len(interval_latencies):>8} "
                        f"{throughput:>10.2f}/s {mean_lat:>9.1f}ms {p95_lat:>9.1f}ms "
                        f"{rtf:>7.2f}" if rtf else "N/A"
                    )

                # Reset interval counters
                interval_latencies = []
                interval_errors = 0
                interval_start = time.perf_counter()

        result.end_time = datetime.now(timezone.utc)

        # Calculate drift metrics
        if len(result.intervals) >= 2:
            rtfs = [i.rtf for i in result.intervals if i.rtf is not None]
            if len(rtfs) >= 2:
                result.rtf_drift = rtfs[-1] - rtfs[0]

            mean_lats = [i.mean_latency_ms for i in result.intervals]
            if len(mean_lats) > 1:
                import statistics
                result.latency_jitter_ms = statistics.stdev(mean_lats)

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
        print(f"  L_sys (estimated): {result.jwmsp.l_sys_ms:.1f}ms")
        print(f"  RTF:               {result.jwmsp.rtf:.2f}")
        if result.jwmsp.rtf >= 1.0:
            print(f"  Status:            Real-time capable")
        else:
            print(f"  Status:            Below real-time ({1/result.jwmsp.rtf:.1f}x slower)")

    # True L_sys from timestamp injection
    if "l_sys_true" in result.metadata:
        l_sys = result.metadata["l_sys_true"]
        print(f"\nTrue L_sys (from timestamps):")
        print(f"  Mean:    {l_sys['mean_ms']:.1f}ms")
        print(f"  Min/Max: {l_sys['min_ms']:.1f}ms / {l_sys['max_ms']:.1f}ms")
        print(f"  Std:     {l_sys['std_ms']:.1f}ms")


def print_saturation_results(result: SaturationResult) -> None:
    """Print formatted saturation test results."""
    print("\n" + "=" * 70)
    print("SATURATION TEST RESULTS")
    print("=" * 70)

    print(f"\nThroughput vs Latency Curve:")
    print(f"{'Concurrency':>12} {'Throughput':>14} {'Mean Latency':>14} {'P99 Latency':>14}")
    print("-" * 56)
    for p in result.points:
        marker = " <-- saturation" if p.concurrency == result.saturation_concurrency else ""
        print(f"{p.concurrency:>12} {p.throughput_rps:>12.2f}/s {p.mean_latency_ms:>12.1f}ms {p.p99_latency_ms:>12.1f}ms{marker}")

    print(f"\nSummary:")
    print(f"  Peak throughput:       {result.peak_throughput_rps:.2f} req/sec")
    if result.saturation_concurrency:
        print(f"  Saturation point:      concurrency={result.saturation_concurrency}")
        print(f"  Recommendation:        Use concurrency < {result.saturation_concurrency} for stable latency")
    else:
        print(f"  Saturation point:      Not reached (system has headroom)")


def print_soak_results(result: SoakResult) -> None:
    """Print formatted soak test results."""
    print("\n" + "=" * 70)
    print("SOAK TEST RESULTS")
    print("=" * 70)

    duration = (
        (result.end_time - result.start_time).total_seconds()
        if result.start_time and result.end_time
        else 0
    )

    print(f"\nSummary:")
    print(f"  Duration:         {duration:.1f}s")
    print(f"  Total requests:   {result.total_requests}")
    print(f"  Total errors:     {result.total_errors}")
    print(f"  Overall rate:     {result.total_requests / duration:.2f} req/sec" if duration > 0 else "N/A")

    print(f"\nStability Metrics:")
    print(f"  RTF drift:        {result.rtf_drift:+.4f} (start to end)")
    print(f"  Latency jitter:   {result.latency_jitter_ms:.2f}ms (std of interval means)")

    if result.intervals:
        rtfs = [i.rtf for i in result.intervals if i.rtf]
        if rtfs:
            min_rtf = min(rtfs)
            max_rtf = max(rtfs)
            print(f"  RTF range:        {min_rtf:.3f} - {max_rtf:.3f}")

    if abs(result.rtf_drift) < 0.05 and result.latency_jitter_ms < 50:
        print(f"\n  Status: STABLE - System maintains consistent performance")
    elif abs(result.rtf_drift) > 0.1:
        print(f"\n  Status: DRIFT DETECTED - Performance degraded over time")
    else:
        print(f"\n  Status: ACCEPTABLE - Minor variations within tolerance")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="External benchmark for V-JEPA2 inference API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic benchmark (fixed mode)
  python -m benchmark.benchmark --target http://localhost:8000 --video samples/sample.mp4

  # With Jaeger trace collection
  python -m benchmark.benchmark \\
      --target http://localhost:8000 \\
      --video samples/sample.mp4 \\
      --jaeger http://localhost:16686

  # Saturation test: find throughput ceiling
  python -m benchmark.benchmark \\
      --target http://localhost:8000 \\
      --video samples/sample.mp4 \\
      --mode saturation \\
      --concurrency-levels 1,2,4,8,16

  # Soak test: sustained load for drift detection
  python -m benchmark.benchmark \\
      --target http://localhost:8000 \\
      --video samples/sample.mp4 \\
      --mode soak \\
      --duration 300 \\
      --report-interval 30
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
    # Mode selection
    parser.add_argument(
        "--mode",
        choices=["fixed", "saturation", "soak"],
        default="fixed",
        help="Benchmark mode: fixed (default), saturation, or soak",
    )
    # Saturation test options
    parser.add_argument(
        "--concurrency-levels",
        default="1,2,4,8,16",
        help="Comma-separated concurrency levels for saturation test (default: 1,2,4,8,16)",
    )
    parser.add_argument(
        "--requests-per-level",
        type=int,
        default=20,
        help="Requests per concurrency level in saturation test (default: 20)",
    )
    # Soak test options
    parser.add_argument(
        "--duration",
        type=float,
        default=300.0,
        help="Duration in seconds for soak test (default: 300)",
    )
    parser.add_argument(
        "--report-interval",
        type=float,
        default=30.0,
        help="Reporting interval in seconds for soak test (default: 30)",
    )
    # Timestamp injection for true L_sys
    parser.add_argument(
        "--inject-timestamp",
        action="store_true",
        help="Inject observation timestamps for true L_sys measurement (requires Jaeger)",
    )
    parser.add_argument(
        "--insecure", "-k",
        action="store_true",
        help="Skip SSL certificate verification (for self-signed certs)",
    )

    args = parser.parse_args()

    # Parse concurrency levels
    concurrency_levels = [int(x.strip()) for x in args.concurrency_levels.split(",")]

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
        concurrency_levels=concurrency_levels,
        requests_per_level=args.requests_per_level,
        duration_seconds=args.duration,
        report_interval=args.report_interval,
        inject_timestamp=args.inject_timestamp,
        insecure=args.insecure,
    )

    try:
        if args.mode == "saturation":
            result = asyncio.run(run_saturation_test(config))
            print_saturation_results(result)
        elif args.mode == "soak":
            result = asyncio.run(run_soak_test(config))
            print_soak_results(result)
        else:
            result = asyncio.run(run_benchmark(config))
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
