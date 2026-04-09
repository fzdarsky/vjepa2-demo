"""
Benchmark Metrics Aggregation

Shared statistics and JWMSP methodology calculations for benchmark results.
Used by both external (benchmark.py) and whitebox (benchmark_whitebox.py) benchmarks.
"""

import statistics
from dataclasses import dataclass, field
from typing import Any


@dataclass
class StageMetrics:
    """Statistics for a single pipeline stage."""

    name: str
    samples: list[float] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.samples)

    @property
    def mean_ms(self) -> float:
        return statistics.mean(self.samples) if self.samples else 0.0

    @property
    def std_ms(self) -> float:
        return statistics.stdev(self.samples) if len(self.samples) > 1 else 0.0

    @property
    def min_ms(self) -> float:
        return min(self.samples) if self.samples else 0.0

    @property
    def max_ms(self) -> float:
        return max(self.samples) if self.samples else 0.0

    def percentile_ms(self, p: float) -> float:
        """Calculate percentile (0-100) in milliseconds."""
        if not self.samples:
            return 0.0
        sorted_samples = sorted(self.samples)
        idx = int(len(sorted_samples) * p / 100)
        idx = min(idx, len(sorted_samples) - 1)
        return sorted_samples[idx]

    @property
    def p50_ms(self) -> float:
        return self.percentile_ms(50)

    @property
    def p95_ms(self) -> float:
        return self.percentile_ms(95)

    @property
    def p99_ms(self) -> float:
        return self.percentile_ms(99)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "count": self.count,
            "mean_ms": round(self.mean_ms, 3),
            "std_ms": round(self.std_ms, 3),
            "min_ms": round(self.min_ms, 3),
            "max_ms": round(self.max_ms, 3),
            "p50_ms": round(self.p50_ms, 3),
            "p95_ms": round(self.p95_ms, 3),
            "p99_ms": round(self.p99_ms, 3),
        }


# Standard JWMSP span names for the V-JEPA2 pipeline
JWMSP_SPANS = [
    "input_receive",
    "input_open",
    "input_decode",
    "input_preprocess",
    "jepa_encode",
    "jepa_predict",
    "jepa_pool",
    "output_postprocess",
]


@dataclass
class JWSMPMetrics:
    """JEPA World Model Streaming Performance metrics.

    Implements the methodology defined in benchmark/METHODOLOGY.md.
    """

    # Computational latency: wall-clock time to process one inference
    l_comp_ms: float = 0.0

    # Algorithmic latency: video window duration required by model
    # Default: 16 frames @ 30fps = 533ms
    l_algo_ms: float = 533.0

    # Total system lag: L_sys = L_algo + L_comp
    @property
    def l_sys_ms(self) -> float:
        return self.l_algo_ms + self.l_comp_ms

    # Real-Time Factor: RTF = T_video / T_process
    # RTF >= 1.0 means keeping up with real-time
    @property
    def rtf(self) -> float:
        if self.l_comp_ms <= 0:
            return float("inf")
        return self.l_algo_ms / self.l_comp_ms

    # Jitter: variance in L_comp over sustained stream
    jitter_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "L_comp_ms": round(self.l_comp_ms, 3),
            "L_algo_ms": round(self.l_algo_ms, 3),
            "L_sys_ms": round(self.l_sys_ms, 3),
            "rtf": round(self.rtf, 3),
            "jitter_ms": round(self.jitter_ms, 3),
        }


def compute_jwmsp_metrics(
    stage_metrics: dict[str, StageMetrics],
    num_frames: int = 16,
    source_fps: float = 30.0,
) -> JWSMPMetrics:
    """Compute JWMSP metrics from stage timing data.

    Args:
        stage_metrics: Dict of span name to StageMetrics
        num_frames: Number of frames per clip
        source_fps: Source video frame rate

    Returns:
        JWSMPMetrics with computed values
    """
    # L_comp = sum of all stage means
    l_comp = sum(m.mean_ms for m in stage_metrics.values())

    # L_algo = clip duration at source FPS
    l_algo = (num_frames / source_fps) * 1000  # ms

    # Jitter = std of total per-sample latencies
    # We need per-trace totals, but if we only have per-stage samples,
    # approximate using the stage with most variance
    jitter = max(
        (m.std_ms for m in stage_metrics.values()),
        default=0.0,
    )

    return JWSMPMetrics(
        l_comp_ms=l_comp,
        l_algo_ms=l_algo,
        jitter_ms=jitter,
    )


def aggregate_span_durations(
    durations: dict[str, list[float]],
) -> dict[str, StageMetrics]:
    """Convert raw duration lists to StageMetrics objects.

    Args:
        durations: Dict mapping span name to list of durations (ms)

    Returns:
        Dict mapping span name to StageMetrics
    """
    return {
        name: StageMetrics(name=name, samples=samples)
        for name, samples in durations.items()
        if samples  # Skip empty lists
    }


def format_latency_table(
    stage_metrics: dict[str, StageMetrics],
    span_order: list[str] | None = None,
) -> str:
    """Format stage metrics as a text table.

    Args:
        stage_metrics: Dict of span name to StageMetrics
        span_order: Optional ordering of spans (default: JWMSP_SPANS)

    Returns:
        Formatted table string
    """
    if span_order is None:
        span_order = JWMSP_SPANS

    lines = []
    lines.append(f"{'Stage':<20} {'Mean':>8} {'Std':>8} {'P50':>8} {'P95':>8} {'P99':>8}")
    lines.append(f"{'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    total_mean = 0.0
    for name in span_order:
        if name not in stage_metrics:
            continue
        m = stage_metrics[name]
        total_mean += m.mean_ms
        lines.append(
            f"{name:<20} "
            f"{m.mean_ms:>7.2f}ms "
            f"{m.std_ms:>7.2f}ms "
            f"{m.p50_ms:>7.2f}ms "
            f"{m.p95_ms:>7.2f}ms "
            f"{m.p99_ms:>7.2f}ms"
        )

    lines.append(f"{'-'*20} {'-'*8}")
    lines.append(f"{'TOTAL':<20} {total_mean:>7.2f}ms")

    return "\n".join(lines)


def format_latency_breakdown(
    stage_metrics: dict[str, StageMetrics],
    span_order: list[str] | None = None,
) -> str:
    """Format stage metrics as a percentage breakdown with bar chart.

    Args:
        stage_metrics: Dict of span name to StageMetrics
        span_order: Optional ordering of spans

    Returns:
        Formatted breakdown string
    """
    if span_order is None:
        span_order = JWMSP_SPANS

    total = sum(m.mean_ms for m in stage_metrics.values())
    if total <= 0:
        return "No data"

    lines = []
    for name in span_order:
        if name not in stage_metrics:
            continue
        m = stage_metrics[name]
        pct = (m.mean_ms / total) * 100
        bar_len = int(pct / 2)
        bar = "#" * bar_len
        lines.append(f"{name:<20} {bar:<50} {pct:5.1f}%")

    return "\n".join(lines)
