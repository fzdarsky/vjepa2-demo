# V-JEPA2 Benchmark Suite

Latency and throughput benchmarks for V-JEPA2 inference, aligned with the [JWMSP methodology](METHODOLOGY.md).

## Benchmarks

| Benchmark | Measures | Use when |
|-----------|----------|----------|
| `benchmark.py` | External load gen + OTel trace collection | Production benchmarking, containerized deployments |
| `benchmark.py --mode saturation` | Throughput vs latency curve | Capacity planning, find concurrency limits |
| `benchmark.py --mode soak` | RTF drift over time | Stability testing, detect memory leaks |
| `benchmark_whitebox.py` | In-process per-stage latency | Development, debugging, quick A/B tests |
| `benchmark_stream.py` | Throughput, realtime ratio, pipeline parallelism | Capacity planning |
| `benchmark_coldstart.py` | Model load time, first inference overhead | Serverless/scale-to-zero |
| `benchmark_e2e.py` | Full capture-to-prediction latency by source type | Architecture decisions |

## Quick Start

```bash
# External benchmark (requires server + Jaeger running)
python -m benchmark.benchmark \
    --target http://localhost:8000 \
    --video samples/sample.mp4 \
    --jaeger http://localhost:16686

# In-process benchmark (quick local testing)
python -m benchmark.benchmark_whitebox --video samples/sample.mp4

# Stream throughput
python -m benchmark.benchmark_stream --video samples/hands.mp4

# Cold start penalty
python -m benchmark.benchmark_coldstart
```

## Benchmark Details

### benchmark.py — External Load Generator (Default)

The primary benchmark that works with any deployment (local, Podman, Kubernetes):

```bash
# Start server with observability
podman compose --profile cpu --profile observability up

# Run benchmark
python -m benchmark.benchmark \
    --target http://localhost:8000 \
    --video samples/sample.mp4 \
    --requests 50 \
    --concurrency 4 \
    --jaeger http://localhost:16686 \
    --output results.json
```

Collects timing from OTel traces via Jaeger, measuring actual server behavior.

#### Saturation Test Mode

Find the throughput ceiling by ramping up concurrency:

```bash
python -m benchmark.benchmark \
    --target http://localhost:8000 \
    --video samples/sample.mp4 \
    --mode saturation \
    --concurrency-levels 1,2,4,8,16
```

Output:
```
Saturation test:
  Conc   Throughput       Mean        P95        P99   Errors
--------------------------------------------------------------
     1       1.52/s    657.8ms    665.2ms    668.1ms     0.0%
     2       2.89/s    691.2ms    720.3ms    735.8ms     0.0%
     4       4.12/s    970.5ms   1105.2ms   1180.3ms     0.0%
     8       4.35/s   1838.2ms   2450.1ms   2680.5ms     0.0% <-- saturation
    16       4.21/s   3795.5ms   4520.3ms   4850.2ms     2.5%

Peak throughput:       4.35 req/sec
Saturation point:      concurrency=8
Recommendation:        Use concurrency < 8 for stable latency
```

#### Soak Test Mode

Run sustained load to detect performance drift over time:

```bash
python -m benchmark.benchmark \
    --target http://localhost:8000 \
    --video samples/sample.mp4 \
    --mode soak \
    --duration 300 \
    --report-interval 30 \
    --concurrency 2
```

Tracks RTF drift and latency jitter to identify:

- Memory leaks
- Thermal throttling
- Resource exhaustion

#### True L_sys Measurement

For accurate end-to-end latency measurement, inject timestamps:

```bash
python -m benchmark.benchmark \
    --target http://localhost:8000 \
    --video samples/sample.mp4 \
    --jaeger http://localhost:16686 \
    --inject-timestamp
```

This embeds an observation timestamp in each request. The server records it as a span attribute, allowing calculation of true L_sys (prediction time - observation time) from traces.

### benchmark_whitebox.py — In-Process Testing

For development and quick A/B testing without full server setup:

```bash
python -m benchmark.benchmark_whitebox --video samples/sample.mp4 --iterations 50
```

Output:
```
Stage               Mean      P50      P95      P99
input_decode       20.1ms   20.0ms   20.5ms   20.8ms
input_preprocess    5.4ms    5.4ms    5.7ms    5.9ms
jepa_encode       180.5ms  180.0ms  182.3ms  183.1ms
jepa_predict      320.2ms  319.8ms  322.1ms  323.0ms
jepa_pool          50.3ms   50.1ms   51.2ms   51.8ms
output_postprocess  1.4ms    1.4ms    1.5ms    1.5ms
```

Options:
- `--warmup N` — Warmup iterations (default: 5)
- `--iterations N` — Timed iterations (default: 20)
- `--output FILE` — Save JSON results
- `--compare FILE` — Compare against baseline

### benchmark_stream.py — Stream Throughput

Tests sustained throughput and realtime capability:

```bash
# Sequential processing
python -m benchmark.benchmark_stream --video samples/hands.mp4

# Pipelined (decode overlaps inference)
python -m benchmark.benchmark_stream --video samples/hands.mp4 --pipeline

# Compare both modes
python -m benchmark.benchmark_stream --video samples/hands.mp4 --compare-modes
```

Key metrics:
- **Throughput** (clips/sec, effective FPS)
- **RT ratio** — <1.0 means keeping up, >1.0 means falling behind
- **Queue depth** — Tracks backpressure in pipelined mode

### benchmark_coldstart.py — Startup Latency

Measures the penalty for serverless/scale-to-zero deployments:

```bash
python -m benchmark.benchmark_coldstart --iterations 3
```

Output:
```
Model Load:           2847.3ms
First Inference:       892.1ms  ← JIT compilation
Warm Inference:        615.2ms
────────────────────────────────
Total Cold Start:     3739.4ms
```

### benchmark_e2e.py — End-to-End Latency

Measures complete capture-to-prediction path including source latency:

```bash
python -m benchmark.benchmark_e2e --video samples/sample.mp4
python -m benchmark.benchmark_e2e --rtsp rtsp://192.168.1.100:554/stream
python -m benchmark.benchmark_e2e --camera 0
```

## Span Naming Convention

All spans follow the `{subsystem}_{verb}` pattern (see [METHODOLOGY.md](METHODOLOGY.md)):

| Span | Stage | Description |
|------|-------|-------------|
| `input_receive` | Input | Network/file reception |
| `input_decode` | Input | Video codec decoding |
| `input_preprocess` | Input | Frame → tensor preparation |
| `jepa_encode` | Model | ViT encoder forward pass |
| `jepa_predict` | Model | Predictor forward pass |
| `jepa_pool` | Model | Attentive temporal pooling |
| `output_postprocess` | Output | Softmax, top-k, labels |

## A/B Testing Workflow

```bash
# 1. Establish baseline
python -m benchmark.benchmark_whitebox --video samples/sample.mp4 -o baseline.json

# 2. Make optimization (torch.compile, FP16, etc.)

# 3. Compare
python -m benchmark.benchmark_whitebox --video samples/sample.mp4 -c baseline.json
```

## Interpreting Results

### Realtime Capability

At 30fps with 16-frame clips (stride=16):
- Clip duration = 533ms
- If L_comp > 533ms → RT ratio > 1.0 → falling behind

To achieve realtime:
1. Reduce inference latency (torch.compile, FP16, quantization)
2. Increase stride (skip frames)
3. Accept queue buildup or drop frames

### Where to Optimize

Typical breakdown:
```
input_decode:        3%  ← PyAV, hard to optimize
input_preprocess:    1%  ← AutoVideoProcessor
jepa_encode:        30%  ← ViT backbone
jepa_predict:       55%  ← Predictor (main bottleneck)
jepa_pool:           8%  ← Attentive pooling
output_postprocess: <1%  ← softmax + topk
```

High-impact optimizations:
- `torch.compile()` — 10-30% speedup
- FP16/BF16 — 2x throughput on GPU
- ST-A² area attention — 4x attention reduction

## Output Format

All benchmarks support `--output FILE` to save JSON results:

```json
{
  "config": {
    "target_url": "http://localhost:8000",
    "device": "cuda"
  },
  "summary": {
    "total_mean_ms": 642.0,
    "throughput_clips_per_sec": 1.56
  },
  "stages": {
    "input_decode": { "mean_ms": 20.1, "p50_ms": 20.0, "p99_ms": 20.8 },
    "jepa_encode": { "mean_ms": 180.5, "p50_ms": 180.0, "p99_ms": 183.1 },
    "jepa_predict": { "mean_ms": 320.2, "p50_ms": 319.8, "p99_ms": 323.0 }
  },
  "methodology": {
    "L_comp_ms": 615.0,
    "L_algo_ms": 533.0,
    "rtf": 0.87
  }
}
```

Use for CI integration, tracking over time, or automated regression detection.
