#!/usr/bin/env python3
"""Minimal reproducer: cudaMemcpyAsync bimodality on unified memory.

Tests whether GPU↔CPU memory copies show bimodal latency on unified memory
architectures (GB10), which is the root cause of vLLM-Omni's encode
bimodality on DGX Spark.

Simulates the vLLM model runner's data flow:
  1. Stage input to GPU (copy_to_gpu: CPU→GPU)
  2. Run compute (simulated with a GEMM)
  3. Copy output back to CPU (.to("cpu"))

Tests with varying tensor sizes and memory pressure to find the threshold
where bimodality appears.

Usage:
    python3 reproduce_memcpy.py
    python3 reproduce_memcpy.py --sizes 1,10,100,500 --pressure-gb 20
"""

import argparse
import json
import statistics
import time

import torch


def report(label, times):
    n = len(times)
    mean = statistics.mean(times)
    std = statistics.stdev(times) if n > 1 else 0
    s = sorted(times)
    p50 = s[n // 2]
    p95 = s[min(int(n * 0.95), n - 1)]
    bimodal = n > 1 and std > 0.3 * mean

    print(f"\n  {label}")
    print(f"  n={n}  Mean: {mean:.3f}ms  Std: {std:.3f}ms  P50: {p50:.3f}ms  P95: {p95:.3f}ms")
    print(f"  Min: {min(times):.3f}ms  Max: {max(times):.3f}ms")
    status = "*** BIMODAL ***" if bimodal else "STABLE"
    print(f"  {status} (std/mean={std/mean:.0%})")

    if n >= 5:
        lo, hi = min(times), max(times)
        if hi - lo > 0.001:
            nb = min(15, max(5, n // 2))
            bw = (hi - lo) / nb
            bins = [0] * nb
            for t in times:
                bins[min(int((t - lo) / bw), nb - 1)] += 1
            mx = max(bins)
            for i, c in enumerate(bins):
                bar = "#" * int(c / mx * 30) if mx else ""
                print(f"    {lo + i*bw:8.3f}ms |{bar:<30s} {c}")

    return {"label": label, "times_ms": [round(t, 4) for t in times],
            "mean_ms": round(mean, 4), "std_ms": round(std, 4),
            "p50_ms": round(p50, 4), "p95_ms": round(p95, 4),
            "bimodal": bimodal}


def bench_copy(label, src, dst_device, warmup, rounds):
    """Benchmark copying a tensor to dst_device."""
    print(f"\n--- {label} ---")
    print(f"  Tensor: {src.shape}, {src.dtype}, {src.device} → {dst_device}")
    print(f"  Size: {src.nelement() * src.element_size() / 1024**2:.1f} MB")

    for i in range(warmup):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = src.to(dst_device, non_blocking=False)
        torch.cuda.synchronize()
        t = (time.perf_counter() - t0) * 1000
        print(f"  warmup {i}: {t:.3f}ms")

    times = []
    for i in range(rounds):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = src.to(dst_device, non_blocking=False)
        torch.cuda.synchronize()
        t = (time.perf_counter() - t0) * 1000
        times.append(t)

    return report(label, times)


def bench_roundtrip(label, size_mb, warmup, rounds, compute=False):
    """Benchmark CPU→GPU copy, optional compute, GPU→CPU copy."""
    print(f"\n--- {label} ---")
    nelems = int(size_mb * 1024 * 1024 / 2)  # fp16
    cpu_tensor = torch.randn(nelems, dtype=torch.float16)

    # Optional compute matrix (simulates model weights)
    if compute:
        side = int(nelems ** 0.5)
        weight = torch.randn(side, side, dtype=torch.float16, device="cuda")

    for i in range(warmup):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        gpu = cpu_tensor.to("cuda", non_blocking=False)
        if compute:
            x = gpu[:side*side].reshape(side, side)
            _ = x @ weight
        result = gpu.to("cpu", non_blocking=False)
        torch.cuda.synchronize()
        t = (time.perf_counter() - t0) * 1000
        print(f"  warmup {i}: {t:.3f}ms")

    times = []
    for i in range(rounds):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        gpu = cpu_tensor.to("cuda", non_blocking=False)
        if compute:
            x = gpu[:side*side].reshape(side, side)
            _ = x @ weight
        result = gpu.to("cpu", non_blocking=False)
        torch.cuda.synchronize()
        t = (time.perf_counter() - t0) * 1000
        times.append(t)

    return report(f"{label} ({size_mb}MB)", times)


def bench_nonblocking_roundtrip(label, size_mb, warmup, rounds):
    """Benchmark with non_blocking=True (like vLLM's actual code)."""
    print(f"\n--- {label} ---")
    nelems = int(size_mb * 1024 * 1024 / 2)
    cpu_tensor = torch.randn(nelems, dtype=torch.float16)

    for i in range(warmup):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        gpu = cpu_tensor.to("cuda", non_blocking=True)
        result = gpu.to("cpu", non_blocking=True)
        torch.cuda.synchronize()
        t = (time.perf_counter() - t0) * 1000
        print(f"  warmup {i}: {t:.3f}ms")

    times = []
    for i in range(rounds):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        gpu = cpu_tensor.to("cuda", non_blocking=True)
        result = gpu.to("cpu", non_blocking=True)
        torch.cuda.synchronize()
        t = (time.perf_counter() - t0) * 1000
        times.append(t)

    return report(f"{label} ({size_mb}MB, non_blocking)", times)


def main():
    parser = argparse.ArgumentParser(description="cudaMemcpyAsync bimodality reproducer")
    parser.add_argument("--sizes", default="1,6,25,100,500",
                        help="Comma-separated tensor sizes in MB to test")
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--pressure-gb", type=float, default=0,
                        help="Allocate N GB on GPU before tests (simulates model weights)")
    parser.add_argument("--output", "-o", type=str, default=None)
    args = parser.parse_args()

    sizes = [float(s) for s in args.sizes.split(",")]

    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    free, total = torch.cuda.mem_get_info()
    print(f"GPU memory: {free/1024**3:.1f} / {total/1024**3:.1f} GB free")

    results = {}

    # Optional memory pressure (simulates model weights on GPU)
    pressure_buf = None
    if args.pressure_gb > 0:
        print(f"\nAllocating {args.pressure_gb}GB pressure buffer...")
        pressure_buf = torch.empty(
            int(args.pressure_gb * 1024**3 / 2), dtype=torch.float16, device="cuda"
        )
        # Touch the buffer to ensure pages are allocated
        pressure_buf.fill_(0.0)
        torch.cuda.synchronize()
        free, total = torch.cuda.mem_get_info()
        print(f"GPU memory after pressure: {free/1024**3:.1f} / {total/1024**3:.1f} GB free")

    # Test 1: CPU→GPU copy at various sizes
    print("\n" + "=" * 60)
    print("  TEST 1: CPU → GPU copy (blocking)")
    print("=" * 60)
    for size_mb in sizes:
        nelems = int(size_mb * 1024 * 1024 / 2)
        src = torch.randn(nelems, dtype=torch.float16)
        results[f"cpu_to_gpu_{size_mb}MB"] = bench_copy(
            f"CPU→GPU {size_mb}MB", src, "cuda", args.warmup, args.rounds)
        del src

    # Test 2: GPU→CPU copy at various sizes
    print("\n" + "=" * 60)
    print("  TEST 2: GPU → CPU copy (blocking)")
    print("=" * 60)
    for size_mb in sizes:
        nelems = int(size_mb * 1024 * 1024 / 2)
        src = torch.randn(nelems, dtype=torch.float16, device="cuda")
        results[f"gpu_to_cpu_{size_mb}MB"] = bench_copy(
            f"GPU→CPU {size_mb}MB", src, "cpu", args.warmup, args.rounds)
        del src
        torch.cuda.synchronize()

    # Test 3: Roundtrip (CPU→GPU→CPU) — like vLLM model runner
    print("\n" + "=" * 60)
    print("  TEST 3: Roundtrip CPU→GPU→CPU (blocking)")
    print("=" * 60)
    for size_mb in sizes:
        results[f"roundtrip_{size_mb}MB"] = bench_roundtrip(
            "Roundtrip", size_mb, args.warmup, args.rounds)

    # Test 4: Roundtrip with non_blocking=True (like vLLM actually does)
    print("\n" + "=" * 60)
    print("  TEST 4: Roundtrip CPU→GPU→CPU (non_blocking)")
    print("=" * 60)
    for size_mb in sizes:
        results[f"roundtrip_nb_{size_mb}MB"] = bench_nonblocking_roundtrip(
            "Roundtrip non_blocking", size_mb, args.warmup, args.rounds)

    # Test 5: Roundtrip with compute in between
    print("\n" + "=" * 60)
    print("  TEST 5: Roundtrip with GEMM (simulates model forward)")
    print("=" * 60)
    for size_mb in [s for s in sizes if s >= 6]:
        results[f"roundtrip_compute_{size_mb}MB"] = bench_roundtrip(
            "Roundtrip+GEMM", size_mb, args.warmup, args.rounds, compute=True)

    if pressure_buf is not None:
        del pressure_buf
        torch.cuda.empty_cache()

    if args.output:
        with open(args.output, "w") as f:
            json.dump({
                "device": torch.cuda.get_device_name(0),
                "pytorch": torch.__version__,
                "pressure_gb": args.pressure_gb,
                "experiments": results,
            }, f, indent=2)
        print(f"\nResults saved to {args.output}")

    print("\nDone.")


if __name__ == "__main__":
    main()
