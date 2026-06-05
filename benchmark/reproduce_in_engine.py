#!/usr/bin/env python3
"""Reproduce bimodality by running VJEPA2 through vLLM-Omni's engine subprocess.

This script replicates the actual vLLM-Omni execution context:
- Spawns a subprocess (like StageEngineCoreProc)
- Initializes NCCL distributed backend (like vLLM does)
- Allocates GPU memory pool (like gpu_memory_utilization)
- Then loads and runs the VJEPA2 model

Each experiment adds one more vLLM component to isolate the trigger.

Run inside the vllm-omni container:
    python3 /tmp/reproduce_in_engine.py --rounds 20
"""

import argparse
import json
import multiprocessing
import os
import statistics
import time

import numpy as np
import torch


def report(label, times):
    n = len(times)
    mean = statistics.mean(times)
    std = statistics.stdev(times) if n > 1 else 0
    s = sorted(times)
    p50 = s[n // 2]
    p95 = s[min(int(n * 0.95), n - 1)]
    bimodal = n > 1 and std > 0.3 * mean
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  n={n}  Mean: {mean:.1f}ms  Std: {std:.1f}ms  P50: {p50:.1f}ms  P95: {p95:.1f}ms")
    print(f"  {'*** BIMODAL ***' if bimodal else 'STABLE'} (std/mean={std/mean:.0%})")
    if n >= 5:
        lo, hi = min(times), max(times)
        if hi - lo > 1:
            nb = min(15, max(5, n // 2))
            bw = (hi - lo) / nb
            bins = [0] * nb
            for t in times:
                bins[min(int((t - lo) / bw), nb - 1)] += 1
            mx = max(bins)
            for i, c in enumerate(bins):
                bar = "#" * int(c / mx * 30) if mx else ""
                print(f"  {lo + i*bw:7.1f}ms |{bar:<30s} {c}")
    return {"label": label, "times_ms": [round(t, 2) for t in times],
            "mean_ms": round(mean, 2), "std_ms": round(std, 2),
            "p50_ms": round(p50, 2), "p95_ms": round(p95, 2), "bimodal": bimodal}


def load_and_bench(model_id, warmup, rounds):
    """Load VJEPA2 and run inference, return times."""
    from transformers import AutoModelForVideoClassification, AutoVideoProcessor
    processor = AutoVideoProcessor.from_pretrained(model_id)
    model = AutoModelForVideoClassification.from_pretrained(model_id)
    model.to("cuda").eval()
    torch.cuda.synchronize()

    frames = [np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8) for _ in range(16)]
    inputs = processor(frames, return_tensors="pt")
    key = "pixel_values_videos" if "pixel_values_videos" in inputs else "pixel_values"
    pv = inputs[key].to("cuda")

    def timed():
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(pv)
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1000

    for i in range(warmup):
        print(f"    warmup {i}: {timed():.1f}ms", flush=True)

    times = []
    for i in range(rounds):
        t = timed()
        times.append(t)
        print(f"    [{i:2d}] {t:.1f}ms", flush=True)
    return times


def experiment_subprocess_plain(model_id, warmup, rounds, q):
    """A: Plain subprocess — no NCCL, no memory pool."""
    print("  [A] Plain subprocess", flush=True)
    times = load_and_bench(model_id, warmup, rounds)
    q.put(("A", times))


def experiment_subprocess_nccl(model_id, warmup, rounds, q):
    """B: Subprocess + NCCL init (like vLLM's distributed init)."""
    print("  [B] Subprocess + NCCL init", flush=True)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29501"
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    torch.distributed.init_process_group(backend="nccl", world_size=1, rank=0)
    print("    NCCL initialized", flush=True)

    times = load_and_bench(model_id, warmup, rounds)
    torch.distributed.destroy_process_group()
    q.put(("B", times))


def experiment_subprocess_nccl_mempool(model_id, warmup, rounds, gpu_mem_frac, q):
    """C: Subprocess + NCCL + GPU memory pool (like vLLM's full init)."""
    print(f"  [C] Subprocess + NCCL + {gpu_mem_frac*100:.0f}% GPU memory pool", flush=True)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29502"
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    torch.distributed.init_process_group(backend="nccl", world_size=1, rank=0)

    # Allocate GPU memory pool like vLLM does
    free, total = torch.cuda.mem_get_info()
    target = int(total * gpu_mem_frac)
    print(f"    Allocating {target/1024**3:.1f}GB GPU memory pool...", flush=True)
    pool = torch.empty(target // 2, dtype=torch.float16, device="cuda")
    torch.cuda.synchronize()
    # Keep pool alive (like vLLM's KV cache)

    times = load_and_bench(model_id, warmup, rounds)

    del pool
    torch.distributed.destroy_process_group()
    q.put(("C", times))


def experiment_subprocess_nccl_mempool_free(model_id, warmup, rounds, gpu_mem_frac, q):
    """D: Subprocess + NCCL + GPU memory alloc+free (pool freed before inference)."""
    print(f"  [D] Subprocess + NCCL + {gpu_mem_frac*100:.0f}% GPU alloc+free", flush=True)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29503"
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    torch.distributed.init_process_group(backend="nccl", world_size=1, rank=0)

    free, total = torch.cuda.mem_get_info()
    target = int(total * gpu_mem_frac)
    print(f"    Allocating then freeing {target/1024**3:.1f}GB...", flush=True)
    pool = torch.empty(target // 2, dtype=torch.float16, device="cuda")
    torch.cuda.synchronize()
    del pool
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    times = load_and_bench(model_id, warmup, rounds)
    torch.distributed.destroy_process_group()
    q.put(("D", times))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="facebook/vjepa2-vitl-fpc16-256-ssv2")
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--gpu-mem-frac", type=float, default=0.2)
    parser.add_argument("--output", "-o", type=str, default=None)
    args = parser.parse_args()

    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    free, total = torch.cuda.mem_get_info()
    print(f"GPU memory: {free/1024**3:.1f} / {total/1024**3:.1f} GB free")

    ctx = multiprocessing.get_context("spawn")
    results = {}

    experiments = [
        ("A", experiment_subprocess_plain, (args.model, args.warmup, args.rounds)),
        ("B", experiment_subprocess_nccl, (args.model, args.warmup, args.rounds)),
        ("C", experiment_subprocess_nccl_mempool, (args.model, args.warmup, args.rounds, args.gpu_mem_frac)),
        ("D", experiment_subprocess_nccl_mempool_free, (args.model, args.warmup, args.rounds, args.gpu_mem_frac)),
    ]

    for label, fn, fn_args in experiments:
        print(f"\n--- Experiment {label} ---", flush=True)
        q = ctx.Queue()
        p = ctx.Process(target=fn, args=(*fn_args, q))
        p.start()
        p.join(timeout=600)
        if not q.empty():
            name, times = q.get()
            results[name] = report(f"Exp {name}: {fn.__doc__.strip()}", times)
        else:
            print(f"  FAILED (subprocess exit code: {p.exitcode})")

    if args.output:
        with open(args.output, "w") as f:
            json.dump({"device": torch.cuda.get_device_name(0), "experiments": results}, f, indent=2)
        print(f"\nResults saved to {args.output}")

    print("\nDone.")


if __name__ == "__main__":
    main()
