#!/usr/bin/env python3
"""Minimal reproducer for GB10 encode bimodality.

Compares inference latency stability under different conditions to isolate
why vLLM shows bimodal encode latency on unified memory (GB10) while
plain PyTorch does not.

Experiments:
  A: Baseline — load model via HF, run inference directly (vjepa2-demo style)
  B: Pre-allocate + free GPU memory before inference (simulates vLLM's KV cache)
  C: Thread pool execution (simulates asyncio.to_thread)
  D: vLLM engine — load model through vLLM with --load-format dummy

Usage (on DGX Spark):
    # Baseline only
    python3 reproduce_bimodality.py

    # All experiments
    python3 reproduce_bimodality.py --prealloc-gb 20 --use-threads --use-vllm

    # Quick test
    python3 reproduce_bimodality.py --rounds 10 --warmup 2

    # With profiling
    python3 reproduce_bimodality.py --profile
"""

import argparse
import statistics
import time

import numpy as np
import torch


def load_model(model_id: str = "facebook/vjepa2-vitl-fpc16-256-ssv2"):
    """Load model the vjepa2-demo way: from_pretrained + .to(device)."""
    from transformers import AutoModelForVideoClassification, AutoVideoProcessor

    processor = AutoVideoProcessor.from_pretrained(model_id)
    model = AutoModelForVideoClassification.from_pretrained(model_id)
    model.to("cuda")
    model.eval()
    torch.cuda.synchronize()
    return model, processor


def make_dummy_input(processor, num_frames=16, resolution=256):
    """Create a dummy video clip."""
    frames = [np.random.randint(0, 255, (resolution, resolution, 3), dtype=np.uint8)
              for _ in range(num_frames)]
    inputs = processor(frames, return_tensors="pt")
    key = "pixel_values_videos" if "pixel_values_videos" in inputs else "pixel_values"
    return inputs[key].to("cuda")


def run_inference(model, pixel_values):
    """Single forward pass, returns time in ms."""
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        _ = model(pixel_values)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000


def run_inference_in_thread(model, pixel_values):
    """Run inference via thread pool (like vLLM's asyncio.to_thread)."""
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(run_inference, model, pixel_values).result()


def prealloc_gpu_memory(gb: float):
    """Allocate then free GPU memory (simulates vLLM's KV cache reservation).

    PyTorch's caching allocator retains the freed blocks, which on unified
    memory may cause the OS to reclaim model weight pages.
    """
    nbytes = int(gb * 1024**3)
    print(f"  Pre-allocating {gb:.1f} GB on GPU...")
    buf = torch.empty(nbytes // 4, dtype=torch.float32, device="cuda")
    torch.cuda.synchronize()
    del buf
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    print(f"  GPU memory after alloc+free: {free/1024**3:.1f} / {total/1024**3:.1f} GB free")


def run_experiment(label, inference_fn, model, pixel_values, warmup, rounds):
    """Run warmup + measurement rounds, print results, return summary dict."""
    print(f"\n--- {label} ---")

    for i in range(warmup):
        t = inference_fn(model, pixel_values)
        print(f"  warmup {i}: {t:.1f}ms")

    times = []
    for i in range(rounds):
        t = inference_fn(model, pixel_values)
        times.append(t)
        print(f"  [{i:2d}] {t:.1f}ms")

    return report(label, times)


def report(label, times):
    """Print latency statistics, detect bimodality, and return summary dict."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    n = len(times)
    mean = statistics.mean(times)
    std = statistics.stdev(times) if n > 1 else 0
    sorted_t = sorted(times)
    p50 = sorted_t[n // 2]
    p95 = sorted_t[min(int(n * 0.95), n - 1)]
    bimodal = n > 1 and std > 0.3 * mean

    print(f"  Rounds:  {n}")
    print(f"  Mean:    {mean:.1f}ms")
    print(f"  Std:     {std:.1f}ms")
    print(f"  Min:     {min(times):.1f}ms")
    print(f"  Max:     {max(times):.1f}ms")
    print(f"  P50:     {p50:.1f}ms")
    print(f"  P95:     {p95:.1f}ms")

    if bimodal:
        print(f"  *** BIMODAL (std/mean = {std/mean:.0%}) ***")
    elif n > 1:
        print(f"  STABLE (std/mean = {std/mean:.0%})")

    # ASCII histogram (10 bins)
    if n >= 5:
        lo, hi = min(times), max(times)
        if hi - lo > 0.1:
            num_bins = min(20, n // 2)
            bin_width = (hi - lo) / num_bins
            bins = [0] * num_bins
            for t in times:
                idx = min(int((t - lo) / bin_width), num_bins - 1)
                bins[idx] += 1
            max_count = max(bins)
            print(f"\n  Histogram ({lo:.0f}ms - {hi:.0f}ms, {num_bins} bins):")
            for i, count in enumerate(bins):
                edge = lo + i * bin_width
                bar = "#" * int(count / max_count * 40) if max_count > 0 else ""
                print(f"  {edge:7.1f}ms |{bar:<40s} {count}")

    return {
        "label": label,
        "times_ms": [round(t, 2) for t in times],
        "mean_ms": round(mean, 2),
        "std_ms": round(std, 2),
        "min_ms": round(min(times), 2),
        "max_ms": round(max(times), 2),
        "p50_ms": round(p50, 2),
        "p95_ms": round(p95, 2),
        "bimodal": bimodal,
    }


def run_vllm_experiment(model_id, rounds, warmup):
    """Test whether vLLM's memory management triggers bimodality.

    Simulates vLLM's GPU memory lifecycle:
    1. Allocate a large pool (like gpu_memory_utilization)
    2. Run many small alloc/free cycles (like profiling warmup)
    3. Call empty_cache() (like vLLM does between phases)
    4. Then run VJEPA2 inference and check for bimodality

    Also tests whether running vLLM's actual engine (with OPT-125M dummy
    weights) leaves the GPU in a state that causes bimodality.
    """
    print("\n--- D: Simulated vLLM memory lifecycle ---")

    # D1: Simulate vLLM's profiling warmup — many alloc/free cycles
    print("  D1: Simulating vLLM profiling (many alloc/free cycles)...")
    for _ in range(50):
        tensors = [torch.empty(1024 * 1024, dtype=torch.float16, device="cuda")
                   for _ in range(20)]
        del tensors
    torch.cuda.synchronize()

    # D2: Allocate large pool then free (like KV cache reservation)
    print("  D2: Simulating KV cache reservation (40GB alloc+free)...")
    big = torch.empty(40 * 1024**3 // 2, dtype=torch.float16, device="cuda")
    torch.cuda.synchronize()
    del big

    # D3: empty_cache — this is what vLLM does between profiling and serving
    print("  D3: Calling torch.cuda.empty_cache()...")
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    free, total = torch.cuda.mem_get_info()
    print(f"  GPU memory after lifecycle: {free/1024**3:.1f} / {total/1024**3:.1f} GB free")

    # Now reload VJEPA2 and test
    print(f"\n  Loading VJEPA2 after memory lifecycle...")
    model, processor = load_model(model_id)
    pixel_values = make_dummy_input(processor)

    for i in range(warmup):
        t = run_inference(model, pixel_values)
        print(f"  warmup {i}: {t:.1f}ms")

    times_d = []
    for i in range(rounds):
        t = run_inference(model, pixel_values)
        times_d.append(t)
        print(f"  [{i:2d}] {t:.1f}ms")

    result_d = report("D: After vLLM memory lifecycle", times_d)

    # E: Actually run vLLM engine if available
    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        print("\n  vLLM not installed, skipping experiment E")
        return times_d

    print("\n--- E: After actual vLLM engine run ---")
    del model, processor, pixel_values
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    print("  Loading OPT-125M through vLLM (dummy weights)...")
    try:
        llm = LLM(
            model="facebook/opt-125m",
            load_format="dummy",
            enforce_eager=True,
            gpu_memory_utilization=0.5,
        )
        sampling_params = SamplingParams(max_tokens=16, temperature=0)
        llm.generate([{"prompt_token_ids": [1, 2, 3, 4]}] * 5, sampling_params)
        print("  vLLM generate() completed")

        del llm
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    except Exception as e:
        print(f"  vLLM failed: {e}")
        return times_d

    free, total = torch.cuda.mem_get_info()
    print(f"  GPU memory after vLLM teardown: {free/1024**3:.1f} / {total/1024**3:.1f} GB free")

    print(f"\n  Reloading VJEPA2 after vLLM run...")
    model, processor = load_model(model_id)
    pixel_values = make_dummy_input(processor)

    for i in range(warmup):
        t = run_inference(model, pixel_values)
        print(f"  warmup {i}: {t:.1f}ms")

    times_e = []
    for i in range(rounds):
        t = run_inference(model, pixel_values)
        times_e.append(t)
        print(f"  [{i:2d}] {t:.1f}ms")

    report("E: After actual vLLM engine run", times_e)
    return times_d


def main():
    parser = argparse.ArgumentParser(description="GB10 bimodality reproducer")
    parser.add_argument("--model", default="facebook/vjepa2-vitl-fpc16-256-ssv2")
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--prealloc-gb", type=float, default=0,
                        help="Pre-allocate then free N GB on GPU before inference")
    parser.add_argument("--use-threads", action="store_true",
                        help="Also test thread pool execution")
    parser.add_argument("--use-vllm", action="store_true",
                        help="Also test vLLM engine path (--load-format dummy)")
    parser.add_argument("--profile", action="store_true",
                        help="Enable torch.profiler for one round")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Save results as JSON (includes raw times for histograms)")
    args = parser.parse_args()

    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    free, total = torch.cuda.mem_get_info()
    print(f"GPU memory: {free/1024**3:.1f} / {total/1024**3:.1f} GB free")

    # --- Load model ---
    print("\nLoading model (HuggingFace from_pretrained)...")
    model, processor = load_model(args.model)
    pixel_values = make_dummy_input(processor)
    print(f"Input shape: {pixel_values.shape}")

    free, total = torch.cuda.mem_get_info()
    print(f"GPU memory after load: {free/1024**3:.1f} / {total/1024**3:.1f} GB free")

    results = {}

    # --- A: Baseline ---
    results["A"] = run_experiment(
        "A: Baseline (direct call, no pre-alloc)",
        run_inference, model, pixel_values, args.warmup, args.rounds,
    )

    # --- B: Pre-allocate ---
    if args.prealloc_gb > 0:
        prealloc_gpu_memory(args.prealloc_gb)
        results["B"] = run_experiment(
            f"B: After {args.prealloc_gb}GB pre-alloc+free",
            run_inference, model, pixel_values, args.warmup, args.rounds,
        )

    # --- C: Thread pool ---
    if args.use_threads:
        results["C"] = run_experiment(
            "C: Thread pool execution",
            run_inference_in_thread, model, pixel_values, args.warmup, args.rounds,
        )

    # --- Cleanup before vLLM test ---
    if args.use_vllm:
        del model, processor, pixel_values
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        run_vllm_experiment(args.model, args.rounds, args.warmup)
        # Results are printed inline by run_vllm_experiment

    # --- Optional profiling ---
    if args.profile and not args.use_vllm:
        print("\n--- Profiling one inference round ---")
        from torch.profiler import profile, ProfilerActivity
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            with_stack=True,
        ) as prof:
            run_inference(model, pixel_values)
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
        prof.export_chrome_trace("/tmp/vjepa_trace.json")
        print("Trace saved to /tmp/vjepa_trace.json")

    # --- Save results ---
    if args.output:
        import json
        output = {
            "device": torch.cuda.get_device_name(0),
            "pytorch": torch.__version__,
            "model": args.model,
            "config": {
                "rounds": args.rounds,
                "warmup": args.warmup,
                "prealloc_gb": args.prealloc_gb,
                "use_threads": args.use_threads,
                "use_vllm": args.use_vllm,
            },
            "experiments": results,
        }
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to {args.output}")

    print("\n\nDone.")


if __name__ == "__main__":
    main()
