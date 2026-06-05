#!/usr/bin/env python3
"""Minimal reproducer: run VJEPA2 through vLLM's actual model loading path.

This script loads VJEPA2 the way vLLM-Omni does (via VJepa2Model + VJepa2Encoder)
and runs inference to see if bimodality appears. Compares against the plain
HuggingFace from_pretrained path.

Usage (on DGX Spark with vllm-omni repo at /home/fzdarsky/vllm-jepa-research/vllm-omni):
    PYTHONPATH=/home/fzdarsky/vllm-jepa-research/vllm-omni:$PYTHONPATH \
    python3 reproduce_vllm_path.py --rounds 30 -o /tmp/vllm_path_results.json
"""

import argparse
import json
import statistics
import time

import numpy as np
import torch


def report(label, times):
    """Print stats with histogram, return summary dict."""
    n = len(times)
    mean = statistics.mean(times)
    std = statistics.stdev(times) if n > 1 else 0
    sorted_t = sorted(times)
    p50 = sorted_t[n // 2]
    p95 = sorted_t[min(int(n * 0.95), n - 1)]
    bimodal = n > 1 and std > 0.3 * mean

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Rounds: {n}  Mean: {mean:.1f}ms  Std: {std:.1f}ms")
    print(f"  P50: {p50:.1f}ms  P95: {p95:.1f}ms  Min: {min(times):.1f}ms  Max: {max(times):.1f}ms")
    print(f"  {'*** BIMODAL ***' if bimodal else 'STABLE'} (std/mean = {std/mean:.0%})")

    if n >= 5:
        lo, hi = min(times), max(times)
        if hi - lo > 0.1:
            num_bins = min(20, max(5, n // 2))
            bw = (hi - lo) / num_bins
            bins = [0] * num_bins
            for t in times:
                bins[min(int((t - lo) / bw), num_bins - 1)] += 1
            mx = max(bins)
            print(f"\n  Histogram ({lo:.0f}ms - {hi:.0f}ms):")
            for i, c in enumerate(bins):
                bar = "#" * int(c / mx * 40) if mx else ""
                print(f"  {lo + i*bw:7.1f}ms |{bar:<40s} {c}")

    return {
        "label": label,
        "times_ms": [round(t, 2) for t in times],
        "mean_ms": round(mean, 2),
        "std_ms": round(std, 2),
        "p50_ms": round(p50, 2),
        "p95_ms": round(p95, 2),
        "bimodal": bimodal,
    }


def bench(label, fn, warmup, rounds):
    """Run warmup + measurement."""
    print(f"\n--- {label} ---")
    for i in range(warmup):
        t = fn()
        print(f"  warmup {i}: {t:.1f}ms")
    times = []
    for i in range(rounds):
        t = fn()
        times.append(t)
        print(f"  [{i:2d}] {t:.1f}ms")
    return report(label, times)


def timed_forward(model, pixel_values):
    """Single forward pass, returns ms."""
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        _ = model(pixel_values)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="facebook/vjepa2-vitl-fpc16-256-ssv2")
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--output", "-o", type=str, default=None)
    args = parser.parse_args()

    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    free, total = torch.cuda.mem_get_info()
    print(f"GPU memory: {free/1024**3:.1f} / {total/1024**3:.1f} GB free")

    results = {}

    # --- A: Plain HuggingFace (baseline) ---
    from transformers import AutoModelForVideoClassification, AutoVideoProcessor

    print("\n[A] Loading via HuggingFace from_pretrained...")
    processor = AutoVideoProcessor.from_pretrained(args.model)
    model_hf = AutoModelForVideoClassification.from_pretrained(args.model)
    model_hf.to("cuda").eval()
    torch.cuda.synchronize()

    frames = [np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8) for _ in range(16)]
    inputs = processor(frames, return_tensors="pt")
    key = "pixel_values_videos" if "pixel_values_videos" in inputs else "pixel_values"
    pv = inputs[key].to("cuda")

    results["A"] = bench("A: HuggingFace from_pretrained",
                         lambda: timed_forward(model_hf, pv),
                         args.warmup, args.rounds)

    del model_hf
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # --- B: vLLM-Omni's VJepa2Model path ---
    print("\n[B] Loading via VJepa2Model (vLLM-Omni path)...")
    try:
        from vllm_omni.model_executor.models.vjepa.encoder import VJepa2Config, VJepa2Model

        config = VJepa2Config(model_id=args.model, num_frames=16, stride=8)
        model_vllm = VJepa2Model(config=config)
        model_vllm._load_model()

        results["B"] = bench("B: VJepa2Model (vLLM-Omni)",
                             lambda: timed_forward(model_vllm._model, pv.clone()),
                             args.warmup, args.rounds)

        del model_vllm
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    except ImportError as e:
        print(f"  vllm-omni not in PYTHONPATH: {e}")
        print("  Run with: PYTHONPATH=/path/to/vllm-omni:$PYTHONPATH python3 ...")

    # --- C: HuggingFace but with vLLM's weight re-loading pattern ---
    print("\n[C] Loading via HF, then re-loading weights (simulates vLLM load_weights)...")
    model_c = AutoModelForVideoClassification.from_pretrained(args.model)
    model_c.to("cuda").eval()
    torch.cuda.synchronize()

    # Simulate vLLM's load_weights: iterate params and copy them onto themselves
    # This mimics the double-load pattern
    from safetensors.torch import load_file
    from huggingface_hub import hf_hub_download
    st_path = hf_hub_download(args.model, "model.safetensors")
    checkpoint = load_file(st_path, device="cpu")
    param_dict = dict(model_c.named_parameters())
    for name, weight in checkpoint.items():
        if name in param_dict and param_dict[name].shape == weight.shape:
            param_dict[name].data.copy_(weight.to("cuda"))
    del checkpoint
    torch.cuda.synchronize()
    print("  Weights re-loaded from safetensors (double-load simulated)")

    results["C"] = bench("C: HF + weight re-load (vLLM pattern)",
                         lambda: timed_forward(model_c, pv.clone()),
                         args.warmup, args.rounds)

    del model_c
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # --- D: HuggingFace but loading weights via safetensors iterator (vLLM style) ---
    print("\n[D] Loading model skeleton, then weights via safetensors iterator...")
    from transformers import AutoConfig
    hf_config = AutoConfig.from_pretrained(args.model)
    model_d = AutoModelForVideoClassification.from_config(hf_config)
    model_d.to("cuda").eval()
    torch.cuda.synchronize()

    # Load weights one by one from safetensors (like vLLM's get_all_weights)
    from safetensors import safe_open
    param_dict = dict(model_d.named_parameters())
    with safe_open(st_path, framework="pt", device="cpu") as f:
        for name in f.keys():
            if name in param_dict:
                param_dict[name].data.copy_(f.get_tensor(name).to("cuda"))
    torch.cuda.synchronize()
    print("  Weights loaded via safetensors iterator (vLLM get_all_weights style)")

    results["D"] = bench("D: Config + safetensors iterator (vLLM loader)",
                         lambda: timed_forward(model_d, pv.clone()),
                         args.warmup, args.rounds)

    # --- Save ---
    if args.output:
        output = {
            "device": torch.cuda.get_device_name(0),
            "pytorch": torch.__version__,
            "model": args.model,
            "experiments": results,
        }
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to {args.output}")

    print("\nDone.")


if __name__ == "__main__":
    main()
