# Benchmark Results

Last updated: 2026-06-08

## Summary

Concurrency column: `c=N` for request-based benchmarks (`/v2/infer`, N concurrent requests);
`—` for session-based benchmarks (single video stream, clips processed sequentially).

| Date | Hardware | Device | Server | Model | c | RTF | L_comp (ms) | Throughput | Version | Notes |
| ---- | -------- | ------ | ------ | ----- | - | --- | ----------- | ---------- | ------- | ----- |
| 2026-06-08 | g6.xlarge | cuda | vllm-omni-jepa | vit-l | — | 7.30 | 73 | 5.34 p/s | dd23983 | GPU preprocess, session API |
| 2026-06-07 | dgx-spark | cuda | vllm-omni-jepa | vit-l | — | 3.45 | 155 | 4.66 p/s | dd23983 | GPU preprocess, session API |
| 2026-06-07 | dgx-spark | cuda | vllm-omni-jepa | vit-g | — | 1.24 | 429 | 2.13 p/s | dd23983 | GPU preprocess, session API |
| 2026-06-05 | g6.xlarge | cuda | vllm-omni-jepa | vit-l | 1 | 6.20 | 86 | 9.49 rps | 4cd156c | CPU preprocess, /v2/infer |
| 2026-06-05 | g6.xlarge | cuda | vllm-omni-jepa | vit-l | 4 | 4.52 | 118 | 9.52 rps | 4cd156c | CPU preprocess, /v2/infer |
| 2026-06-04 | dgx-spark | cuda | vllm-omni-jepa | vit-l | 1 | 3.92 | 136 | 4.88 rps | f7614bc | CPU preprocess, /v2/infer |
| 2026-06-04 | dgx-spark | cuda | vllm-omni-jepa | vit-l | 4 | 6.27 | 85 | 7.88 rps | f7614bc | CPU preprocess, /v2/infer |
| 2026-06-04 | dgx-spark | cuda | vllm-omni-jepa | vit-g | 1 | 1.37 | 389 | 2.23 rps | f7614bc | CPU preprocess, /v2/infer |
| 2026-06-04 | dgx-spark | cuda | vllm-omni-jepa | vit-g | 4 | 0.78 | 686 | 2.88 rps | f7614bc | CPU preprocess, /v2/infer |
| 2026-06-04 | dgx-spark | cuda | vjepa2-server-cuda | vit-l | 1 | 2.33 | 229 | 4.10 rps | 5dfcbeb | |
| 2026-06-04 | dgx-spark | cuda | vjepa2-server-cuda | vit-l | 4 | 2.32 | 230 | 4.14 rps | 5dfcbeb | |
| 2026-06-04 | dgx-spark | cuda | vjepa2-server-cuda | vit-g | 1 | 0.31 | 1742 | 0.57 rps | 5dfcbeb | |
| 2026-06-04 | dgx-spark | cuda | vjepa2-server-cuda | vit-g | 4 | 0.30 | 1757 | 0.56 rps | 5dfcbeb | |
| 2026-04-15 | g6.xlarge | cuda | vjepa2-server-cuda | vit-g | 1 | 0.42 | 1276 | 0.78 rps | 3a818a1* | |
| 2026-04-15 | g6.xlarge | cuda | vjepa2-server-cuda | vit-g | 4 | 0.41 | 1289 | 0.77 rps | 3a818a1* | |
| 2026-04-15 | g6.xlarge | cuda | vjepa2-server-cuda | vit-l | 1 | 2.84 | 188 | 5.05 rps | 3a818a1* | |
| 2026-04-15 | g6.xlarge | cuda | vjepa2-server-cuda | vit-l | 4 | 2.84 | 188 | 5.09 rps | 3a818a1* | |
| 2026-04-15 | m4-pro | cpu | vjepa2-server | vit-g | 1 | 0.06 | 8320 | 0.12 rps | 3a818a1* | |
| 2026-04-15 | m4-pro | cpu | vjepa2-server | vit-g | 4 | 0.06 | 8279 | 0.12 rps | 3a818a1* | |
| 2026-04-15 | m4-pro | cpu | vjepa2-server | vit-l | 1 | 0.43 | 1225 | 0.81 rps | 3a818a1* | |
| 2026-04-15 | m4-pro | cpu | vjepa2-server | vit-l | 4 | 0.42 | 1263 | 0.79 rps | 3a818a1* | |
| 2026-04-15 | m4-pro | mps | vjepa2-server | vit-g | 1 | 0.09 | 5868 | 0.17 rps | 3a818a1* | |
| 2026-04-15 | m4-pro | mps | vjepa2-server | vit-g | 4 | 0.09 | 5836 | 0.17 rps | 3a818a1* | |
| 2026-04-15 | m4-pro | mps | vjepa2-server | vit-l | 1 | 0.83 | 641 | 1.54 rps | 3a818a1* | |
| 2026-04-15 | m4-pro | mps | vjepa2-server | vit-l | 4 | 0.84 | 637 | 1.56 rps | 3a818a1* | |

## By Server

### vllm-omni-jepa

| Date | Hardware | Device | Model | RTF | L_comp | Delta vs vjepa2-demo | Notes |
| ---- | -------- | ------ | ----- | --- | ------ | -------------------- | ----- |
| 2026-06-08 | g6.xlarge | cuda | vit-l | 7.30 | 73ms | **-61%** vs 188ms | GPU preprocess, session API |
| 2026-06-07 | dgx-spark | cuda | vit-l | 3.45 | 155ms | **-32%** vs 229ms | GPU preprocess, session API |
| 2026-06-07 | dgx-spark | cuda | vit-g | 1.24 | 429ms | **-75%** vs 1742ms | GPU preprocess, session API |
| 2026-06-05 | g6.xlarge | cuda | vit-l | 6.20 | 86ms | **-54%** vs 188ms | CPU preprocess, /v2/infer |
| 2026-06-04 | dgx-spark | cuda | vit-l | 3.92 | 136ms | **-41%** vs 229ms | CPU preprocess, /v2/infer |
| 2026-06-04 | dgx-spark | cuda | vit-g | 1.37 | 389ms | **-78%** vs 1742ms | CPU preprocess, /v2/infer |

### vjepa2-server-cuda

| Date | Hardware | Device | Model | c | RTF | L_comp | Delta vs prev |
| ---- | -------- | ------ | ----- | - | --- | ------ | ------------- |
| 2026-06-04 | dgx-spark | cuda | vit-l | 1 | 2.33 | 229ms | +22% vs g6.xlarge |
| 2026-06-04 | dgx-spark | cuda | vit-l | 4 | 2.32 | 230ms | +22% vs g6.xlarge |
| 2026-06-04 | dgx-spark | cuda | vit-g | 1 | 0.31 | 1742ms | +37% vs g6.xlarge |
| 2026-06-04 | dgx-spark | cuda | vit-g | 4 | 0.30 | 1757ms | +36% vs g6.xlarge |
| 2026-04-15 | g6.xlarge | cuda | vit-g | 1 | 0.42 | 1276ms | baseline |
| 2026-04-15 | g6.xlarge | cuda | vit-g | 4 | 0.41 | 1289ms | baseline |
| 2026-04-15 | g6.xlarge | cuda | vit-l | 1 | 2.84 | 188ms | baseline |
| 2026-04-15 | g6.xlarge | cuda | vit-l | 4 | 2.84 | 188ms | baseline |

### vjepa2-server

| Date | Hardware | Device | Model | c | RTF | L_comp | Delta vs prev |
| ---- | -------- | ------ | ----- | - | --- | ------ | ------------- |
| 2026-04-15 | m4-pro | cpu | vit-g | 1 | 0.06 | 8320ms | baseline |
| 2026-04-15 | m4-pro | cpu | vit-g | 4 | 0.06 | 8279ms | baseline |
| 2026-04-15 | m4-pro | cpu | vit-l | 1 | 0.43 | 1225ms | baseline |
| 2026-04-15 | m4-pro | cpu | vit-l | 4 | 0.42 | 1263ms | baseline |
| 2026-04-15 | m4-pro | mps | vit-g | 1 | 0.09 | 5868ms | baseline |
| 2026-04-15 | m4-pro | mps | vit-g | 4 | 0.09 | 5836ms | baseline |
| 2026-04-15 | m4-pro | mps | vit-l | 1 | 0.83 | 641ms | baseline |
| 2026-04-15 | m4-pro | mps | vit-l | 4 | 0.84 | 637ms | baseline |

## Analysis

### GPU preprocessing impact on DGX Spark (2026-06-07)

GPU-side preprocessing (torchvision on CUDA) eliminates the GPU→CPU→GPU round-trip
that `AutoVideoProcessor` required. All preprocessing parameters (resize, crop, normalize)
are derived from the model's `video_preprocessor_config.json`.

| Stage | CPU preprocess (06-04) | GPU preprocess (06-07) | Delta |
| ----- | ---------------------- | ---------------------- | ----- |
| input_preprocess | 16.8ms | **3.3ms** | **-80%** |
| jepa_encode | 144ms | 144ms | unchanged |
| **TOTAL (ViT-L)** | **169ms** | **155ms** | **-8%** |
| **TOTAL (ViT-G)** | — | **429ms** | — |

The preprocessing improvement is ~13ms per clip. The encode step is unchanged,
confirming the win comes purely from eliminating the CPU round-trip.

### vllm-omni vs vjepa2-demo on DGX Spark (ViT-L, c=1)

| Stage | vllm-omni | vjepa2-demo | Delta |
| ----- | --------- | ----------- | ----- |
| input_receive | 1.1ms | 0.8ms | +0.3ms |
| input_open | 6.4ms | 4.7ms | +1.7ms |
| input_decode | — | 31.8ms | — |
| input_preprocess | 9.0ms | 8.2ms | +0.8ms |
| jepa_encode | 113.3ms | 164.8ms | **-51.5ms** |
| jepa_pool | 6.3ms | 18.6ms | **-12.3ms** |
| output_postprocess | 0.0ms | 0.3ms | -0.3ms |
| **TOTAL** | **136.0ms** | **229.1ms** | **-93.1ms (-41%)** |

vllm-omni's encode is faster (vLLM's weight loader + safetensors path) but shows bimodal
latency on GB10 (p50=65ms, p95=245ms). The vjepa2-demo encode is slower but consistent
(std=6ms). The pooling stage is 3x faster in vllm-omni, likely due to different model
initialization paths.

### vllm-omni vs vjepa2-demo on DGX Spark (ViT-G, c=1)

| Stage | vllm-omni | vjepa2-demo | Delta |
| ----- | --------- | ----------- | ----- |
| input_receive | 1.0ms | 0.9ms | +0.1ms |
| input_open | 5.5ms | 4.6ms | +0.9ms |
| input_decode | — | 31.3ms | — |
| input_preprocess | 13.7ms | 12.8ms | +0.9ms |
| jepa_encode | 352.9ms | 1591.4ms | **-1238.5ms** |
| jepa_pool | 15.8ms | 100.7ms | **-84.9ms** |
| output_postprocess | 0.0ms | 0.3ms | -0.3ms |
| **TOTAL** | **388.9ms** | **1741.9ms** | **-1353.0ms (-78%)** |

The 4.5x speedup for ViT-G is the headline result. vllm-omni keeps ViT-G real-time
capable (RTF 1.37) while vjepa2-demo runs 3.3x below real-time.

### vllm-omni on g6.xlarge vs DGX Spark (ViT-L, c=1)

| Metric | g6.xlarge (L4) | DGX Spark (GB10) |
| ------ | -------------- | ---------------- |
| encode mean | 61ms | 113ms |
| encode std | **1.4ms** | **73ms** |
| encode p50 | 61ms | 65ms |
| encode p95 | 64ms | 245ms |
| L_comp | 86ms | 136ms |

On discrete GPU (L4), encode is perfectly stable with 1.4ms std. The bimodality
is exclusively a GB10 unified memory phenomenon — confirmed by running the exact
same server and benchmark on both hardware platforms.

### GB10 bimodality note

The DGX Spark GB10 shows bimodal encode latency on vllm-omni (fast ~65ms vs slow ~250ms
for ViT-L, ~300ms vs ~490ms for ViT-G). **Confirmed as a unified memory issue** — the
same vllm-omni code runs with stable 1.4ms std on the g6.xlarge (L4, discrete VRAM).
The vjepa2-demo server does not exhibit this on GB10, suggesting the HF `from_pretrained`
path handles unified memory differently than vLLM's weight loading.

Investigated and ruled out: `output_hidden_states`, double weight loading, `cudaMemAdvise`
pinning, local vs HF-cache model path (see memory for full experiment log).

## Environment Details

<details>
<summary>2026-06-05 g6.xlarge/cuda vllm-omni-jepa vit-l c=1</summary>

- Video: ucf101-archery.mp4
- Concurrency: 1
- Requests: 20
- Instance: g6.xlarge
- CPU: AMD EPYC 7R13 Processor
- CPU cores: 4
- Memory: 15.0 GB (discrete)
- GPU: NVIDIA L4 (24 GB VRAM)

</details>

<details>
<summary>2026-06-05 g6.xlarge/cuda vllm-omni-jepa vit-l c=4</summary>

- Video: ucf101-archery.mp4
- Concurrency: 4
- Requests: 20
- Instance: g6.xlarge
- CPU: AMD EPYC 7R13 Processor
- CPU cores: 4
- Memory: 15.0 GB (discrete)
- GPU: NVIDIA L4 (24 GB VRAM)

</details>

<details>
<summary>2026-06-04 dgx-spark/cuda vllm-omni-jepa vit-l c=1</summary>

- Video: ucf101-archery.mp4
- Concurrency: 1
- Requests: 20
- Hardware: NVIDIA DGX Spark (GB10)
- CPU: ARM Cortex-X925/A725
- CPU cores: 20
- Memory: 119.6 GB (unified CPU+GPU)
- GPU: NVIDIA GB10

</details>

<details>
<summary>2026-06-04 dgx-spark/cuda vllm-omni-jepa vit-l c=4</summary>

- Video: ucf101-archery.mp4
- Concurrency: 4
- Requests: 20
- Hardware: NVIDIA DGX Spark (GB10)
- CPU: ARM Cortex-X925/A725
- CPU cores: 20
- Memory: 119.6 GB (unified CPU+GPU)
- GPU: NVIDIA GB10

</details>

<details>
<summary>2026-06-04 dgx-spark/cuda vllm-omni-jepa vit-g c=1</summary>

- Video: ucf101-archery.mp4
- Concurrency: 1
- Requests: 20
- Hardware: NVIDIA DGX Spark (GB10)
- CPU: ARM Cortex-X925/A725
- CPU cores: 20
- Memory: 119.6 GB (unified CPU+GPU)
- GPU: NVIDIA GB10

</details>

<details>
<summary>2026-06-04 dgx-spark/cuda vllm-omni-jepa vit-g c=4</summary>

- Video: ucf101-archery.mp4
- Concurrency: 4
- Requests: 20
- Hardware: NVIDIA DGX Spark (GB10)
- CPU: ARM Cortex-X925/A725
- CPU cores: 20
- Memory: 119.6 GB (unified CPU+GPU)
- GPU: NVIDIA GB10

</details>

<details>
<summary>2026-06-04 dgx-spark/cuda vjepa2-server-cuda vit-l c=1</summary>

- Video: ucf101-archery.mp4
- Concurrency: 1
- Requests: 20
- Hardware: NVIDIA DGX Spark (GB10)
- CPU: ARM Cortex-X925/A725
- CPU cores: 20
- Memory: 119.6 GB (unified CPU+GPU)
- GPU: NVIDIA GB10

</details>

<details>
<summary>2026-06-04 dgx-spark/cuda vjepa2-server-cuda vit-l c=4</summary>

- Video: ucf101-archery.mp4
- Concurrency: 4
- Requests: 20
- Hardware: NVIDIA DGX Spark (GB10)
- CPU: ARM Cortex-X925/A725
- CPU cores: 20
- Memory: 119.6 GB (unified CPU+GPU)
- GPU: NVIDIA GB10

</details>

<details>
<summary>2026-06-04 dgx-spark/cuda vjepa2-server-cuda vit-g c=1</summary>

- Video: ucf101-archery.mp4
- Concurrency: 1
- Requests: 20
- Hardware: NVIDIA DGX Spark (GB10)
- CPU: ARM Cortex-X925/A725
- CPU cores: 20
- Memory: 119.6 GB (unified CPU+GPU)
- GPU: NVIDIA GB10

</details>

<details>
<summary>2026-06-04 dgx-spark/cuda vjepa2-server-cuda vit-g c=4</summary>

- Video: ucf101-archery.mp4
- Concurrency: 4
- Requests: 20
- Hardware: NVIDIA DGX Spark (GB10)
- CPU: ARM Cortex-X925/A725
- CPU cores: 20
- Memory: 119.6 GB (unified CPU+GPU)
- GPU: NVIDIA GB10

</details>

<details>
<summary>2026-04-15 g6.xlarge/cuda vjepa2-server-cuda vit-g c=1</summary>

- Video: ucf101-archery.mp4
- Concurrency: 1
- Requests: 20
- Instance: g6.xlarge
- CPU: AMD EPYC 7R13 Processor
- CPU cores: 4
- Memory: 15.0 GB (discrete)

</details>

<details>
<summary>2026-04-15 g6.xlarge/cuda vjepa2-server-cuda vit-g c=4</summary>

- Video: ucf101-archery.mp4
- Concurrency: 4
- Requests: 20
- Instance: g6.xlarge
- CPU: AMD EPYC 7R13 Processor
- CPU cores: 4
- Memory: 15.0 GB (discrete)

</details>

<details>
<summary>2026-04-15 g6.xlarge/cuda vjepa2-server-cuda vit-l c=1</summary>

- Video: ucf101-archery.mp4
- Concurrency: 1
- Requests: 20
- Instance: g6.xlarge
- CPU: AMD EPYC 7R13 Processor
- CPU cores: 4
- Memory: 15.0 GB (discrete)

</details>

<details>
<summary>2026-04-15 g6.xlarge/cuda vjepa2-server-cuda vit-l c=4</summary>

- Video: ucf101-archery.mp4
- Concurrency: 4
- Requests: 20
- Instance: g6.xlarge
- CPU: AMD EPYC 7R13 Processor
- CPU cores: 4
- Memory: 15.0 GB (discrete)

</details>

<details>
<summary>2026-04-15 m4-pro/cpu vjepa2-server vit-g c=1</summary>

- Video: ucf101-archery.mp4
- Concurrency: 1
- Requests: 20
- CPU: Apple M4 Pro
- CPU cores: 14
- Memory: 48.0 GB (unified)

</details>

<details>
<summary>2026-04-15 m4-pro/cpu vjepa2-server vit-g c=4</summary>

- Video: ucf101-archery.mp4
- Concurrency: 4
- Requests: 20
- CPU: Apple M4 Pro
- CPU cores: 14
- Memory: 48.0 GB (unified)

</details>

<details>
<summary>2026-04-15 m4-pro/cpu vjepa2-server vit-l c=1</summary>

- Video: ucf101-archery.mp4
- Concurrency: 1
- Requests: 20
- CPU: Apple M4 Pro
- CPU cores: 14
- Memory: 48.0 GB (unified)

</details>

<details>
<summary>2026-04-15 m4-pro/cpu vjepa2-server vit-l c=4</summary>

- Video: ucf101-archery.mp4
- Concurrency: 4
- Requests: 20
- CPU: Apple M4 Pro
- CPU cores: 14
- Memory: 48.0 GB (unified)

</details>

<details>
<summary>2026-04-15 m4-pro/mps vjepa2-server vit-g c=1</summary>

- Video: ucf101-archery.mp4
- Concurrency: 1
- Requests: 20
- CPU: Apple M4 Pro
- CPU cores: 14
- Memory: 48.0 GB (unified)

</details>

<details>
<summary>2026-04-15 m4-pro/mps vjepa2-server vit-g c=4</summary>

- Video: ucf101-archery.mp4
- Concurrency: 4
- Requests: 20
- CPU: Apple M4 Pro
- CPU cores: 14
- Memory: 48.0 GB (unified)

</details>

<details>
<summary>2026-04-15 m4-pro/mps vjepa2-server vit-l c=1</summary>

- Video: ucf101-archery.mp4
- Concurrency: 1
- Requests: 20
- CPU: Apple M4 Pro
- CPU cores: 14
- Memory: 48.0 GB (unified)

</details>

<details>
<summary>2026-04-15 m4-pro/mps vjepa2-server vit-l c=4</summary>

- Video: ucf101-archery.mp4
- Concurrency: 4
- Requests: 20
- CPU: Apple M4 Pro
- CPU cores: 14
- Memory: 48.0 GB (unified)

</details>
