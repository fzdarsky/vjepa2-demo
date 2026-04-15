# Benchmark Results

Last updated: 2026-04-15

## Summary

| Date | Hardware | Device | Server | Model | c | RTF | L_comp (ms) | Throughput | Version |
| ---- | -------- | ------ | ------ | ----- | - | --- | ----------- | ---------- | ------- |
| 2026-04-15 | g6.xlarge | cuda | vjepa2-server-cuda | vit-g | 1 | 0.42 | 1276 | 0.78 rps | 3a818a1* |
| 2026-04-15 | g6.xlarge | cuda | vjepa2-server-cuda | vit-g | 4 | 0.41 | 1289 | 0.77 rps | 3a818a1* |
| 2026-04-15 | g6.xlarge | cuda | vjepa2-server-cuda | vit-l | 1 | 2.84 | 188 | 5.05 rps | 3a818a1* |
| 2026-04-15 | g6.xlarge | cuda | vjepa2-server-cuda | vit-l | 4 | 2.84 | 188 | 5.09 rps | 3a818a1* |
| 2026-04-15 | m4-pro | cpu | vjepa2-server | vit-g | 1 | 0.06 | 8320 | 0.12 rps | 3a818a1* |
| 2026-04-15 | m4-pro | cpu | vjepa2-server | vit-g | 4 | 0.06 | 8279 | 0.12 rps | 3a818a1* |
| 2026-04-15 | m4-pro | cpu | vjepa2-server | vit-l | 1 | 0.43 | 1225 | 0.81 rps | 3a818a1* |
| 2026-04-15 | m4-pro | cpu | vjepa2-server | vit-l | 4 | 0.42 | 1263 | 0.79 rps | 3a818a1* |
| 2026-04-15 | m4-pro | mps | vjepa2-server | vit-g | 1 | 0.09 | 5868 | 0.17 rps | 3a818a1* |
| 2026-04-15 | m4-pro | mps | vjepa2-server | vit-g | 4 | 0.09 | 5836 | 0.17 rps | 3a818a1* |
| 2026-04-15 | m4-pro | mps | vjepa2-server | vit-l | 1 | 0.83 | 641 | 1.54 rps | 3a818a1* |
| 2026-04-15 | m4-pro | mps | vjepa2-server | vit-l | 4 | 0.84 | 637 | 1.56 rps | 3a818a1* |

## By Server

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

### vjepa2-server-cuda

| Date | Hardware | Device | Model | c | RTF | L_comp | Delta vs prev |
| ---- | -------- | ------ | ----- | - | --- | ------ | ------------- |
| 2026-04-15 | g6.xlarge | cuda | vit-g | 1 | 0.42 | 1276ms | baseline |
| 2026-04-15 | g6.xlarge | cuda | vit-g | 4 | 0.41 | 1289ms | baseline |
| 2026-04-15 | g6.xlarge | cuda | vit-l | 1 | 2.84 | 188ms | baseline |
| 2026-04-15 | g6.xlarge | cuda | vit-l | 4 | 2.84 | 188ms | baseline |

## Environment Details

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
