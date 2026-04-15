# Benchmark Reference Videos

Standard reference videos for reproducible V-JEPA2 benchmarking.

## Videos

| File | Source | Size | Description |
|------|--------|------|-------------|
| `ucf101-archery.mp4` | UCF-101 | 537K | Human archery action |
| `ucf101-baby-crawling.avi` | UCF-101 | 303K | Baby crawling action |
| `ucf101-basketball-dunk.avi` | UCF-101 | 534K | Basketball dunk action |
| `people-detection.mp4` | Intel | 5.3M | People walking in scene |
| `person-bicycle-car-detection.mp4` | Intel | 5.8M | Mixed person/vehicle detection |

## Recommended for Benchmarks

For action recognition benchmarks, use the UCF-101 videos:

```bash
python -m benchmark.run \
    --target http://localhost:8000 \
    --video benchmark/videos/ucf101-archery.mp4 \
    --server vjepa2-server-cuda \
    --model vit-l
```

## Sources & Licensing

### UCF-101 Dataset

The UCF-101 videos are from the [UCF-101 Action Recognition Dataset](https://www.crcv.ucf.edu/data/UCF101.php), a widely-used benchmark containing 13,320 videos across 101 action categories.

- **Paper:** [UCF101: A Dataset of 101 Human Actions Classes From Videos in The Wild](https://arxiv.org/abs/1212.0402)
- **License:** Research use (see [UCF terms](https://www.crcv.ucf.edu/data/UCF101.php))
- **Downloaded from:** [Hugging Face sayakpaul/ucf101-subset](https://huggingface.co/datasets/sayakpaul/ucf101-subset), [nateraw/dino-clips](https://huggingface.co/spaces/nateraw/dino-clips)

### Intel Sample Videos

The Intel videos are from the [intel-iot-devkit/sample-videos](https://github.com/intel-iot-devkit/sample-videos) repository, designed for inference testing with Intel OpenVINO.

- **License:** MIT License
- **Source:** https://github.com/intel-iot-devkit/sample-videos

## Re-downloading

To refresh or re-download these videos:

```bash
# Intel videos
curl -LO https://github.com/intel-iot-devkit/sample-videos/raw/master/people-detection.mp4
curl -LO https://github.com/intel-iot-devkit/sample-videos/raw/master/person-bicycle-car-detection.mp4

# UCF-101 videos (requires huggingface_hub)
python -c "
from huggingface_hub import hf_hub_download
import shutil

# Archery
f = hf_hub_download('nateraw/dino-clips', 'archery.mp4', repo_type='space')
shutil.copy(f, 'ucf101-archery.mp4')

# Baby crawling
f = hf_hub_download('sayakpaul/ucf101-subset', 'v_BabyCrawling_g19_c02.avi', repo_type='dataset')
shutil.copy(f, 'ucf101-baby-crawling.avi')

# Basketball dunk
f = hf_hub_download('sayakpaul/ucf101-subset', 'v_BasketballDunk_g14_c06.avi', repo_type='dataset')
shutil.copy(f, 'ucf101-basketball-dunk.avi')
"
```
