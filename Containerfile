# Stage 1: Download model weights
FROM python:3.12-slim AS downloader

RUN pip install --no-cache-dir huggingface_hub

ARG HF_MODEL_ID=facebook/vjepa2-vitl-fpc16-256-ssv2
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('${HF_MODEL_ID}', cache_dir='/model_cache')"

# Stage 2: Runtime
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies for PyAV (FFmpeg)
RUN apt-get update && \
    apt-get install -y --no-install-recommends libavcodec-dev libavformat-dev libavutil-dev libswscale-dev && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch torchvision && \
    pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY configs/ ./configs/

# Copy model weights from downloader stage
COPY --from=downloader /model_cache /root/.cache/huggingface

# Create samples directory for volume mount
RUN mkdir -p /samples

EXPOSE 8080

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
