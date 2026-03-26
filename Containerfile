FROM python:3.12-slim

WORKDIR /app

# System dependencies for PyAV (FFmpeg)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      libavcodec-dev libavformat-dev libavutil-dev libswscale-dev && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir \
      --index-url https://download.pytorch.org/whl/cpu torch torchvision && \
    pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY configs/ ./configs/

# No model weights -- loaded from volume at runtime
VOLUME /model
VOLUME /input
VOLUME /output

RUN mkdir -p /input /output

EXPOSE 8080

ENTRYPOINT ["python", "-m", "app"]
CMD ["serve"]
