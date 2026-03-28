# app/main.py
import os
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import yaml
from fastapi import FastAPI, File, Form, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from app.model import VJepa2Model, select_device
from app.schemas import (
    InferenceResponse,
    ModelMetadata,
    OutputTensor,
    Prediction,
    StreamComplete,
    StreamConfig,
    StreamPrediction,
)
from app.telemetry import get_meter, get_tracer, init_telemetry, instrument_fastapi
from app.video import decode_video, iter_clips

# Load config
_config_path = os.environ.get("CONFIG_PATH", "configs/model_config.yaml")
with open(_config_path) as f:
    CONFIG = yaml.safe_load(f)

_model: VJepa2Model | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model
    device = select_device(os.environ.get("DEVICE"))

    init_telemetry(
        service_name="vjepa2-server",
        device=device,
        otlp_endpoint=os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"),
    )

    meter = get_meter()
    model_load_gauge = meter.create_gauge(
        "vjepa2_model_load_duration_seconds",
        description="Time to load model at startup",
        unit="s",
    )

    model_path = os.environ.get(
        "MODEL_PATH",
        CONFIG["model"].get("model_path", CONFIG["model"]["hf_model_id"]),
    )

    load_start = time.monotonic()
    _model = VJepa2Model(
        model_path=model_path,
        device=device,
    )
    model_load_gauge.set(time.monotonic() - load_start)

    instrument_fastapi(app)

    yield
    _model = None


app = FastAPI(title="V-JEPA2 Inference", lifespan=lifespan)


@app.get("/v2/health/live")
def health_live():
    return {"status": "alive"}


@app.get("/v2/health/ready")
def health_ready():
    if _model is None:
        return JSONResponse({"error": "Model not ready"}, status_code=503)
    return {"status": "ready"}


@app.get("/v2/models/vjepa2")
def model_metadata():
    return ModelMetadata(
        name=CONFIG["model"]["name"],
        versions=["vit-l-16-ssv2"],
        platform="pytorch",
    )


@app.post("/v2/models/vjepa2/infer")
async def infer(
    file: UploadFile = File(...),
    top_k: int = Form(default=CONFIG["inference"]["default_top_k"]),
    num_frames: int = Form(default=CONFIG["inference"]["num_frames"]),
    stride: int | None = Form(default=None),
):
    if _model is None:
        return JSONResponse({"error": "Model not ready"}, status_code=503)

    meter = get_meter()
    request_duration = meter.create_histogram(
        "vjepa2_request_duration_seconds",
        description="End-to-end API call duration",
        unit="s",
    )
    requests_total = meter.create_counter(
        "vjepa2_requests_total",
        description="Total API requests",
    )
    frames_total = meter.create_counter(
        "vjepa2_frames_processed_total",
        description="Total video frames decoded",
    )

    request_start = time.monotonic()

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    tracer = get_tracer()
    try:
        if stride is not None:
            clips_results = []
            with tracer.start_as_current_span("video_inference"):
                for clip in iter_clips(tmp_path, num_frames=num_frames, stride=stride):
                    with tracer.start_as_current_span("clip_inference"):
                        predictions = _model.predict(
                            clip.frames, top_k=top_k, stride=stride
                        )
                    frames_total.add(clip.end_frame - clip.start_frame)
                    clips_results.append({
                        "clip_index": len(clips_results),
                        "start_frame": clip.start_frame,
                        "end_frame": clip.end_frame,
                        "partial": (clip.end_frame - clip.start_frame) < num_frames,
                        "predictions": [p.model_dump() for p in predictions],
                    })
            requests_total.add(1, {"endpoint": "/infer", "status": "200"})
            request_duration.record(time.monotonic() - request_start)
            return {
                "model_name": CONFIG["model"]["name"],
                "model_version": "vit-l-16-ssv2",
                "id": f"req-{uuid.uuid4().hex[:8]}",
                "clips": clips_results,
            }
        else:
            with tracer.start_as_current_span("video_inference"):
                with tracer.start_as_current_span("clip_inference"):
                    frames = decode_video(tmp_path, num_frames=num_frames)
                    predictions = _model.predict(frames, top_k=top_k)
            frames_total.add(num_frames)
            requests_total.add(1, {"endpoint": "/infer", "status": "200"})
            request_duration.record(time.monotonic() - request_start)
            return InferenceResponse(
                model_name=CONFIG["model"]["name"],
                model_version="vit-l-16-ssv2",
                id=f"req-{uuid.uuid4().hex[:8]}",
                outputs=[
                    OutputTensor(
                        name="predictions",
                        shape=[len(predictions)],
                        datatype="FP32",
                        data=predictions,
                    )
                ],
            )
    except ValueError as e:
        requests_total.add(1, {"endpoint": "/infer", "status": "400"})
        request_duration.record(time.monotonic() - request_start)
        return JSONResponse({"error": str(e)}, status_code=400)
    except Exception as e:
        requests_total.add(1, {"endpoint": "/infer", "status": "400"})
        request_duration.record(time.monotonic() - request_start)
        return JSONResponse(
            {"error": f"Could not decode video: {e}"}, status_code=400
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@app.websocket("/v2/models/vjepa2/stream")
async def stream(websocket: WebSocket):
    await websocket.accept()

    if _model is None:
        await websocket.send_json({"error": "Model not ready"})
        await websocket.close(code=1011)
        return

    meter = get_meter()
    active_connections = meter.create_up_down_counter(
        "vjepa2_active_connections",
        description="Current WebSocket connections",
    )
    active_connections.add(1)

    try:
        raw = await websocket.receive_json()
        config = StreamConfig(**raw)
    except Exception as e:
        active_connections.add(-1)
        await websocket.send_json({"error": f"Invalid config: {e}"})
        await websocket.close(code=1008)
        return

    source_path = Path(config.source)
    allowed_dir = Path(CONFIG["server"]["allowed_source_dir"])

    # Security: resolve and check path is within allowed directory
    try:
        resolved = source_path.resolve()
        if not str(resolved).startswith(str(allowed_dir.resolve())):
            active_connections.add(-1)
            await websocket.send_json({"error": "Source path not allowed"})
            await websocket.close(code=1008)
            return
    except Exception:
        active_connections.add(-1)
        await websocket.send_json({"error": "Source path not allowed"})
        await websocket.close(code=1008)
        return

    if not source_path.exists():
        active_connections.add(-1)
        await websocket.send_json({"error": f"File not found: {config.source}"})
        await websocket.close(code=1008)
        return

    frames_processed = 0
    tracer = get_tracer()
    try:
        with tracer.start_as_current_span("video_inference"):
            for clip in iter_clips(source_path, config.num_frames, config.stride):
                with tracer.start_as_current_span("clip_inference"):
                    predictions = _model.predict(
                        clip.frames, top_k=config.top_k, stride=config.stride
                    )
                msg = StreamPrediction(
                    timestamp_ms=int(clip.start_frame * 1000 / 30),
                    frame_range=[clip.start_frame, clip.end_frame],
                    predictions=predictions,
                )
                await websocket.send_json(msg.model_dump())
                frames_processed = clip.end_frame
    except WebSocketDisconnect:
        active_connections.add(-1)
        return
    except Exception as e:
        active_connections.add(-1)
        await websocket.send_json({"error": f"Inference failed: {e}"})
        await websocket.close(code=1011)
        return

    active_connections.add(-1)
    await websocket.send_json(
        StreamComplete(status="complete", frames_processed=frames_processed).model_dump()
    )
    await websocket.close()
