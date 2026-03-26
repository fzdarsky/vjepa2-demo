# app/main.py
import os
import tempfile
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
from app.video import decode_video, stream_frames

# Load config
_config_path = os.environ.get("CONFIG_PATH", "configs/model_config.yaml")
with open(_config_path) as f:
    CONFIG = yaml.safe_load(f)

_model: VJepa2Model | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model
    device = select_device(os.environ.get("DEVICE"))
    model_path = os.environ.get(
        "MODEL_PATH",
        CONFIG["model"].get("model_path", CONFIG["model"]["hf_model_id"]),
    )
    _model = VJepa2Model(
        model_path=model_path,
        device=device,
    )
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
):
    if _model is None:
        return JSONResponse({"error": "Model not ready"}, status_code=503)

    # Save uploaded file to temp location for PyAV
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        frames = decode_video(tmp_path, num_frames=num_frames)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    except Exception as e:
        return JSONResponse(
            {"error": f"Could not decode video: {e}"}, status_code=400
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    predictions = _model.predict(frames, top_k=top_k)

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


@app.websocket("/v2/models/vjepa2/stream")
async def stream(websocket: WebSocket):
    await websocket.accept()

    if _model is None:
        await websocket.send_json({"error": "Model not ready"})
        await websocket.close(code=1011)
        return

    try:
        raw = await websocket.receive_json()
        config = StreamConfig(**raw)
    except Exception as e:
        await websocket.send_json({"error": f"Invalid config: {e}"})
        await websocket.close(code=1008)
        return

    source_path = Path(config.source)
    allowed_dir = Path(CONFIG["server"]["allowed_source_dir"])

    # Security: resolve and check path is within allowed directory
    try:
        resolved = source_path.resolve()
        if not str(resolved).startswith(str(allowed_dir.resolve())):
            await websocket.send_json({"error": "Source path not allowed"})
            await websocket.close(code=1008)
            return
    except Exception:
        await websocket.send_json({"error": "Source path not allowed"})
        await websocket.close(code=1008)
        return

    if not source_path.exists():
        await websocket.send_json({"error": f"File not found: {config.source}"})
        await websocket.close(code=1008)
        return

    frames_processed = 0
    try:
        for window in stream_frames(source_path, config.num_frames, config.stride):
            predictions = _model.predict(window, top_k=config.top_k)
            msg = StreamPrediction(
                timestamp_ms=int(frames_processed * 1000 / 30),
                frame_range=[frames_processed, frames_processed + config.num_frames],
                predictions=predictions,
            )
            await websocket.send_json(msg.model_dump())
            frames_processed += config.stride
    except WebSocketDisconnect:
        return
    except Exception as e:
        await websocket.send_json({"error": f"Inference failed: {e}"})
        await websocket.close(code=1011)
        return

    await websocket.send_json(
        StreamComplete(status="complete", frames_processed=frames_processed).model_dump()
    )
    await websocket.close()
