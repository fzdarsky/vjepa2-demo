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
    _model = VJepa2Model(
        hf_model_id=CONFIG["model"]["hf_model_id"],
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
