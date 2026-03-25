# app/main.py
import os
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
