# app/schemas.py
from dataclasses import dataclass
import numpy as np
from pydantic import BaseModel, Field, field_validator

# Inference defaults — single source of truth
DEFAULT_NUM_FRAMES = 16
DEFAULT_STRIDE = 16
DEFAULT_TOP_K = 3


@dataclass
class Clip:
    frames: np.ndarray    # (num_frames, H, W, 3) uint8
    start_frame: int      # index in source video
    end_frame: int        # exclusive end index


class Prediction(BaseModel):
    label: str
    score: float = Field(ge=0.0, le=1.0)


class OutputTensor(BaseModel):
    name: str
    shape: list[int]
    datatype: str
    data: list[Prediction]


class InferenceResponse(BaseModel):
    model_name: str
    model_version: str
    id: str
    outputs: list[OutputTensor]


class ModelMetadata(BaseModel):
    name: str
    versions: list[str]
    platform: str
    inputs: list[dict] = [
        {"name": "video", "datatype": "BYTES", "shape": [-1]}
    ]
    outputs: list[dict] = [
        {"name": "predictions", "datatype": "FP32", "shape": [-1]}
    ]


class StreamConfig(BaseModel):
    source: str
    top_k: int = DEFAULT_TOP_K
    num_frames: int = DEFAULT_NUM_FRAMES
    stride: int = DEFAULT_STRIDE

    @field_validator("source")
    @classmethod
    def validate_source_path(cls, v: str) -> str:
        if ".." in v:
            raise ValueError("Path traversal not allowed")
        return v


class StreamPrediction(BaseModel):
    timestamp_ms: int
    frame_range: list[int]
    predictions: list[Prediction]


class StreamComplete(BaseModel):
    status: str
    frames_processed: int


class BrowserStreamConfig(BaseModel):
    top_k: int = DEFAULT_TOP_K
    num_frames: int = DEFAULT_NUM_FRAMES
    stride: int = DEFAULT_STRIDE


class RtspStreamConfig(BaseModel):
    rtsp_url: str
    top_k: int = DEFAULT_TOP_K
    num_frames: int = DEFAULT_NUM_FRAMES
    stride: int = DEFAULT_STRIDE

    @field_validator("rtsp_url")
    @classmethod
    def validate_rtsp_url(cls, v: str) -> str:
        if not v.startswith("rtsp://"):
            raise ValueError("URL must start with rtsp://")
        return v


class StreamAction(BaseModel):
    action: str  # "stop"


class PredictionMessage(BaseModel):
    type: str = "prediction"
    clip_index: int
    timestamp_ms: int
    frame_range: list[int]
    thumbnail: str  # base64 JPEG
    predictions: list[Prediction]


class SessionMessage(BaseModel):
    type: str = "session"
    session_id: str
    status: str


class CompleteMessage(BaseModel):
    type: str = "complete"
    session_id: str
    clips_processed: int
    video_ready: bool


class ErrorMessage(BaseModel):
    type: str = "error"
    message: str
