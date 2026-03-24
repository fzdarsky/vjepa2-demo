# app/schemas.py
from pydantic import BaseModel, Field, field_validator


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
    top_k: int = 5
    num_frames: int = 16
    stride: int = 8

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
