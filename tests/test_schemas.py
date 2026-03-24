# tests/test_schemas.py
import pytest
from app.schemas import (
    Prediction,
    InferenceResponse,
    ModelMetadata,
    StreamConfig,
    StreamPrediction,
    StreamComplete,
)


def test_prediction_valid():
    p = Prediction(label="Pushing something", score=0.87)
    assert p.label == "Pushing something"
    assert p.score == 0.87


def test_prediction_score_bounds():
    with pytest.raises(Exception):
        Prediction(label="x", score=1.5)
    with pytest.raises(Exception):
        Prediction(label="x", score=-0.1)


def test_inference_response_structure():
    resp = InferenceResponse(
        model_name="vjepa2",
        model_version="vit-l-16-ssv2",
        id="req-abc",
        outputs=[
            {
                "name": "predictions",
                "shape": [2],
                "datatype": "FP32",
                "data": [
                    {"label": "Action A", "score": 0.9},
                    {"label": "Action B", "score": 0.1},
                ],
            }
        ],
    )
    assert resp.model_name == "vjepa2"
    assert len(resp.outputs) == 1
    assert resp.outputs[0].data[0].label == "Action A"


def test_model_metadata():
    meta = ModelMetadata(
        name="vjepa2",
        versions=["vit-l-16-ssv2"],
        platform="pytorch",
    )
    assert meta.name == "vjepa2"


def test_stream_config_defaults():
    cfg = StreamConfig(source="/samples/video.mp4")
    assert cfg.top_k == 5
    assert cfg.num_frames == 16
    assert cfg.stride == 8


def test_stream_config_rejects_path_traversal():
    with pytest.raises(Exception):
        StreamConfig(source="/samples/../etc/passwd")


def test_stream_prediction():
    sp = StreamPrediction(
        timestamp_ms=1500,
        frame_range=[24, 40],
        predictions=[Prediction(label="Action", score=0.5)],
    )
    assert sp.timestamp_ms == 1500


def test_stream_complete():
    sc = StreamComplete(status="complete", frames_processed=240)
    assert sc.frames_processed == 240
