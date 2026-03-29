# tests/test_schemas.py
import pytest
from app.schemas import (
    Prediction,
    InferenceResponse,
    ModelMetadata,
    StreamConfig,
    StreamPrediction,
    StreamComplete,
    Clip,
    BrowserStreamConfig,
    RtspStreamConfig,
    StreamAction,
    PredictionMessage,
    SessionMessage,
    CompleteMessage,
    ErrorMessage,
)
import numpy as np


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
    assert cfg.top_k == 3
    assert cfg.num_frames == 16
    assert cfg.stride == 16


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


def test_clip_dataclass():
    frames = np.zeros((16, 240, 320, 3), dtype=np.uint8)
    clip = Clip(frames=frames, start_frame=0, end_frame=16)
    assert clip.start_frame == 0
    assert clip.end_frame == 16
    assert clip.frames.shape == (16, 240, 320, 3)


def test_clip_partial_detection():
    frames = np.zeros((16, 240, 320, 3), dtype=np.uint8)
    partial = Clip(frames=frames, start_frame=48, end_frame=55)
    assert partial.end_frame - partial.start_frame < frames.shape[0]
    full = Clip(frames=frames, start_frame=0, end_frame=16)
    assert full.end_frame - full.start_frame == frames.shape[0]


# --- New streaming schemas tests ---


def test_browser_stream_config_defaults():
    config = BrowserStreamConfig()
    assert config.top_k == 3
    assert config.num_frames == 16
    assert config.stride == 16


def test_browser_stream_config_custom():
    config = BrowserStreamConfig(top_k=5, num_frames=32, stride=8)
    assert config.top_k == 5
    assert config.num_frames == 32
    assert config.stride == 8


def test_rtsp_stream_config_valid():
    config = RtspStreamConfig(rtsp_url="rtsp://192.168.1.1:7447/stream")
    assert config.rtsp_url == "rtsp://192.168.1.1:7447/stream"
    assert config.top_k == 3


def test_rtsp_stream_config_rejects_non_rtsp():
    with pytest.raises(ValueError, match="must start with rtsp://"):
        RtspStreamConfig(rtsp_url="http://example.com/stream")


def test_stream_action_stop():
    action = StreamAction(action="stop")
    assert action.action == "stop"


def test_prediction_message():
    msg = PredictionMessage(
        clip_index=0,
        timestamp_ms=500,
        frame_range=[0, 16],
        thumbnail="base64data",
        predictions=[Prediction(label="Pushing", score=0.5)],
    )
    assert msg.type == "prediction"
    assert msg.clip_index == 0
    assert msg.thumbnail == "base64data"


def test_session_message():
    msg = SessionMessage(session_id="abc123", status="ready")
    assert msg.type == "session"


def test_complete_message():
    msg = CompleteMessage(session_id="abc123", clips_processed=12, video_ready=True)
    assert msg.type == "complete"


def test_error_message():
    msg = ErrorMessage(message="Something failed")
    assert msg.type == "error"
