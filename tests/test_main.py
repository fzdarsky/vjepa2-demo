# tests/test_main.py
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.schemas import Prediction


@pytest.fixture
def mock_vjepa2_model():
    """Mock VJepa2Model for API tests."""
    with patch("app.main.VJepa2Model") as MockClass:
        instance = MagicMock()
        instance.id2label = {i: f"Action {i}" for i in range(174)}
        MockClass.return_value = instance
        yield instance


@pytest.fixture
def client(mock_vjepa2_model):
    from app.main import app
    with TestClient(app) as c:
        yield c


def test_liveness(client):
    resp = client.get("/v2/health/live")
    assert resp.status_code == 200


def test_readiness_when_model_loaded(client):
    resp = client.get("/v2/health/ready")
    assert resp.status_code == 200


def test_model_metadata(client):
    resp = client.get("/v2/models/vjepa2")
    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "vjepa2"
    assert "versions" in data
    assert data["platform"] == "pytorch"


import io

import av
import numpy as np


def _make_video_bytes(num_frames: int = 60) -> bytes:
    """Create a minimal in-memory MP4 video."""
    buf = io.BytesIO()
    container = av.open(buf, mode="w", format="mp4")
    stream = container.add_stream("libx264", rate=30)
    stream.width = 320
    stream.height = 240
    stream.pix_fmt = "yuv420p"
    for i in range(num_frames):
        frame = av.VideoFrame.from_ndarray(
            np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8),
            format="rgb24",
        )
        frame.pts = i
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    buf.seek(0)
    return buf.read()


def test_infer_returns_predictions(client, mock_vjepa2_model):
    mock_vjepa2_model.predict.return_value = [
        Prediction(label="Action 1", score=0.9),
        Prediction(label="Action 2", score=0.1),
    ]
    video_bytes = _make_video_bytes()
    resp = client.post(
        "/v2/models/vjepa2/infer",
        files={"file": ("test.mp4", video_bytes, "video/mp4")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["model_name"] == "vjepa2"
    assert len(data["outputs"]) == 1
    assert data["outputs"][0]["data"][0]["label"] == "Action 1"


def test_infer_custom_top_k(client, mock_vjepa2_model):
    mock_vjepa2_model.predict.return_value = [
        Prediction(label=f"Action {i}", score=round(0.9 - i * 0.1, 1))
        for i in range(3)
    ]
    video_bytes = _make_video_bytes()
    resp = client.post(
        "/v2/models/vjepa2/infer",
        files={"file": ("test.mp4", video_bytes, "video/mp4")},
        data={"top_k": "3"},
    )
    assert resp.status_code == 200
    assert len(resp.json()["outputs"][0]["data"]) == 3


def test_infer_video_too_short(client, mock_vjepa2_model):
    video_bytes = _make_video_bytes(num_frames=4)
    resp = client.post(
        "/v2/models/vjepa2/infer",
        files={"file": ("short.mp4", video_bytes, "video/mp4")},
    )
    assert resp.status_code == 400
    assert "frames" in resp.json()["error"].lower()


def test_websocket_stream(client, mock_vjepa2_model, sample_video_path):
    # Patch allowed_source_dir to accept tmp_path
    import app.main
    original = app.main.CONFIG["server"]["allowed_source_dir"]
    app.main.CONFIG["server"]["allowed_source_dir"] = str(sample_video_path.parent)
    try:
        mock_vjepa2_model.predict.return_value = [
            Prediction(label="Action 1", score=0.9),
        ]
        with client.websocket_connect("/v2/models/vjepa2/stream") as ws:
            ws.send_json({
                "source": str(sample_video_path),
                "top_k": 5,
                "num_frames": 8,
                "stride": 8,
            })
            # Should receive at least one prediction message
            msg = ws.receive_json()
            assert "predictions" in msg or "status" in msg

            # Read until completion
            messages = [msg]
            while True:
                msg = ws.receive_json()
                messages.append(msg)
                if msg.get("status") == "complete":
                    break
            assert messages[-1]["status"] == "complete"
    finally:
        app.main.CONFIG["server"]["allowed_source_dir"] = original


def test_websocket_invalid_config(client, mock_vjepa2_model):
    with client.websocket_connect("/v2/models/vjepa2/stream") as ws:
        ws.send_json({"source": "/samples/../etc/passwd"})
        msg = ws.receive_json()
        assert "error" in msg


def test_websocket_file_not_found(client, mock_vjepa2_model):
    with client.websocket_connect("/v2/models/vjepa2/stream") as ws:
        ws.send_json({"source": "/samples/nonexistent.mp4"})
        msg = ws.receive_json()
        assert "error" in msg
