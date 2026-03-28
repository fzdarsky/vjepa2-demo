import numpy as np
import pytest
from pathlib import Path

from app.annotate import overlay_predictions, encode_video
from app.schemas import Prediction


def test_overlay_predictions_returns_image():
    frame = np.full((240, 320, 3), 200, dtype=np.uint8)
    predictions = [
        Prediction(label="Pushing something", score=0.8),
        Prediction(label="Moving something", score=0.15),
    ]
    result = overlay_predictions(frame, predictions)
    assert result.shape == frame.shape
    assert result.dtype == np.uint8
    # The overlay should modify the frame (not return identical)
    assert not np.array_equal(result, frame)


def test_overlay_predictions_empty_list():
    frame = np.full((240, 320, 3), 200, dtype=np.uint8)
    result = overlay_predictions(frame, [])
    # With no predictions, frame should still be valid
    assert result.shape == frame.shape


def test_encode_video_creates_file(tmp_path):
    frames = [np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8) for _ in range(10)]
    output_path = tmp_path / "output.mp4"
    encode_video(frames, output_path, fps=10)
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_encode_video_roundtrip(tmp_path):
    import av
    frames = [np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8) for _ in range(5)]
    output_path = tmp_path / "roundtrip.mp4"
    encode_video(frames, output_path, fps=10)

    # Read back and verify frame count
    container = av.open(str(output_path))
    decoded = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
    container.close()
    assert len(decoded) == 5
