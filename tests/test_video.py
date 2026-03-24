# tests/test_video.py
import numpy as np
import pytest

from app.video import decode_video, stream_frames


def test_decode_video_returns_numpy_array(sample_video_path):
    frames = decode_video(sample_video_path, num_frames=16)
    assert isinstance(frames, np.ndarray)
    assert frames.shape == (16, 240, 320, 3)
    assert frames.dtype == np.uint8


def test_decode_video_correct_num_frames(sample_video_path):
    frames = decode_video(sample_video_path, num_frames=8)
    assert frames.shape[0] == 8


def test_decode_video_too_few_frames(short_video_path):
    with pytest.raises(ValueError, match="need at least"):
        decode_video(short_video_path, num_frames=16)


def test_decode_video_invalid_path(tmp_path):
    with pytest.raises(Exception):
        decode_video(tmp_path / "nonexistent.mp4", num_frames=16)


def test_stream_frames_yields_windows(sample_video_path):
    windows = list(stream_frames(sample_video_path, num_frames=8, stride=8))
    assert len(windows) >= 1
    for window in windows:
        assert isinstance(window, np.ndarray)
        assert window.shape == (8, 240, 320, 3)


def test_stream_frames_with_overlap(sample_video_path):
    windows = list(stream_frames(sample_video_path, num_frames=16, stride=8))
    # 60 frames, stride 8: should get multiple windows
    assert len(windows) >= 2
