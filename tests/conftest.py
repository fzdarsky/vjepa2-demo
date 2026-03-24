# tests/conftest.py
from pathlib import Path

import av
import numpy as np
import pytest


@pytest.fixture
def sample_video_path(tmp_path: Path) -> Path:
    """Create a minimal 2-second, 30fps synthetic video (60 frames)."""
    path = tmp_path / "test_video.mp4"
    container = av.open(str(path), mode="w")
    stream = container.add_stream("libx264", rate=30)
    stream.width = 320
    stream.height = 240
    stream.pix_fmt = "yuv420p"
    for i in range(60):
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
    return path


@pytest.fixture
def short_video_path(tmp_path: Path) -> Path:
    """Create a video with only 4 frames (too short for default num_frames=16)."""
    path = tmp_path / "short_video.mp4"
    container = av.open(str(path), mode="w")
    stream = container.add_stream("libx264", rate=30)
    stream.width = 320
    stream.height = 240
    stream.pix_fmt = "yuv420p"
    for i in range(4):
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
    return path
