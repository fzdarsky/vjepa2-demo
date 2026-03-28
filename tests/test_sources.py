import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import av
import numpy as np
import pytest

from app.pipeline import StreamSession
from app.sources import browser_source, rtsp_source


@pytest.fixture
def session(tmp_path):
    s = StreamSession(
        session_id="test-src",
        num_frames=4,
        stride=4,
        top_k=5,
        base_dir=str(tmp_path),
    )
    yield s
    s.cleanup()


def _make_video_bytes(num_frames: int = 8) -> bytes:
    """Create a minimal in-memory MP4 as bytes."""
    import io
    buf = io.BytesIO()
    container = av.open(buf, mode="w", format="mp4")
    stream = container.add_stream("libx264", rate=30)
    stream.width = 64
    stream.height = 64
    stream.pix_fmt = "yuv420p"
    for i in range(num_frames):
        frame = av.VideoFrame.from_ndarray(
            np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8),
            format="rgb24",
        )
        frame.pts = i
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    buf.seek(0)
    return buf.getvalue()


@pytest.mark.asyncio
async def test_browser_source_decodes_chunks(session):
    video_bytes = _make_video_bytes(8)
    # Mock websocket that delivers the video as one chunk then stops
    ws = AsyncMock()
    ws.receive = AsyncMock(
        side_effect=[
            {"type": "websocket.receive", "bytes": video_bytes},
            {"type": "websocket.receive", "text": '{"action": "stop"}'},
        ]
    )

    await browser_source(ws, session)

    assert session.buffer.total_frames == 8
    assert session.status == "processing"


@pytest.mark.asyncio
async def test_rtsp_source_reads_frames(session, tmp_path):
    # Create a test video file to simulate RTSP (PyAV opens file paths too)
    video_path = tmp_path / "fake_rtsp.mp4"
    container = av.open(str(video_path), mode="w")
    stream = container.add_stream("libx264", rate=30)
    stream.width = 64
    stream.height = 64
    stream.pix_fmt = "yuv420p"
    for i in range(8):
        frame = av.VideoFrame.from_ndarray(
            np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8),
            format="rgb24",
        )
        frame.pts = i
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()

    # Use file path instead of real RTSP URL
    stop_event = asyncio.Event()
    await rtsp_source(str(video_path), session, stop_event)

    assert session.buffer.total_frames == 8
    assert session.status == "processing"
