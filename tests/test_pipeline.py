import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from app.pipeline import FrameBuffer, StreamSession, inference_worker
from app.schemas import Prediction, Clip


def _make_frame(value: int = 0) -> np.ndarray:
    """Create a 4x4x3 synthetic frame filled with a single value."""
    return np.full((4, 4, 3), value, dtype=np.uint8)


def test_framebuffer_yields_clip_when_full():
    buf = FrameBuffer(num_frames=4, stride=4)
    clips = []
    for i in range(4):
        clips.extend(buf.add_frame(_make_frame(i)))
    assert len(clips) == 1
    assert clips[0].frames.shape == (4, 4, 4, 3)
    assert clips[0].start_frame == 0
    assert clips[0].end_frame == 4


def test_framebuffer_stride_overlap():
    buf = FrameBuffer(num_frames=4, stride=2)
    clips = []
    for i in range(6):
        clips.extend(buf.add_frame(_make_frame(i)))
    # After 4 frames: clip [0,4). After 6 frames: clip [2,6).
    assert len(clips) == 2
    assert clips[0].start_frame == 0
    assert clips[1].start_frame == 2


def test_framebuffer_trims_consumed_frames():
    buf = FrameBuffer(num_frames=4, stride=4)
    for i in range(8):
        buf.add_frame(_make_frame(i))
    # After yielding 2 clips, internal buffer should be trimmed
    # Buffer should not grow unbounded
    assert len(buf._frames) <= 4


def test_framebuffer_flush_pads_remaining():
    buf = FrameBuffer(num_frames=4, stride=4)
    for i in range(6):
        buf.add_frame(_make_frame(i))
    # 4 frames yielded one clip. 2 remaining.
    clips = buf.flush()
    assert len(clips) == 1
    assert clips[0].frames.shape == (4, 4, 4, 3)
    assert clips[0].start_frame == 4
    # end_frame reflects actual frames, not padded length
    assert clips[0].end_frame == 6


def test_framebuffer_flush_empty():
    buf = FrameBuffer(num_frames=4, stride=4)
    clips = buf.flush()
    assert len(clips) == 0


def test_framebuffer_no_clips_before_full():
    buf = FrameBuffer(num_frames=4, stride=4)
    clips = []
    for i in range(3):
        clips.extend(buf.add_frame(_make_frame(i)))
    assert len(clips) == 0


def test_framebuffer_frame_count():
    buf = FrameBuffer(num_frames=4, stride=4)
    for i in range(7):
        buf.add_frame(_make_frame(i))
    assert buf.total_frames == 7


def test_stream_session_creation():
    session = StreamSession(
        session_id="test-123",
        num_frames=16,
        stride=8,
        top_k=5,
    )
    assert session.session_id == "test-123"
    assert session.status == "created"
    assert session.temp_dir.exists()
    assert (session.temp_dir / "frames").exists()
    session.cleanup()


def test_stream_session_status_transitions():
    session = StreamSession(
        session_id="test-456",
        num_frames=16,
        stride=8,
        top_k=5,
    )
    assert session.status == "created"
    session.status = "ingesting"
    assert session.status == "ingesting"
    session.status = "processing"
    assert session.status == "processing"
    session.status = "complete"
    assert session.status == "complete"
    session.cleanup()


def test_stream_session_save_frame(tmp_path):
    session = StreamSession(
        session_id="test-789",
        num_frames=4,
        stride=4,
        top_k=5,
        base_dir=str(tmp_path),
    )
    frame = np.full((4, 4, 3), 128, dtype=np.uint8)
    session.save_frame(frame, 0)
    saved = list((session.temp_dir / "frames").glob("*.jpg"))
    assert len(saved) == 1
    session.cleanup()


def test_stream_session_append_result(tmp_path):
    session = StreamSession(
        session_id="test-res",
        num_frames=4,
        stride=4,
        top_k=5,
        base_dir=str(tmp_path),
    )
    result = {
        "clip_index": 0,
        "timestamp_ms": 0,
        "predictions": [{"label": "test", "score": 0.9}],
    }
    session.append_result(result)
    assert len(session.results) == 1

    results_file = session.temp_dir / "results.jsonl"
    assert results_file.exists()
    line = results_file.read_text().strip()
    assert json.loads(line)["clip_index"] == 0
    session.cleanup()


@pytest.mark.asyncio
async def test_inference_worker_processes_clips(tmp_path):
    session = StreamSession(
        session_id="test-worker",
        num_frames=4,
        stride=4,
        top_k=2,
        base_dir=str(tmp_path),
    )

    # Mock model
    mock_model = MagicMock()
    mock_model.predict.return_value = [
        Prediction(label="Pushing", score=0.8),
        Prediction(label="Pulling", score=0.2),
    ]

    # Add a clip to the queue
    clip = Clip(
        frames=np.zeros((4, 4, 4, 3), dtype=np.uint8),
        start_frame=0,
        end_frame=4,
    )
    await session.clip_queue.put(clip)
    await session.clip_queue.put(None)  # sentinel to stop worker

    results = []
    async def on_result(msg):
        results.append(msg)

    await inference_worker(session, mock_model, on_result)

    assert len(results) == 1
    assert results[0]["type"] == "prediction"
    assert results[0]["clip_index"] == 0
    assert len(results[0]["predictions"]) == 2
    assert "thumbnail" in results[0]
    mock_model.predict.assert_called_once()
    session.cleanup()
