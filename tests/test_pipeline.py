import numpy as np
import pytest

from app.pipeline import FrameBuffer


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
