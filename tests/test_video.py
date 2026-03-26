import numpy as np
import pytest

from app.schemas import Clip
from app.video import decode_video, iter_clips


def test_iter_clips_yields_clips(sample_video_path):
    clips = list(iter_clips(sample_video_path, num_frames=16, stride=16))
    assert len(clips) >= 1
    for clip in clips:
        assert isinstance(clip, Clip)
        assert clip.frames.shape == (16, 240, 320, 3)
        assert clip.frames.dtype == np.uint8


def test_iter_clips_non_overlapping(sample_video_path):
    clips = list(iter_clips(sample_video_path, num_frames=16, stride=16))
    # 60 frames / 16 stride = 3 full clips + 1 partial (12 frames padded)
    assert len(clips) == 4
    assert clips[0].start_frame == 0
    assert clips[0].end_frame == 16
    assert clips[1].start_frame == 16
    assert clips[1].end_frame == 32


def test_iter_clips_overlapping(sample_video_path):
    clips = list(iter_clips(sample_video_path, num_frames=16, stride=8))
    assert len(clips) > 3
    assert clips[0].start_frame == 0
    assert clips[1].start_frame == 8


def test_iter_clips_default_stride_is_non_overlapping(sample_video_path):
    clips_default = list(iter_clips(sample_video_path, num_frames=16))
    clips_explicit = list(iter_clips(sample_video_path, num_frames=16, stride=16))
    assert len(clips_default) == len(clips_explicit)
    for a, b in zip(clips_default, clips_explicit):
        assert a.start_frame == b.start_frame
        assert a.end_frame == b.end_frame


def test_iter_clips_trailing_frames_padded(sample_video_path):
    # 60 frames, stride=16: clips at 0,16,32,48. Last clip starts at 48 with 12 real frames.
    clips = list(iter_clips(sample_video_path, num_frames=16, stride=16))
    last = clips[-1]
    assert last.start_frame == 48
    assert last.end_frame == 60  # only 12 real frames
    assert last.frames.shape == (16, 240, 320, 3)  # padded to 16
    assert last.end_frame - last.start_frame < 16


def test_iter_clips_short_video_pads(short_video_path):
    # 4 frames, num_frames=16: single partial clip, padded
    clips = list(iter_clips(short_video_path, num_frames=16))
    assert len(clips) == 1
    assert clips[0].frames.shape == (16, 240, 320, 3)
    assert clips[0].start_frame == 0
    assert clips[0].end_frame == 4


def test_iter_clips_invalid_path(tmp_path):
    with pytest.raises(Exception):
        list(iter_clips(tmp_path / "nonexistent.mp4", num_frames=16))


def test_decode_video_backward_compat(sample_video_path):
    frames = decode_video(sample_video_path, num_frames=16)
    assert isinstance(frames, np.ndarray)
    assert frames.shape == (16, 240, 320, 3)
    assert frames.dtype == np.uint8
