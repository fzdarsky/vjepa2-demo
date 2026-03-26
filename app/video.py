from pathlib import Path
from typing import Iterator

import av
import numpy as np

from app.schemas import Clip


def iter_clips(
    source: str | Path,
    num_frames: int = 16,
    stride: int | None = None,
) -> Iterator[Clip]:
    """Yield clips from a video source.

    Each Clip contains num_frames frames. When stride is None,
    defaults to num_frames (non-overlapping). Trailing frames
    are padded by repeating the last frame.
    """
    if stride is None:
        stride = num_frames

    container = av.open(str(source))
    buffer: list[np.ndarray] = []
    frame_index = 0
    next_clip_start = 0

    for frame in container.decode(video=0):
        buffer.append(frame.to_ndarray(format="rgb24"))
        frame_index += 1

        while len(buffer) >= next_clip_start + num_frames:
            clip_frames = buffer[next_clip_start:next_clip_start + num_frames]
            yield Clip(
                frames=np.stack(clip_frames),
                start_frame=next_clip_start,
                end_frame=next_clip_start + num_frames,
            )
            next_clip_start += stride

    container.close()

    # Handle trailing frames: pad with last frame
    total_frames = frame_index
    if next_clip_start < total_frames and buffer:
        remaining = buffer[next_clip_start:]
        last_frame = remaining[-1]
        pad_count = num_frames - len(remaining)
        padded = remaining + [last_frame] * pad_count
        yield Clip(
            frames=np.stack(padded),
            start_frame=next_clip_start,
            end_frame=total_frames,
        )


def decode_video(source: str | Path, num_frames: int) -> np.ndarray:
    """Convenience: return frames from first clip.
    Backward-compatible with milestone 1 callers."""
    clip = next(iter_clips(source, num_frames, stride=num_frames))
    return clip.frames
