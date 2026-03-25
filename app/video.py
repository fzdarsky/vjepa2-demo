# app/video.py
from pathlib import Path
from typing import Iterator

import av
import numpy as np


def decode_video(source: str | Path, num_frames: int) -> np.ndarray:
    """Decode a video file and return uniformly sampled frames as a numpy array.

    Returns: np.ndarray of shape (num_frames, H, W, 3), dtype uint8.
    Raises ValueError if video has fewer than num_frames frames.
    """
    container = av.open(str(source))

    # Decode all frames
    all_frames = []
    for frame in container.decode(video=0):
        all_frames.append(frame.to_ndarray(format="rgb24"))
    container.close()

    total = len(all_frames)
    if total < num_frames:
        raise ValueError(
            f"Video has {total} frames, need at least {num_frames}"
        )

    # Uniformly sample num_frames indices
    indices = np.linspace(0, total - 1, num_frames, dtype=int)
    sampled = np.stack([all_frames[i] for i in indices])
    return sampled


def stream_frames(
    source: str | Path, num_frames: int, stride: int
) -> Iterator[np.ndarray]:
    """Yield sliding window frame arrays from a video source.

    Each array has shape (num_frames, H, W, 3), dtype uint8.
    Windows advance by `stride` frames with possible overlap.
    """
    container = av.open(str(source))
    buffer: list[np.ndarray] = []
    frames_since_yield = 0

    for frame in container.decode(video=0):
        arr = frame.to_ndarray(format="rgb24")
        buffer.append(arr)

        if len(buffer) >= num_frames:
            frames_since_yield += 1
            if frames_since_yield >= stride or len(buffer) == num_frames:
                window = np.stack(buffer[-num_frames:])
                yield window
                buffer = buffer[-num_frames:]
                frames_since_yield = 0

    container.close()
