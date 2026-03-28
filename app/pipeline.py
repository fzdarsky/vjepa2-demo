"""Shared streaming pipeline: frame buffering, clip assembly, session management."""

import numpy as np

from app.schemas import Clip


class FrameBuffer:
    """Accumulates frames and yields clips using a sliding window.

    When num_frames frames are available, yields a Clip and advances
    by stride frames. Trims consumed frames to bound memory.
    """

    def __init__(self, num_frames: int = 16, stride: int = 8):
        self.num_frames = num_frames
        self.stride = stride
        self._frames: list[np.ndarray] = []
        self._next_clip_start: int = 0
        self.total_frames: int = 0

    def add_frame(self, frame: np.ndarray) -> list[Clip]:
        """Add a frame and return any complete clips."""
        self._frames.append(frame)
        self.total_frames += 1

        clips = []
        while len(self._frames) >= self._next_clip_start + self.num_frames:
            clip_frames = self._frames[
                self._next_clip_start : self._next_clip_start + self.num_frames
            ]
            global_start = self.total_frames - len(self._frames) + self._next_clip_start
            clips.append(
                Clip(
                    frames=np.stack(clip_frames),
                    start_frame=global_start,
                    end_frame=global_start + self.num_frames,
                )
            )
            self._next_clip_start += self.stride

        # Trim frames that can't be part of any future clip
        if self._next_clip_start > 0:
            self._frames = self._frames[self._next_clip_start :]
            self._next_clip_start = 0

        return clips

    def flush(self) -> list[Clip]:
        """Flush remaining frames as a padded clip."""
        if not self._frames:
            return []

        remaining = self._frames[self._next_clip_start :]
        if not remaining:
            return []

        last_frame = remaining[-1]
        pad_count = self.num_frames - len(remaining)
        padded = remaining + [last_frame] * pad_count

        global_start = self.total_frames - len(self._frames) + self._next_clip_start
        clip = Clip(
            frames=np.stack(padded),
            start_frame=global_start,
            end_frame=self.total_frames,
        )
        self._frames.clear()
        self._next_clip_start = 0
        return [clip]
