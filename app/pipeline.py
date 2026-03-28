"""Shared streaming pipeline: frame buffering, clip assembly, session management."""

import asyncio
import base64
import io
import json
import time
import uuid
from pathlib import Path
from typing import Callable, Awaitable

import numpy as np
from PIL import Image

from app.schemas import Clip
from app.telemetry import get_meter, get_tracer


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


class StreamSession:
    """Manages state for a single streaming session."""

    def __init__(
        self,
        session_id: str | None = None,
        num_frames: int = 16,
        stride: int = 8,
        top_k: int = 5,
        base_dir: str = "/tmp/vjepa2-sessions",
    ):
        self.session_id = session_id or uuid.uuid4().hex[:12]
        self.buffer = FrameBuffer(num_frames=num_frames, stride=stride)
        self.clip_queue: asyncio.Queue = asyncio.Queue()
        self.top_k = top_k
        self.status = "created"
        self.results: list[dict] = []
        self.created_at = time.monotonic()
        self._clip_index = 0
        self._frame_index = 0
        self.clips_queued = 0
        self._latest_frame: np.ndarray | None = None

        # Create temp directory
        self.temp_dir = Path(base_dir) / self.session_id
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        (self.temp_dir / "frames").mkdir(exist_ok=True)

        meter = get_meter()
        self._active_gauge = meter.create_up_down_counter(
            "vjepa2_active_sessions",
            description="Number of active streaming sessions",
        )
        self._active_gauge.add(1)
        self._duration_hist = meter.create_histogram(
            "vjepa2_session_duration_seconds",
            description="Total session duration",
            unit="s",
        )
        self._queue_depth = meter.create_gauge(
            "vjepa2_queue_depth",
            description="Current clip queue depth",
        )

    def save_frame(self, frame: np.ndarray, index: int) -> None:
        """Save a frame to disk as JPEG for later video annotation."""
        img = Image.fromarray(frame)
        path = self.temp_dir / "frames" / f"{index:06d}.jpg"
        img.save(path, "JPEG", quality=85)

    def append_result(self, result: dict) -> None:
        """Append a prediction result and persist to disk."""
        self.results.append(result)
        with open(self.temp_dir / "results.jsonl", "a") as f:
            f.write(json.dumps(result) + "\n")

    def complete(self) -> None:
        """Mark session as complete and record metrics."""
        self.status = "complete"
        self._active_gauge.add(-1)
        self._duration_hist.record(time.monotonic() - self.created_at)

    def cleanup(self) -> None:
        """Remove session temp directory."""
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)


def _frame_to_thumbnail(frame: np.ndarray, width: int = 160) -> str:
    """Encode a frame as a base64 JPEG thumbnail."""
    img = Image.fromarray(frame)
    aspect = img.height / img.width
    img = img.resize((width, int(width * aspect)))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=75)
    return base64.b64encode(buf.getvalue()).decode("ascii")


async def inference_worker(
    session: StreamSession,
    model,
    on_result: Callable[[dict], Awaitable[None]],
    source_fps: float = 30.0,
    thumbnail_width: int = 160,
) -> None:
    """Pull clips from session queue and run inference.

    Stops when it receives None (sentinel) from the queue.
    """
    meter = get_meter()
    tracer = get_tracer()
    frames_counter = meter.create_counter(
        "vjepa2_frames_processed_total",
        description="Total video frames decoded",
    )
    clips_counter = meter.create_counter(
        "vjepa2_clips_processed_total",
        description="Total clips inferred",
    )
    request_duration = meter.create_histogram(
        "vjepa2_request_duration_seconds",
        description="Per-clip inference duration",
        unit="s",
    )

    with tracer.start_as_current_span("stream_inference") as stream_span:
        stream_span.set_attribute("session.id", session.session_id)

        while True:
            session._queue_depth.set(session.clip_queue.qsize())
            clip = await session.clip_queue.get()
            if clip is None:
                session._queue_depth.set(0)
                break

            clip_start = time.monotonic()
            with tracer.start_as_current_span("clip_inference") as clip_span:
                clip_span.set_attribute("clip.index", session._clip_index)
                clip_span.set_attribute("clip.start_frame", clip.start_frame)
                clip_span.set_attribute("clip.end_frame", clip.end_frame)

                predictions = await asyncio.to_thread(
                    model.predict,
                    clip.frames,
                    top_k=session.top_k,
                    stride=session.buffer.stride,
                    source_fps=source_fps,
                )

            # Generate thumbnail from middle frame
            mid = len(clip.frames) // 2
            thumbnail = _frame_to_thumbnail(clip.frames[mid], width=thumbnail_width)

            result = {
                "type": "prediction",
                "clip_index": session._clip_index,
                "timestamp_ms": int(clip.start_frame * 1000 / source_fps),
                "frame_range": [clip.start_frame, clip.end_frame],
                "thumbnail": thumbnail,
                "predictions": [p.model_dump() for p in predictions],
            }
            session._clip_index += 1
            session.append_result(result)
            frames_counter.add(clip.end_frame - clip.start_frame)
            clips_counter.add(1)
            request_duration.record(time.monotonic() - clip_start)

            await on_result(result)
