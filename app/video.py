from pathlib import Path
from typing import Iterator

import av
import numpy as np

from app.schemas import Clip
from app.telemetry import get_tracer


def iter_clips(
    source: str | Path,
    num_frames: int = 16,
    stride: int | None = None,
) -> Iterator[Clip]:
    """Yield clips from a video source with streaming decode.

    Each Clip contains num_frames frames. When stride is None,
    defaults to num_frames (non-overlapping). Trailing frames
    are padded by repeating the last frame.

    Clips are yielded as soon as enough frames are decoded,
    allowing inference to overlap with decoding.
    """
    if stride is None:
        stride = num_frames

    tracer = get_tracer()

    # Container initialization (file parse or RTSP connect)
    with tracer.start_as_current_span("input_open") as span:
        container = av.open(str(source))
        # Extract video stream metadata
        stream = container.streams.video[0]
        span.set_attribute("input.codec", stream.codec_context.name)
        span.set_attribute("input.width", stream.width)
        span.set_attribute("input.height", stream.height)
        if stream.frames:
            span.set_attribute("input.total_frames", stream.frames)
        # Detect source type from path
        source_str = str(source)
        if source_str.startswith("rtsp://"):
            span.set_attribute("input.source_type", "rtsp")
        elif source_str.startswith(("http://", "https://")):
            span.set_attribute("input.source_type", "http")
        else:
            span.set_attribute("input.source_type", "file")

    buffer: list[np.ndarray] = []
    frame_index = 0
    next_clip_start = 0
    decode_span = None

    for frame in container.decode(video=0):
        # Start decode span when we begin accumulating for a new clip
        if decode_span is None:
            decode_span = tracer.start_span("input_decode")
            decode_span.set_attribute("clip.start_frame", next_clip_start)

        buffer.append(frame.to_ndarray(format="rgb24"))
        frame_index += 1

        # Yield clips as soon as we have enough frames
        while len(buffer) >= next_clip_start + num_frames:
            # End decode span for this clip
            if decode_span is not None:
                decode_span.set_attribute("clip.end_frame", next_clip_start + num_frames)
                decode_span.end()
                decode_span = None

            clip_frames = buffer[next_clip_start : next_clip_start + num_frames]
            yield Clip(
                frames=np.stack(clip_frames),
                start_frame=next_clip_start,
                end_frame=next_clip_start + num_frames,
            )
            next_clip_start += stride

            # Start new decode span if more clips expected
            if next_clip_start < frame_index:
                decode_span = tracer.start_span("input_decode")
                decode_span.set_attribute("clip.start_frame", next_clip_start)

    container.close()

    # Handle trailing frames: pad with last frame
    total_frames = frame_index
    if next_clip_start < total_frames and buffer:
        # Reuse existing decode span or create new one
        if decode_span is None:
            decode_span = tracer.start_span("input_decode")
            decode_span.set_attribute("clip.start_frame", next_clip_start)

        decode_span.set_attribute("clip.end_frame", total_frames)
        decode_span.set_attribute("clip.padded", True)
        decode_span.end()
        decode_span = None

        remaining = buffer[next_clip_start:]
        last_frame = remaining[-1]
        pad_count = num_frames - len(remaining)
        padded = remaining + [last_frame] * pad_count
        yield Clip(
            frames=np.stack(padded),
            start_frame=next_clip_start,
            end_frame=total_frames,
        )
    elif decode_span is not None:
        # No trailing frames but span was started - end it
        decode_span.end()
        decode_span = None


def decode_video(source: str | Path, num_frames: int) -> np.ndarray:
    """Convenience: return frames from first clip.
    Backward-compatible with milestone 1 callers."""
    # input_decode span is now inside iter_clips
    clip = next(iter_clips(source, num_frames, stride=num_frames))
    return clip.frames
