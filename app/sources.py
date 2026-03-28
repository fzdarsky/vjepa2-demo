"""Source adapters: browser camera and RTSP stream ingestion."""

import asyncio
import io
import json

import av
import numpy as np

from app.pipeline import StreamSession


async def browser_source(websocket, session: StreamSession) -> None:
    """Receive binary video chunks from a browser WebSocket.

    Expects binary messages containing video data (webm/mp4 chunks
    from MediaRecorder) and a JSON {"action": "stop"} to end.
    Accumulates chunks then decodes frames with PyAV (MediaRecorder
    chunks are not independently decodable) and pushes them into the
    session buffer.
    """
    session.status = "ingesting"
    accumulated = bytearray()

    while True:
        msg = await websocket.receive()

        if "text" in msg and msg["text"]:
            data = json.loads(msg["text"])
            if data.get("action") == "stop":
                break
            continue

        if "bytes" in msg and msg["bytes"]:
            accumulated.extend(msg["bytes"])

    # Decode accumulated video data — MediaRecorder chunks form a single
    # container so they must be accumulated before decoding.
    if not accumulated:
        session.status = "processing"
        await session.clip_queue.put(None)
        return

    buf = io.BytesIO(bytes(accumulated))
    try:
        container = av.open(buf)
        for frame in container.decode(video=0):
            arr = frame.to_ndarray(format="rgb24")
            session.save_frame(arr, session.buffer.total_frames)
            session._latest_frame = arr
            clips = session.buffer.add_frame(arr)
            for clip in clips:
                await session.clip_queue.put(clip)
        container.close()
    except av.error.InvalidDataError as e:
        raise ValueError(f"Could not decode video data: {e}") from e

    # Flush remaining frames
    for clip in session.buffer.flush():
        await session.clip_queue.put(clip)

    session.status = "processing"
    await session.clip_queue.put(None)  # sentinel


async def rtsp_source(
    url: str,
    session: StreamSession,
    stop_event: asyncio.Event,
) -> None:
    """Pull frames from an RTSP URL using PyAV.

    Runs PyAV frame decode in a thread since it's blocking.
    Uses a queue to stream frames to the event loop incrementally,
    avoiding loading all frames into memory at once.
    Stops when stop_event is set or the stream ends.
    """
    session.status = "ingesting"
    frame_queue: asyncio.Queue = asyncio.Queue(maxsize=32)
    loop = asyncio.get_running_loop()

    def _read_frames():
        container = av.open(url)
        try:
            for frame in container.decode(video=0):
                if stop_event.is_set():
                    break
                arr = frame.to_ndarray(format="rgb24")
                asyncio.run_coroutine_threadsafe(
                    frame_queue.put(arr), loop
                ).result()
        finally:
            container.close()
            asyncio.run_coroutine_threadsafe(
                frame_queue.put(None), loop
            ).result()

    reader_future = loop.run_in_executor(None, _read_frames)

    while True:
        arr = await frame_queue.get()
        if arr is None:
            break
        session.save_frame(arr, session.buffer.total_frames)
        session._latest_frame = arr
        clips = session.buffer.add_frame(arr)
        for clip in clips:
            await session.clip_queue.put(clip)

    await asyncio.wrap_future(reader_future)

    # Flush remaining frames
    for clip in session.buffer.flush():
        await session.clip_queue.put(clip)

    session.status = "processing"
    await session.clip_queue.put(None)  # sentinel
