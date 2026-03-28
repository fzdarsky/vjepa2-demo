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
    Decodes frames with PyAV and pushes them into the session buffer.
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

    # Decode accumulated video data
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
    except av.error.InvalidDataError:
        pass  # Handled by caller via error message

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

    Runs PyAV frame iteration in a thread since it's blocking.
    Stops when stop_event is set or the stream ends.
    """
    session.status = "ingesting"

    def _read_frames():
        container = av.open(url)
        for frame in container.decode(video=0):
            if stop_event.is_set():
                break
            arr = frame.to_ndarray(format="rgb24")
            yield arr
        container.close()

    # Run blocking I/O in thread, process frames in event loop
    frames_iter = await asyncio.to_thread(lambda: list(_read_frames()))

    for arr in frames_iter:
        session.save_frame(arr, session.buffer.total_frames)
        session._latest_frame = arr
        clips = session.buffer.add_frame(arr)
        for clip in clips:
            await session.clip_queue.put(clip)

    # Flush remaining frames
    for clip in session.buffer.flush():
        await session.clip_queue.put(clip)

    session.status = "processing"
    await session.clip_queue.put(None)  # sentinel
