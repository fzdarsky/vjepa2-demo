"""Source adapters: browser camera and RTSP stream ingestion."""

import asyncio
import io
import json
import logging
import threading

import av
import numpy as np

from app.pipeline import StreamSession

logger = logging.getLogger(__name__)


class _StreamingBuffer:
    """Thread-safe buffer that PyAV can read from while data is being written.

    Supports the file-like read interface PyAV needs, blocking on reads
    until data is available. Does not support seek (returns False for seekable).
    """

    def __init__(self):
        self._buf = bytearray()
        self._pos = 0
        self._closed = False
        self._lock = threading.Lock()
        self._has_data = threading.Event()

    def write(self, data: bytes) -> None:
        with self._lock:
            self._buf.extend(data)
            self._has_data.set()

    def read(self, size: int = -1) -> bytes:
        while True:
            with self._lock:
                available = len(self._buf) - self._pos
                if available > 0:
                    if size == -1 or size > available:
                        size = available
                    chunk = bytes(self._buf[self._pos : self._pos + size])
                    self._pos += len(chunk)
                    return chunk
                if self._closed:
                    return b""
                self._has_data.clear()
            self._has_data.wait(timeout=1.0)

    def seekable(self) -> bool:
        return False

    def close(self) -> None:
        self._closed = True
        self._has_data.set()


_MIME_TO_PYAV_FORMAT = {
    "video/mp4": "mp4",
    "video/quicktime": "mov",
    "video/x-matroska": "matroska",
    "video/webm": "matroska",
    "video/avi": "avi",
    "video/x-msvideo": "avi",
}


async def browser_source(websocket, session: StreamSession, on_clip_queued=None, media_type: str | None = None) -> None:
    """Receive binary video chunks from a browser WebSocket.

    Expects binary messages containing video data (webm chunks from
    MediaRecorder, or a complete file) and a JSON {"action": "stop"}
    to end.

    When media_type is set (file upload), buffers all data then decodes
    from a seekable BytesIO (required for formats like mp4 that need
    seeking). When media_type is None (camera stream), decodes from a
    streaming buffer concurrently as chunks arrive.

    on_clip_queued: optional async callback(clips_queued: int) called
    each time a clip is added to the inference queue.
    """
    session.status = "ingesting"
    frame_queue: asyncio.Queue = asyncio.Queue(maxsize=32)
    loop = asyncio.get_running_loop()

    async def _queue_clip(clip):
        await session.clip_queue.put(clip)
        session.clips_queued += 1
        if on_clip_queued:
            await on_clip_queued(session.clips_queued)

    async def _process_frames():
        """Process decoded frames as they arrive from the decoder thread."""
        while True:
            arr = await frame_queue.get()
            if arr is None:
                break
            session.save_frame(arr, session.buffer.total_frames)
            session._latest_frame = arr
            clips = session.buffer.add_frame(arr)
            for clip in clips:
                await _queue_clip(clip)

    def _decode_into_queue(source, fmt=None):
        """Decode video from source, pushing frames into frame_queue."""
        try:
            kwargs = {"format": fmt} if fmt else {}
            container = av.open(source, **kwargs)
            for frame in container.decode(video=0):
                arr = frame.to_ndarray(format="rgb24")
                asyncio.run_coroutine_threadsafe(
                    frame_queue.put(arr), loop
                ).result()
            container.close()
        except Exception as e:
            logger.warning("Decoder error: %s", e)
        finally:
            asyncio.run_coroutine_threadsafe(
                frame_queue.put(None), loop
            ).result()

    if media_type:
        # File upload: collect all data, then decode from seekable BytesIO
        chunks = []
        while True:
            msg = await websocket.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            if "text" in msg and msg["text"]:
                data = json.loads(msg["text"])
                if data.get("action") == "stop":
                    break
                continue
            if "bytes" in msg and msg["bytes"]:
                chunks.append(msg["bytes"])

        if not chunks:
            session.status = "processing"
            await session.clip_queue.put(None)
            return

        file_data = io.BytesIO(b"".join(chunks))
        fmt = _MIME_TO_PYAV_FORMAT.get(media_type)
        decoder_future = loop.run_in_executor(None, _decode_into_queue, file_data, fmt)
        await _process_frames()
        await asyncio.wrap_future(decoder_future)
    else:
        # Camera stream: decode concurrently via streaming buffer
        stream_buf = _StreamingBuffer()

        decoder_future = loop.run_in_executor(
            None, _decode_into_queue, stream_buf, "matroska"
        )

        async def _receive_chunks():
            try:
                while True:
                    msg = await websocket.receive()
                    if msg.get("type") == "websocket.disconnect":
                        break
                    if "text" in msg and msg["text"]:
                        data = json.loads(msg["text"])
                        if data.get("action") == "stop":
                            break
                        continue
                    if "bytes" in msg and msg["bytes"]:
                        stream_buf.write(msg["bytes"])
            finally:
                stream_buf.close()

        await asyncio.gather(_receive_chunks(), _process_frames())
        await asyncio.wrap_future(decoder_future)

    # Flush remaining frames
    for clip in session.buffer.flush():
        await _queue_clip(clip)

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
