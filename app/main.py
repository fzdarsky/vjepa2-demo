# app/main.py
import asyncio
import logging
import os
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

logger = logging.getLogger(__name__)

import numpy as np
import yaml
from fastapi import FastAPI, File, Form, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from app.model import VJepa2Model, select_device
from app.pipeline import StreamSession, inference_worker
from app.schemas import (
    BrowserStreamConfig,
    CompleteMessage,
    ErrorMessage,
    ModelMetadata,
    Prediction,
    RtspStreamConfig,
    SessionMessage,
    StreamAction,
    StreamComplete,
    StreamConfig,
    StreamPrediction,
)
from app.sources import browser_source, rtsp_source
from app.telemetry import get_meter, get_tracer, init_telemetry, instrument_fastapi
from app.video import iter_clips

# Load config
_config_path = os.environ.get("CONFIG_PATH", "configs/model_config.yaml")
with open(_config_path) as f:
    CONFIG = yaml.safe_load(f)

_model: VJepa2Model | None = None
_sessions: dict[str, StreamSession] = {}


async def _load_model_background():
    """Load model in background thread so the server can serve UI immediately."""
    global _model
    device = select_device(os.environ.get("DEVICE"))

    model_path = os.environ.get(
        "MODEL_PATH",
        CONFIG["model"].get("model_path", CONFIG["model"]["hf_model_id"]),
    )

    meter = get_meter()
    tracer = get_tracer()

    model_load_gauge = meter.create_gauge(
        "vjepa2_model_load_duration_seconds",
        description="Time to load model at startup",
        unit="s",
    )

    # Standalone span for model loading (server lifecycle event)
    with tracer.start_as_current_span("init_model_load") as span:
        span.set_attribute("model.path", model_path)
        span.set_attribute("model.device", device)

        load_start = time.monotonic()
        _model = await asyncio.to_thread(
            VJepa2Model, model_path=model_path, device=device
        )
        load_duration = time.monotonic() - load_start

        span.set_attribute("model.load_duration_s", round(load_duration, 3))
        model_load_gauge.set(load_duration)


@asynccontextmanager
async def lifespan(app: FastAPI):
    device = select_device(os.environ.get("DEVICE"))

    init_telemetry(
        service_name="vjepa2-server",
        device=device,
        otlp_endpoint=os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"),
    )

    instrument_fastapi(app)

    load_task = asyncio.create_task(_load_model_background())
    cleanup_task = asyncio.create_task(_session_cleanup_loop())
    yield
    load_task.cancel()
    cleanup_task.cancel()
    global _model
    _model = None


app = FastAPI(title="V-JEPA2 Inference", lifespan=lifespan)


@app.get("/v2/health/live")
def health_live():
    return {"status": "alive"}


@app.get("/v2/health/ready")
def health_ready():
    if _model is None:
        return JSONResponse(
            {"status": "loading", "model": CONFIG["model"]["hf_model_id"].split("/")[-1]},
            status_code=503,
        )
    return {
        "status": "ready",
        "model": CONFIG["model"]["hf_model_id"].split("/")[-1],
        "device": _model.device,
    }


@app.get("/v2/models/vjepa2")
def model_metadata():
    return ModelMetadata(
        name=CONFIG["model"]["name"],
        versions=["vit-l-16-ssv2"],
        platform="pytorch",
    )


@app.post("/v2/models/vjepa2/infer")
async def infer(
    file: UploadFile = File(...),
    top_k: int = Form(default=CONFIG["inference"]["default_top_k"]),
    num_frames: int = Form(default=CONFIG["inference"]["num_frames"]),
    stride: int | None = Form(default=None),
    obs_timestamp_ms: int | None = Form(default=None),
):
    if _model is None:
        return JSONResponse({"error": "Model not ready"}, status_code=503)

    meter = get_meter()
    request_duration = meter.create_histogram(
        "vjepa2_request_duration_seconds",
        description="End-to-end API call duration",
        unit="s",
    )
    requests_total = meter.create_counter(
        "vjepa2_requests_total",
        description="Total API requests",
    )
    frames_total = meter.create_counter(
        "vjepa2_frames_processed_total",
        description="Total video frames decoded",
    )

    request_start = time.monotonic()
    tracer = get_tracer()

    with tracer.start_as_current_span("video_inference") as root_span:
        # Record observation timestamp for L_sys calculation (if provided)
        if obs_timestamp_ms is not None:
            root_span.set_attribute("input.obs_timestamp_ms", obs_timestamp_ms)

        # Receive and buffer the uploaded file
        with tracer.start_as_current_span("input_receive") as receive_span:
            receive_span.set_attribute("input.filename", file.filename or "unknown")
            receive_span.set_attribute("input.content_type", file.content_type or "unknown")
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                content = await file.read()
                tmp.write(content)
                tmp_path = tmp.name
            receive_span.set_attribute("input.size_bytes", len(content))

        try:
            # Unified clip processing path
            # stride=None means single-clip mode (sample num_frames from video)
            effective_stride = stride if stride is not None else num_frames
            root_span.set_attribute("video.stride", effective_stride)

            clips_results = []

            for clip in iter_clips(tmp_path, num_frames=num_frames, stride=effective_stride):
                with tracer.start_as_current_span("clip_inference") as clip_span:
                    clip_span.set_attribute("clip.index", len(clips_results))
                    clip_span.set_attribute("clip.start_frame", clip.start_frame)
                    clip_span.set_attribute("clip.end_frame", clip.end_frame)
                    predictions = _model.predict(
                        clip.frames, top_k=top_k, stride=effective_stride
                    )

                frames_total.add(clip.end_frame - clip.start_frame)
                clips_results.append({
                    "clip_index": len(clips_results),
                    "start_frame": clip.start_frame,
                    "end_frame": clip.end_frame,
                    "partial": (clip.end_frame - clip.start_frame) < num_frames,
                    "predictions": [p.model_dump() for p in predictions],
                })

                # Single-clip mode: only process first clip
                if stride is None:
                    break

            root_span.set_attribute("video.clips_count", len(clips_results))
            requests_total.add(1, {"endpoint": "/infer", "status": "200"})
            request_duration.record(time.monotonic() - request_start)

            return {
                "model_name": CONFIG["model"]["name"],
                "model_version": "vit-l-16-ssv2",
                "id": f"req-{uuid.uuid4().hex[:8]}",
                "clips": clips_results,
            }
        except ValueError as e:
            requests_total.add(1, {"endpoint": "/infer", "status": "400"})
            request_duration.record(time.monotonic() - request_start)
            return JSONResponse({"error": str(e)}, status_code=400)
        except Exception as e:
            requests_total.add(1, {"endpoint": "/infer", "status": "400"})
            request_duration.record(time.monotonic() - request_start)
            return JSONResponse(
                {"error": f"Could not decode video: {e}"}, status_code=400
            )
        finally:
            Path(tmp_path).unlink(missing_ok=True)


@app.websocket("/v2/models/vjepa2/stream")
async def stream(websocket: WebSocket):
    await websocket.accept()

    if _model is None:
        await websocket.send_json({"error": "Model not ready"})
        await websocket.close(code=1011)
        return

    meter = get_meter()
    tracer = get_tracer()
    active_connections = meter.create_up_down_counter(
        "vjepa2_active_connections",
        description="Current WebSocket connections",
    )
    active_connections.add(1)

    # Receive stream configuration
    with tracer.start_as_current_span("input_receive") as span:
        span.set_attribute("input.type", "websocket_config")
        try:
            raw = await websocket.receive_json()
            config = StreamConfig(**raw)
            span.set_attribute("input.source", config.source)
        except Exception as e:
            active_connections.add(-1)
            await websocket.send_json({"error": f"Invalid config: {e}"})
            await websocket.close(code=1008)
            return

    source_path = Path(config.source)
    allowed_dir = Path(CONFIG["server"]["allowed_source_dir"])

    # Security: resolve and check path is within allowed directory
    try:
        resolved = source_path.resolve()
        if not str(resolved).startswith(str(allowed_dir.resolve())):
            active_connections.add(-1)
            await websocket.send_json({"error": "Source path not allowed"})
            await websocket.close(code=1008)
            return
    except Exception:
        active_connections.add(-1)
        await websocket.send_json({"error": "Source path not allowed"})
        await websocket.close(code=1008)
        return

    if not source_path.exists():
        active_connections.add(-1)
        await websocket.send_json({"error": f"File not found: {config.source}"})
        await websocket.close(code=1008)
        return

    frames_processed = 0
    try:
        with tracer.start_as_current_span("video_inference"):
            for clip in iter_clips(source_path, config.num_frames, config.stride):
                with tracer.start_as_current_span("clip_inference"):
                    predictions = _model.predict(
                        clip.frames, top_k=config.top_k, stride=config.stride
                    )
                msg = StreamPrediction(
                    timestamp_ms=int(clip.start_frame * 1000 / 30),
                    frame_range=[clip.start_frame, clip.end_frame],
                    predictions=predictions,
                )
                await websocket.send_json(msg.model_dump())
                frames_processed = clip.end_frame
    except WebSocketDisconnect:
        active_connections.add(-1)
        return
    except Exception as e:
        active_connections.add(-1)
        await websocket.send_json({"error": f"Inference failed: {e}"})
        await websocket.close(code=1011)
        return

    active_connections.add(-1)
    await websocket.send_json(
        StreamComplete(status="complete", frames_processed=frames_processed).model_dump()
    )
    await websocket.close()


@app.websocket("/v2/models/vjepa2/stream/browser")
async def stream_browser(websocket: WebSocket):
    await websocket.accept()
    if _model is None:
        await websocket.send_json(ErrorMessage(message="Model not ready").model_dump())
        await websocket.close(code=1013)
        return

    meter = get_meter()
    tracer = get_tracer()
    active_connections = meter.create_up_down_counter(
        "vjepa2_active_connections",
        description="Current WebSocket connections",
    )
    active_connections.add(1)

    # Receive stream configuration
    with tracer.start_as_current_span("input_receive") as span:
        span.set_attribute("input.type", "browser_stream_config")
        try:
            raw = await websocket.receive_json()
            config = BrowserStreamConfig(**raw)
            span.set_attribute("input.media_type", config.media_type or "camera")
        except Exception as e:
            active_connections.add(-1)
            await websocket.send_json(
                ErrorMessage(message=f"Invalid config: {e}").model_dump()
            )
            await websocket.close(code=1008)
            return
    max_sessions = CONFIG.get("streaming", {}).get("max_concurrent_sessions", 10)
    if len(_sessions) >= max_sessions:
        active_connections.add(-1)
        await websocket.send_json(
            ErrorMessage(message="Too many concurrent sessions").model_dump()
        )
        await websocket.close(code=1013)
        return
    session = StreamSession(
        num_frames=config.num_frames, stride=config.stride, top_k=config.top_k
    )
    _sessions[session.session_id] = session
    await websocket.send_json(
        SessionMessage(session_id=session.session_id, status="ready").model_dump()
    )

    async def on_result(result):
        await websocket.send_json(result)

    async def on_clip_queued(total):
        await websocket.send_json({"type": "progress", "clips_queued": total})

    thumb_w = CONFIG.get("streaming", {}).get("thumbnail_width", 160)
    worker_task = asyncio.create_task(inference_worker(session, _model, on_result, thumbnail_width=thumb_w))
    try:
        from opentelemetry import context as otel_context
        with tracer.start_as_current_span("stream_inference") as stream_span:
            stream_span.set_attribute("session.id", session.session_id)
            stream_span.set_attribute("input.type", config.media_type or "camera")
            parent_ctx = otel_context.get_current()
            await browser_source(websocket, session, on_clip_queued=on_clip_queued, media_type=config.media_type, parent_context=parent_ctx)
        await worker_task
    except WebSocketDisconnect:
        session.status = "processing"
        await session.clip_queue.put(None)
        await worker_task
    except Exception as e:
        logger.error("WebSocket stream error: %s", e)
        await session.clip_queue.put(None)
        await worker_task
        active_connections.add(-1)
        try:
            await websocket.send_json(ErrorMessage(message=str(e)).model_dump())
            await websocket.close(code=1011)
        except Exception:
            pass
        return
    active_connections.add(-1)
    session.complete()
    await websocket.send_json(
        CompleteMessage(
            session_id=session.session_id,
            clips_processed=session._clip_index,
            video_ready=True,
        ).model_dump()
    )
    await websocket.close()


@app.websocket("/v2/models/vjepa2/stream/rtsp")
async def stream_rtsp(websocket: WebSocket):
    await websocket.accept()
    if _model is None:
        await websocket.send_json(ErrorMessage(message="Model not ready").model_dump())
        await websocket.close(code=1013)
        return

    meter = get_meter()
    tracer = get_tracer()
    active_connections = meter.create_up_down_counter(
        "vjepa2_active_connections",
        description="Current WebSocket connections",
    )
    active_connections.add(1)

    # Receive stream configuration
    with tracer.start_as_current_span("input_receive") as span:
        span.set_attribute("input.type", "rtsp_stream_config")
        try:
            raw = await websocket.receive_json()
            config = RtspStreamConfig(**raw)
            span.set_attribute("input.rtsp_url", config.rtsp_url)
        except Exception as e:
            active_connections.add(-1)
            await websocket.send_json(
                ErrorMessage(message=f"Invalid config: {e}").model_dump()
            )
            await websocket.close(code=1008)
            return
    max_sessions = CONFIG.get("streaming", {}).get("max_concurrent_sessions", 10)
    if len(_sessions) >= max_sessions:
        active_connections.add(-1)
        await websocket.send_json(
            ErrorMessage(message="Too many concurrent sessions").model_dump()
        )
        await websocket.close(code=1013)
        return
    session = StreamSession(
        num_frames=config.num_frames, stride=config.stride, top_k=config.top_k
    )
    _sessions[session.session_id] = session
    await websocket.send_json(
        SessionMessage(session_id=session.session_id, status="connected").model_dump()
    )
    stop_event = asyncio.Event()

    async def on_result(result):
        await websocket.send_json(result)

    thumb_w = CONFIG.get("streaming", {}).get("thumbnail_width", 160)
    worker_task = asyncio.create_task(inference_worker(session, _model, on_result, thumbnail_width=thumb_w))

    async def wait_for_stop():
        try:
            while True:
                raw = await websocket.receive_json()
                action = StreamAction(**raw)
                if action.action == "stop":
                    stop_event.set()
                    break
        except WebSocketDisconnect:
            stop_event.set()

    stop_task = asyncio.create_task(wait_for_stop())
    try:
        await rtsp_source(config.rtsp_url, session, stop_event)
        await worker_task
    except Exception as e:
        await websocket.send_json(
            ErrorMessage(message=f"RTSP error: {e}").model_dump()
        )
        await session.clip_queue.put(None)
        await worker_task
        active_connections.add(-1)
        await websocket.close(code=1011)
        stop_task.cancel()
        return
    stop_task.cancel()
    active_connections.add(-1)
    session.complete()
    await websocket.send_json(
        CompleteMessage(
            session_id=session.session_id,
            clips_processed=session._clip_index,
            video_ready=True,
        ).model_dump()
    )
    await websocket.close()


@app.get("/v2/models/vjepa2/sessions/{session_id}/preview")
async def session_preview(session_id: str):
    session = _sessions.get(session_id)
    if session is None:
        return JSONResponse({"error": "Session not found"}, status_code=404)

    async def mjpeg_stream():
        import io as _io

        fps_cap = CONFIG.get("streaming", {}).get("mjpeg_fps", 10)
        interval = 1.0 / fps_cap
        while session.status in ("created", "ingesting"):
            if session._latest_frame is not None:
                img = Image.fromarray(session._latest_frame)
                buf = _io.BytesIO()
                img.save(buf, "JPEG", quality=70)
                frame_bytes = buf.getvalue()
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: "
                    + str(len(frame_bytes)).encode()
                    + b"\r\n\r\n"
                    + frame_bytes
                    + b"\r\n"
                )
            await asyncio.sleep(interval)

    return StreamingResponse(
        mjpeg_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/v2/models/vjepa2/sessions/{session_id}/video")
async def session_video(session_id: str):
    session = _sessions.get(session_id)
    if session is None:
        return JSONResponse({"error": "Session not found"}, status_code=404)

    video_path = session.temp_dir / "output.mp4"
    if not video_path.exists():
        if session.status != "complete":
            return JSONResponse({"error": "Video not ready yet"}, status_code=409)
        from app.annotate import encode_video, overlay_predictions

        frames_dir = session.temp_dir / "frames"
        frame_files = sorted(frames_dir.glob("*.jpg"))
        frames = []
        for ff in frame_files:
            img = Image.open(ff)
            frames.append(np.array(img))

        for result in session.results:
            preds = [Prediction(**p) for p in result["predictions"]]
            start, end = result["frame_range"]
            for i in range(start, min(end, len(frames))):
                frames[i] = overlay_predictions(frames[i], preds)

        if frames:
            encode_video(frames, video_path)
        else:
            return JSONResponse({"error": "No frames recorded"}, status_code=409)

    return FileResponse(
        video_path,
        media_type="video/mp4",
        filename=f"vjepa2-{session_id}.mp4",
    )


async def _session_cleanup_loop():
    """Periodically clean up expired sessions."""
    interval = CONFIG.get("streaming", {}).get("cleanup_interval_seconds", 300)
    ttl = CONFIG.get("streaming", {}).get("session_ttl_seconds", 1800)
    while True:
        await asyncio.sleep(interval)
        now = time.monotonic()
        expired = [
            sid
            for sid, s in _sessions.items()
            if s.status in ("complete", "expired") and (now - s.created_at) > ttl
        ]
        for sid in expired:
            _sessions[sid].cleanup()
            del _sessions[sid]


_static_dir = Path(__file__).parent.parent / "static"
if _static_dir.exists():
    app.mount("/", StaticFiles(directory=str(_static_dir), html=True), name="static")
