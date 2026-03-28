# Milestone 4: Stream Sources & Web UI

**Date:** 2026-03-28
**Status:** Design
**Depends on:** Milestone 3 (observability) — complete

## Overview

Add live video stream ingestion (browser camera, RTSP cameras) and a single-page web UI for interacting with the V-JEPA2 inference service. The existing file-upload and WebSocket streaming endpoints remain unchanged.

## Goals

1. Accept live video from browser cameras via MediaRecorder chunks over WebSocket
2. Accept RTSP streams from IP cameras (e.g., UniFi Protect) via PyAV in-process decode
3. Provide a web UI for uploading video, recording from camera, or connecting to RTSP streams
4. Display inference results as a scrolling list with thumbnails alongside the input video
5. Generate downloadable annotated video (label overlay) after a session ends
6. Decouple ingestion from inference via a clip queue to enable future parallel workers

## Non-Goals

- Real WebTransport (HTTP/3) — protocol is designed for future upgrade but runs over WebSocket now
- Parallel inference workers — queue is in place but M4 uses a single worker
- Camera discovery / UniFi Protect API integration
- User authentication or multi-tenancy
- Mobile-optimized UI

## Architecture

### System Layers

```
Sources (independent adapters)
  ├── Browser camera  → WS /stream/browser  → MediaRecorder chunks
  ├── RTSP camera     → WS /stream/rtsp     → PyAV av.open(rtsp://)
  └── File upload     → WS /stream/browser   → File read as chunks
                              │
                    ┌─────────▼──────────┐
                    │   Shared Pipeline   │
                    │                     │
                    │  FrameBuffer        │
                    │    → accumulates    │
                    │    → yields clips   │
                    │                     │
                    │  ClipQueue          │
                    │    → asyncio.Queue  │
                    └─────────┬──────────┘
                              │
                    ┌─────────▼──────────┐
                    │  Inference Worker   │
                    │  VJepa2Model        │
                    │  .predict()         │
                    └─────────┬──────────┘
                              │
                    ┌─────────▼──────────┐
                    │  Results            │
                    │  → WS to browser    │
                    │  → thumbnail + preds│
                    │  → disk (for video) │
                    └────────────────────┘
```

### Source Adapters

Each source type has its own ingestion logic but feeds the same shared pipeline:

- **Browser source:** Receives binary WebSocket messages (webm/mp4 chunks from MediaRecorder), decodes frames with PyAV using an in-memory container, and pushes numpy arrays into the FrameBuffer.
- **RTSP source:** Opens the RTSP URL with `av.open(rtsp_url)` and iterates frames. Since PyAV frame iteration is blocking, it runs in a thread via `asyncio.to_thread()` and pushes numpy arrays into the FrameBuffer.
- **File upload:** The browser reads the file and sends chunks over the same browser WebSocket endpoint. Same server-side code path as camera recording.

### Shared Pipeline

- **FrameBuffer:** Accumulates frames per session. When `num_frames` frames are available, yields a `Clip` and advances by `stride` frames. Trims consumed frames to bound memory.
- **ClipQueue:** `asyncio.Queue` connecting the buffer to inference. Decouples ingestion speed from inference speed. Single worker initially; queue makes adding workers trivial later.
- **Inference worker:** Async task that pulls clips from the queue, runs `VJepa2Model.predict()`, and emits results back to the session.

### RTSP Preview Relay

The server relays decoded RTSP frames back to the browser as an MJPEG stream:

- Endpoint: `GET /v2/models/vjepa2/sessions/{id}/preview`
- Content-Type: `multipart/x-mixed-replace; boundary=frame`
- Each part is a JPEG-encoded frame
- Browser displays via `<img src="...">` — no JavaScript needed
- Frames are already decoded in-process, just re-encoded to JPEG

This gives RTSP sources a live preview in the UI without requiring the browser to speak RTSP.

## API Endpoints

### Existing (unchanged)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/v2/health/live` | Liveness probe |
| GET | `/v2/health/ready` | Readiness probe |
| GET | `/v2/models/vjepa2` | Model metadata |
| POST | `/v2/models/vjepa2/infer` | File upload inference |
| WS | `/v2/models/vjepa2/stream` | File-path streaming |

### New

| Method | Path | Description |
|--------|------|-------------|
| WS | `/v2/models/vjepa2/stream/browser` | Browser camera/file streaming |
| WS | `/v2/models/vjepa2/stream/rtsp` | RTSP stream ingestion |
| GET | `/v2/models/vjepa2/sessions/{id}/preview` | MJPEG preview relay |
| GET | `/v2/models/vjepa2/sessions/{id}/video` | Annotated video download |
| GET | `/` | Static web UI |

## WebSocket Protocol

### Browser Stream (`/stream/browser`)

**Client → Server:**

1. JSON config (first message):
```json
{"top_k": 5, "num_frames": 16, "stride": 8}
```

2. Binary messages: MediaRecorder video chunks (webm or mp4)

3. JSON stop (final message):
```json
{"action": "stop"}
```

**Server → Client:**

Session acknowledgment:
```json
{"type": "session", "session_id": "abc123", "status": "ready"}
```

Prediction results (one per clip):
```json
{
  "type": "prediction",
  "clip_index": 0,
  "timestamp_ms": 500,
  "frame_range": [0, 16],
  "thumbnail": "<base64 JPEG>",
  "predictions": [
    {"label": "Pushing something", "score": 0.423},
    {"label": "Moving something", "score": 0.187}
  ]
}
```

Session complete:
```json
{
  "type": "complete",
  "session_id": "abc123",
  "clips_processed": 12,
  "video_ready": true
}
```

Error:
```json
{"type": "error", "message": "..."}
```

### RTSP Stream (`/stream/rtsp`)

**Client → Server:**

1. JSON config (first message):
```json
{"rtsp_url": "rtsp://192.168.1.1:7447/stream", "top_k": 5, "num_frames": 16, "stride": 8}
```

2. JSON stop:
```json
{"action": "stop"}
```

**Server → Client:** Same prediction/complete/error messages as browser stream.

## Web UI

### Technology

- **Alpine.js** via CDN import — reactive state, no build step
- Static files served by FastAPI from `static/` directory
- Single HTML page with CSS and JS

### Layout

- **Top bar:** App title, model status indicator (live/ready)
- **Tab bar:** Three tabs — "Upload Video", "Record Camera", "RTSP Stream"
- **Left panel (Input):** Source-specific controls and video preview
- **Right panel (Results):** Scrolling list of clip results
- **Post-session:** Download button for annotated video

### Tab Behaviors

**Upload Video:**
- Drag-and-drop zone or file picker
- `<video>` element for preview after selection
- Start button reads file as chunks, sends via `/stream/browser` WS

**Record Camera:**
- `getUserMedia()` for camera access with `<video>` preview
- MediaRecorder sends chunks via `/stream/browser` WS
- Start/Stop controls recording session

**RTSP Stream:**
- Text input for RTSP URL
- `<img>` pointing at `/sessions/{id}/preview` for MJPEG relay preview
- Connect/Disconnect controls

### Results Panel

Each result item shows:
- **Thumbnail:** JPEG still frame from the clip (base64 from server)
- **Timestamp:** Start–end time and frame range
- **Predictions:** Top-k labels with scores, top prediction highlighted in green
- **Processing indicator:** Dashed border with spinner for clip currently being inferred

### Inference Parameters

Configurable in the UI before starting a session:
- `top_k` (default 5)
- `num_frames` (default 16)
- `stride` (default 8)

### Annotated Video Download

- Download button grayed out during active session
- Activates when session completes and annotated video is generated
- Triggers `GET /sessions/{id}/video` which returns the MP4 file

## Modules

### New Files

**`app/pipeline.py`** — Shared streaming pipeline:
- `FrameBuffer` — Accumulates frames, yields `Clip` objects when `num_frames` reached, advances by `stride`, trims consumed frames
- `StreamSession` — Per-session state: buffer, clip queue, results list, session ID, status, temp directory
- `inference_worker(session, model)` — Async task pulling clips from session queue, running `model.predict()`, appending results, sending via callback

**`app/sources.py`** — Source adapters:
- `browser_source(websocket, session)` — Receives binary WS messages, decodes with PyAV in-memory container, pushes frames to session buffer
- `rtsp_source(url, session)` — Opens RTSP URL with `av.open()`, iterates frames in thread via `asyncio.to_thread()`, pushes to session buffer

**`app/annotate.py`** — Video annotation:
- `overlay_predictions(frame, predictions)` — Burns top-k labels and scores onto a frame image using Pillow (ImageDraw)
- `encode_video(session)` — Reads stored frames from session temp dir, applies overlays using session results, encodes to MP4 via PyAV

**`static/index.html`** — Single-page UI
**`static/app.js`** — Alpine.js application logic, WebSocket client
**`static/style.css`** — Light-theme styling

### Modified Files

**`app/main.py`:**
- Mount `static/` directory via `StaticFiles`
- Add `/stream/browser` WebSocket endpoint
- Add `/stream/rtsp` WebSocket endpoint
- Add `/sessions/{id}/preview` MJPEG streaming endpoint
- Add `/sessions/{id}/video` file download endpoint
- Session registry (dict of active sessions) with cleanup task

**`app/schemas.py`:**
- `BrowserStreamConfig` — top_k, num_frames, stride
- `RtspStreamConfig` — rtsp_url, top_k, num_frames, stride (with URL validation)
- `StreamAction` — action enum (stop)
- `PredictionMessage` — type, clip_index, timestamp_ms, frame_range, thumbnail, predictions
- `SessionMessage` — type, session_id, status
- `CompleteMessage` — type, session_id, clips_processed, video_ready
- `ErrorMessage` — type, message

**`app/video.py`:**
- Extract core clip assembly logic used by both `iter_clips()` and `FrameBuffer` to avoid duplication

## Session Lifecycle

### States

```
created → ingesting → processing → complete → expired
```

- **created:** Config received, session ID assigned, temp directory created
- **ingesting:** Source adapter feeding frames. Inference runs concurrently as clips become available.
- **processing:** Source stopped. Remaining queued clips still being inferred.
- **complete:** All clips inferred. Annotated video generated. Download available.
- **expired:** TTL exceeded (default 30 minutes). Temp files deleted.

### Storage

Each session uses a temp directory: `/tmp/vjepa2-sessions/{session_id}/`

Contents:
- `frames/` — Decoded frames written to disk as JPEG files (for annotated video generation)
- `results.jsonl` — Append-only prediction results per clip
- `output.mp4` — Annotated video (written on session complete)

### Memory Management

- FrameBuffer holds only the current sliding window (`num_frames` frames) in memory
- Frames are written to disk as they arrive for later use in annotated video generation
- This bounds memory to approximately `num_frames * frame_size` regardless of stream length

### Cleanup

- Background asyncio task runs periodically (every 5 minutes)
- Scans for sessions in `complete` or `expired` state past their TTL
- Deletes session temp directory
- TTL configurable in `model_config.yaml` (default 30 minutes)

## Error Handling

| Scenario | Behavior |
|----------|----------|
| RTSP URL unreachable | Server sends error message, closes WebSocket |
| RTSP stream drops mid-session | Server sends error, transitions to `processing` (finishes remaining clips) |
| Browser disconnects mid-stream | Session transitions to `processing`, completes remaining clips |
| Invalid MediaRecorder chunks | Server sends error, keeps session alive for retry |
| Model not ready | WebSocket rejects connection with close code 1013 |
| Session not found | GET endpoints return 404 |
| Session expired | GET endpoints return 410 Gone |

## Testing

### Unit Tests

- **`test_pipeline.py`** — FrameBuffer accumulation and clip yielding with various stride values, queue mechanics, session state transitions
- **`test_sources.py`** — Browser source with mock WebSocket and synthetic video chunks, RTSP source with mock PyAV container
- **`test_annotate.py`** — Label overlay rendering on synthetic frames, video encoding roundtrip

### Integration Tests

- **`test_main.py`** (extend) — Full WebSocket flow for browser and RTSP endpoints with synthetic data, MJPEG preview endpoint, session video download

### Manual Testing

- Web UI tested manually in Chrome
- RTSP tested with FFmpeg as a synthetic RTSP source: `ffmpeg -re -i sample.mp4 -f rtsp rtsp://localhost:8554/test`

## New Dependencies

None. All functionality is covered by existing dependencies:
- `av` (PyAV) — RTSP decode, video encoding, MediaRecorder chunk decode
- `pillow` — JPEG encoding for thumbnails/MJPEG, text overlay for annotations
- `alpine.js` — CDN import in HTML, no Python package

## Configuration

New entries in `configs/model_config.yaml`:

```yaml
streaming:
  session_ttl_seconds: 1800        # 30 minutes
  cleanup_interval_seconds: 300    # 5 minutes
  mjpeg_fps: 10                    # MJPEG relay frame rate cap
  thumbnail_width: 160             # Thumbnail JPEG width (height proportional)
  max_concurrent_sessions: 10      # Limit active sessions
```

## Observability

Existing OTel instrumentation (M3) covers inference metrics automatically. New metrics to add:

- `vjepa2_active_sessions` (gauge) — Number of active streaming sessions
- `vjepa2_session_duration_seconds` (histogram) — Total session duration
- `vjepa2_frames_ingested_total` (counter) — Frames received from all sources
- `vjepa2_queue_depth` (gauge) — Current clip queue depth per session
