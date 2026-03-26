import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from app.schemas import Clip, Prediction


def make_contact_sheet(images: list[np.ndarray], cols: int = 4) -> np.ndarray:
    """Arrange images into a grid. Returns single image array."""
    rows = math.ceil(len(images) / cols)
    h, w = images[0].shape[:2]
    sheet = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    for i, img in enumerate(images):
        r, c = divmod(i, cols)
        sheet[r * h:(r + 1) * h, c * w:(c + 1) * w] = img
    return sheet


def save_clip_frames(
    clip: Clip,
    clip_index: int,
    output_dir: Path,
    source_name: str,
    predictions: list[Prediction],
    processor=None,
) -> dict:
    """Save original and (optionally) processed frames for a clip."""
    clip_dir = output_dir / f"clip_{clip_index:03d}"
    clip_dir.mkdir(parents=True, exist_ok=True)

    num_frames = clip.frames.shape[0]
    is_partial = (clip.end_frame - clip.start_frame) < num_frames

    # Save original frames
    original_images = []
    for i in range(num_frames):
        frame = clip.frames[i]
        original_images.append(frame)
        Image.fromarray(frame).save(clip_dir / f"frame_{i:03d}_original.png")

    # Original contact sheet
    contact = make_contact_sheet(original_images)
    Image.fromarray(contact).save(clip_dir / "contact_original.png")

    # Save processed frames if processor provided
    if processor is not None:
        processed_output = processor(list(clip.frames), return_tensors="pt")
        key = "pixel_values_videos" if "pixel_values_videos" in processed_output else "pixel_values"
        pixel_values = processed_output[key]

        # pixel_values shape: (1, T, C, H, W) -- rescale to [0,255] for viz
        frames_tensor = pixel_values[0]  # (T, C, H, W)
        processed_images = []
        for i in range(frames_tensor.shape[0]):
            frame_t = frames_tensor[i]  # (C, H, W)
            frame_np = frame_t.permute(1, 2, 0).numpy()  # (H, W, C)
            frame_np = frame_np - frame_np.min()
            if frame_np.max() > 0:
                frame_np = frame_np / frame_np.max()
            frame_np = (frame_np * 255).astype(np.uint8)
            processed_images.append(frame_np)
            Image.fromarray(frame_np).save(clip_dir / f"frame_{i:03d}_processed.png")

        contact_proc = make_contact_sheet(processed_images)
        Image.fromarray(contact_proc).save(clip_dir / "contact_processed.png")

    # Write manifest
    manifest = {
        "clip_index": clip_index,
        "start_frame": clip.start_frame,
        "end_frame": clip.end_frame,
        "partial": is_partial,
        "num_frames": num_frames,
        "source": source_name,
        "original_resolution": [int(clip.frames.shape[1]), int(clip.frames.shape[2])],
        "predictions": [
            {"label": p.label, "score": p.score} for p in predictions
        ],
    }
    (clip_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
