"""Video annotation: overlay predictions on frames and encode to MP4."""

from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from app.schemas import Prediction


def overlay_predictions(
    frame: np.ndarray,
    predictions: list[Prediction],
    font_size: int = 16,
) -> np.ndarray:
    """Burn prediction labels and scores onto a frame image.

    Returns a new frame with text overlay at the bottom.
    """
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)

    if not predictions:
        return np.array(img)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
    except (OSError, IOError):
        font = ImageFont.load_default()

    # Draw semi-transparent background bar at the bottom
    bar_height = (font_size + 4) * len(predictions) + 8
    y_start = img.height - bar_height
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.rectangle(
        [(0, y_start), (img.width, img.height)],
        fill=(0, 0, 0, 160),
    )
    img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(img)

    # Draw prediction text
    y = y_start + 4
    for i, pred in enumerate(predictions):
        color = (52, 168, 83) if i == 0 else (200, 200, 200)  # green for top
        text = f"{pred.label}: {pred.score:.3f}"
        draw.text((8, y), text, fill=color, font=font)
        y += font_size + 4

    return np.array(img)


def encode_video(
    frames: list[np.ndarray],
    output_path: Path | str,
    fps: int = 30,
) -> None:
    """Encode a list of RGB frames to an MP4 file via PyAV."""
    output_path = Path(output_path)
    container = av.open(str(output_path), mode="w")
    h, w = frames[0].shape[:2]
    stream = container.add_stream("libx264", rate=fps)
    stream.width = w
    stream.height = h
    stream.pix_fmt = "yuv420p"

    for i, frame_data in enumerate(frames):
        frame = av.VideoFrame.from_ndarray(frame_data, format="rgb24")
        frame.pts = i
        for packet in stream.encode(frame):
            container.mux(packet)

    for packet in stream.encode():
        container.mux(packet)
    container.close()
