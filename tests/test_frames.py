import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from app.frames import make_contact_sheet, save_clip_frames
from app.schemas import Clip, Prediction


@pytest.fixture
def sample_clip():
    frames = np.random.randint(0, 255, (16, 240, 320, 3), dtype=np.uint8)
    return Clip(frames=frames, start_frame=0, end_frame=16)


@pytest.fixture
def sample_predictions():
    return [
        Prediction(label="Pushing something", score=0.87),
        Prediction(label="Pulling something", score=0.06),
    ]


def test_make_contact_sheet_shape():
    images = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(16)]
    sheet = make_contact_sheet(images, cols=4)
    assert sheet.shape == (400, 400, 3)


def test_make_contact_sheet_uneven():
    images = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(5)]
    sheet = make_contact_sheet(images, cols=4)
    # 5 images, 4 cols: 2 rows
    assert sheet.shape == (200, 400, 3)


def test_save_clip_frames_creates_files(tmp_path, sample_clip, sample_predictions):
    save_clip_frames(
        clip=sample_clip,
        clip_index=0,
        output_dir=tmp_path,
        source_name="test_video.mp4",
        predictions=sample_predictions,
    )
    clip_dir = tmp_path / "clip_000"
    assert clip_dir.exists()
    assert (clip_dir / "frame_000_original.png").exists()
    assert (clip_dir / "frame_015_original.png").exists()
    assert (clip_dir / "contact_original.png").exists()
    assert (clip_dir / "manifest.json").exists()


def test_save_clip_frames_manifest_content(tmp_path, sample_clip, sample_predictions):
    save_clip_frames(
        clip=sample_clip,
        clip_index=2,
        output_dir=tmp_path,
        source_name="hands.mp4",
        predictions=sample_predictions,
    )
    manifest = json.loads((tmp_path / "clip_002" / "manifest.json").read_text())
    assert manifest["clip_index"] == 2
    assert manifest["start_frame"] == 0
    assert manifest["end_frame"] == 16
    assert manifest["partial"] is False
    assert manifest["source"] == "hands.mp4"
    assert len(manifest["predictions"]) == 2


def test_save_clip_frames_partial_clip(tmp_path, sample_predictions):
    frames = np.zeros((16, 240, 320, 3), dtype=np.uint8)
    partial = Clip(frames=frames, start_frame=48, end_frame=55)
    save_clip_frames(
        clip=partial,
        clip_index=3,
        output_dir=tmp_path,
        source_name="video.mp4",
        predictions=sample_predictions,
    )
    manifest = json.loads((tmp_path / "clip_003" / "manifest.json").read_text())
    assert manifest["partial"] is True


def test_save_clip_frames_with_processor(tmp_path, sample_clip, sample_predictions):
    mock_processor = MagicMock()
    mock_processor.return_value = {
        "pixel_values_videos": torch.randn(1, 16, 3, 256, 256)
    }

    save_clip_frames(
        clip=sample_clip,
        clip_index=0,
        output_dir=tmp_path,
        source_name="test.mp4",
        predictions=sample_predictions,
        processor=mock_processor,
    )
    clip_dir = tmp_path / "clip_000"
    assert (clip_dir / "frame_000_processed.png").exists()
    assert (clip_dir / "contact_processed.png").exists()
