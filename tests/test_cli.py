import argparse
import json
import subprocess
import sys
from unittest.mock import MagicMock, patch

import pytest

from app.schemas import Prediction


def test_main_module_shows_subcommands():
    result = subprocess.run(
        [sys.executable, "-m", "app", "--help"],
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0
    assert "serve" in result.stdout
    assert "infer" in result.stdout
    assert "download" in result.stdout


@pytest.fixture
def mock_model():
    import torch
    model = MagicMock()
    model.predict.return_value = [
        Prediction(label="Pushing something", score=0.87),
        Prediction(label="Pulling something", score=0.06),
    ]
    # Mock processor to return a tensor when called
    mock_processor = MagicMock()
    mock_processor.return_value = {
        "pixel_values_videos": torch.randn(1, 16, 3, 224, 224)
    }
    model.processor = mock_processor
    return model


def test_infer_text_output(mock_model, sample_video_path, capsys):
    args = argparse.Namespace(
        files=[str(sample_video_path)],
        stride=None, num_frames=16, top_k=5,
        save_frames=False, output="/tmp/out", format="text",
    )
    with patch("app.cli._load_model", return_value=mock_model):
        from app.cli import cmd_infer
        cmd_infer(args)
    captured = capsys.readouterr()
    assert "Pushing something" in captured.out


def test_infer_jsonl_output(mock_model, sample_video_path, capsys):
    args = argparse.Namespace(
        files=[str(sample_video_path)],
        stride=None, num_frames=16, top_k=5,
        save_frames=False, output="/tmp/out", format="jsonl",
    )
    with patch("app.cli._load_model", return_value=mock_model):
        from app.cli import cmd_infer
        cmd_infer(args)
    captured = capsys.readouterr()
    lines = [l for l in captured.out.strip().split("\n") if l]
    for line in lines:
        data = json.loads(line)
        assert "clip_index" in data
        assert "predictions" in data


def test_infer_json_output(mock_model, sample_video_path, capsys):
    args = argparse.Namespace(
        files=[str(sample_video_path)],
        stride=None, num_frames=16, top_k=5,
        save_frames=False, output="/tmp/out", format="json",
    )
    with patch("app.cli._load_model", return_value=mock_model):
        from app.cli import cmd_infer
        cmd_infer(args)
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert isinstance(data, list)
    assert len(data) >= 1


def test_infer_save_frames_creates_output(mock_model, sample_video_path, tmp_path):
    args = argparse.Namespace(
        files=[str(sample_video_path)],
        stride=None, num_frames=16, top_k=5,
        save_frames=True, output=str(tmp_path / "output"), format="text",
    )
    with patch("app.cli._load_model", return_value=mock_model):
        from app.cli import cmd_infer
        cmd_infer(args)
    output_dir = tmp_path / "output"
    assert output_dir.exists()
