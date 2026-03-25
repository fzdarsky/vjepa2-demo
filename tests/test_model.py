# tests/test_model.py
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from app.model import VJepa2Model
from app.schemas import Prediction


@pytest.fixture
def mock_model_and_processor():
    """Mock HuggingFace model and processor."""
    with (
        patch("app.model.AutoModelForVideoClassification") as MockModel,
        patch("app.model.AutoVideoProcessor") as MockProcessor,
    ):
        # Mock processor
        processor_instance = MagicMock()
        processor_instance.return_value = {
            "pixel_values_videos": torch.randn(1, 16, 3, 256, 256)
        }
        MockProcessor.from_pretrained.return_value = processor_instance

        # Mock model
        model_instance = MagicMock()
        model_instance.config.id2label = {
            i: f"Action {i}" for i in range(174)
        }
        logits = torch.randn(1, 174)
        logits[0, 42] = 10.0  # Make class 42 the top prediction
        model_instance.return_value = MagicMock(logits=logits)
        model_instance.to.return_value = model_instance
        model_instance.eval.return_value = model_instance
        MockModel.from_pretrained.return_value = model_instance

        yield model_instance, processor_instance


def test_model_init(mock_model_and_processor):
    model = VJepa2Model(
        hf_model_id="facebook/vjepa2-vitl-fpc16-256-ssv2", device="cpu"
    )
    assert model.device == "cpu"


def test_model_predict_returns_predictions(mock_model_and_processor):
    model = VJepa2Model(
        hf_model_id="facebook/vjepa2-vitl-fpc16-256-ssv2", device="cpu"
    )
    frames = np.random.randint(0, 255, (16, 256, 256, 3), dtype=np.uint8)
    predictions = model.predict(frames, top_k=5)

    assert len(predictions) == 5
    assert all(isinstance(p, Prediction) for p in predictions)
    assert predictions[0].label == "Action 42"
    assert predictions[0].score > predictions[1].score


def test_model_predict_top_k(mock_model_and_processor):
    model = VJepa2Model(
        hf_model_id="facebook/vjepa2-vitl-fpc16-256-ssv2", device="cpu"
    )
    frames = np.random.randint(0, 255, (16, 256, 256, 3), dtype=np.uint8)

    preds_3 = model.predict(frames, top_k=3)
    assert len(preds_3) == 3

    preds_10 = model.predict(frames, top_k=10)
    assert len(preds_10) == 10


def test_model_scores_sum_to_one(mock_model_and_processor):
    model = VJepa2Model(
        hf_model_id="facebook/vjepa2-vitl-fpc16-256-ssv2", device="cpu"
    )
    frames = np.random.randint(0, 255, (16, 256, 256, 3), dtype=np.uint8)
    predictions = model.predict(frames, top_k=174)
    total = sum(p.score for p in predictions)
    assert abs(total - 1.0) < 1e-5
