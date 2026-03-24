# app/model.py
import numpy as np
import torch
from transformers import AutoModelForVideoClassification, AutoVideoProcessor

from app.schemas import Prediction


def select_device(requested: str | None = None) -> str:
    """Auto-detect or validate the compute device."""
    if requested:
        device = requested.lower()
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        if device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available")
        return device

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class VJepa2Model:
    def __init__(self, hf_model_id: str, device: str):
        self.device = device
        self.processor = AutoVideoProcessor.from_pretrained(hf_model_id)
        self.model = AutoModelForVideoClassification.from_pretrained(hf_model_id)
        self.model.to(device)
        self.model.eval()
        self.id2label: dict[int, str] = self.model.config.id2label

    def predict(self, frames: np.ndarray, top_k: int = 5) -> list[Prediction]:
        """Run inference on video frames.

        Args:
            frames: numpy array of shape (num_frames, H, W, 3), dtype uint8.
            top_k: number of top predictions to return.

        Returns:
            List of Prediction(label, score), sorted by score descending.
        """
        inputs = self.processor(list(frames), return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)

        with torch.no_grad():
            outputs = self.model(pixel_values)

        probs = torch.softmax(outputs.logits, dim=-1)[0]
        top_scores, top_indices = torch.topk(probs, k=top_k)

        return [
            Prediction(
                label=self.id2label[idx.item()],
                score=round(score.item(), 6),
            )
            for score, idx in zip(top_scores, top_indices)
        ]
