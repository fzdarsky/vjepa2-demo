# tests/test_main.py
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.schemas import Prediction


@pytest.fixture
def mock_vjepa2_model():
    """Mock VJepa2Model for API tests."""
    with patch("app.main.VJepa2Model") as MockClass:
        instance = MagicMock()
        instance.id2label = {i: f"Action {i}" for i in range(174)}
        MockClass.return_value = instance
        yield instance


@pytest.fixture
def client(mock_vjepa2_model):
    from app.main import app
    with TestClient(app) as c:
        yield c


def test_liveness(client):
    resp = client.get("/v2/health/live")
    assert resp.status_code == 200


def test_readiness_when_model_loaded(client):
    resp = client.get("/v2/health/ready")
    assert resp.status_code == 200


def test_model_metadata(client):
    resp = client.get("/v2/models/vjepa2")
    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "vjepa2"
    assert "versions" in data
    assert data["platform"] == "pytorch"
