import io
from unittest.mock import MagicMock, patch

import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from serving.app import CLASS_NAMES, MODEL_VERSION, app


def make_image_bytes(fmt: str = "JPEG") -> bytes:
    img = Image.new("RGB", (64, 64), color=(120, 80, 60))
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


@pytest.fixture(scope="module")
def client():
    mock_model = MagicMock()
    # Return uniform logits so softmax gives equal probabilities
    mock_model.return_value = torch.zeros(1, len(CLASS_NAMES))

    with (
        patch("serving.app.build_model", return_value=mock_model),
        patch("serving.app.torch.load", return_value={}),
    ):
        with TestClient(app) as c:
            yield c


# --- /health ---

def test_health_status(client):
    r = client.get("/health")
    assert r.status_code == 200


def test_health_schema(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model_version"] == MODEL_VERSION


# --- /predict ---

def test_predict_jpeg(client):
    r = client.post(
        "/predict",
        files={"file": ("lesion.jpg", make_image_bytes("JPEG"), "image/jpeg")},
    )
    assert r.status_code == 200


def test_predict_png(client):
    r = client.post(
        "/predict",
        files={"file": ("lesion.png", make_image_bytes("PNG"), "image/png")},
    )
    assert r.status_code == 200


def test_predict_response_schema(client):
    body = client.post(
        "/predict",
        files={"file": ("lesion.jpg", make_image_bytes("JPEG"), "image/jpeg")},
    ).json()
    assert "predicted_class" in body
    assert "confidence" in body
    assert "probabilities" in body
    assert "model_version" in body
    assert body["model_version"] == MODEL_VERSION


def test_predict_all_classes_present(client):
    body = client.post(
        "/predict",
        files={"file": ("lesion.jpg", make_image_bytes("JPEG"), "image/jpeg")},
    ).json()
    assert set(body["probabilities"].keys()) == set(CLASS_NAMES)


def test_predict_probabilities_sum_to_one(client):
    body = client.post(
        "/predict",
        files={"file": ("lesion.jpg", make_image_bytes("JPEG"), "image/jpeg")},
    ).json()
    total = sum(body["probabilities"].values())
    assert abs(total - 1.0) < 1e-4


def test_predict_confidence_matches_predicted_class(client):
    body = client.post(
        "/predict",
        files={"file": ("lesion.jpg", make_image_bytes("JPEG"), "image/jpeg")},
    ).json()
    assert body["confidence"] == body["probabilities"][body["predicted_class"]]


def test_predict_invalid_content_type(client):
    r = client.post(
        "/predict",
        files={"file": ("report.pdf", b"not-an-image", "application/pdf")},
    )
    assert r.status_code == 400
    assert "JPEG" in r.json()["detail"] or "PNG" in r.json()["detail"]
