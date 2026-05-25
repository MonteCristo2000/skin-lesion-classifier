import io
from unittest.mock import MagicMock, patch

import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from serving.app import CLASS_NAMES, app


@pytest.fixture(scope="session")
def jpeg_bytes() -> bytes:
    img = Image.new("RGB", (64, 64), color=(120, 80, 60))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture(scope="session")
def png_bytes() -> bytes:
    img = Image.new("RGB", (64, 64), color=(120, 80, 60))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture(scope="session")
def client():
    """FastAPI TestClient with model loading fully mocked out."""
    mock_model = MagicMock()
    # Uniform logits → softmax gives equal probabilities across all classes
    mock_model.return_value = torch.zeros(1, len(CLASS_NAMES))

    with (
        patch("serving.app.build_model", return_value=mock_model),
        patch("serving.app.torch.load", return_value={}),
    ):
        with TestClient(app) as c:
            yield c
