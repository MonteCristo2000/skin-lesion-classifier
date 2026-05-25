import io
import time
from contextlib import asynccontextmanager
from pathlib import Path

import albumentations as A
import numpy as np
import torch
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from fastapi import FastAPI, HTTPException, UploadFile
from PIL import Image
from prometheus_client import make_asgi_app

from model.classifier import build_model
from monitoring.drift_detector import DriftDetector
from monitoring.metrics import (
    CONFIDENCE_HISTOGRAM,
    INFERENCE_LATENCY,
    MODEL_LOADED,
    PREDICTION_COUNTER,
    REQUEST_COUNTER,
)
from monitoring.metrics_logger import MetricsLogger
from serving.schemas import HealthResponse, PredictionResponse

inference_transforms = A.Compose([
    A.Resize(384, 384),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

CLASS_NAMES = [
    "Melanoma",
    "Nevus (mole)",
    "Basal Cell Carcinoma",
    "Actinic Keratosis",
    "Benign Keratosis",
    "Dermatofibroma",
    "Vascular Lesion",
    "Squamous Cell Carcinoma",
]

MODEL_PATH = Path("model/best_model.pth")
MODEL_VERSION = "1.0.0"

_REF_CONFIDENCES_PATH = Path("model/reference_confidences.npy")

_model: torch.nn.Module | None = None
_metrics_logger = MetricsLogger()

# Loaded with real validation-set confidences when the file exists; otherwise
# drift detection is silently disabled (DriftDetector.is_active == False).
_drift_detector = DriftDetector(
    reference=np.load(_REF_CONFIDENCES_PATH).tolist()
    if _REF_CONFIDENCES_PATH.exists()
    else [],
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model
    _model = build_model(num_classes=len(CLASS_NAMES), pretrained=False)
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    _model.load_state_dict(state_dict)
    _model.eval()
    MODEL_LOADED.set(1)
    yield
    _model = None
    MODEL_LOADED.set(0)


app = FastAPI(title="Skin Lesion Classifier", version=MODEL_VERSION, lifespan=lifespan)
app.mount("/metrics", make_asgi_app())


@app.get("/health", response_model=HealthResponse)
def health():
    REQUEST_COUNTER.labels(endpoint="/health", status="200").inc()
    return HealthResponse(
        status="ok",
        model_loaded=_model is not None,
        model_version=MODEL_VERSION,
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile):
    if file.content_type not in ("image/jpeg", "image/png"):
        REQUEST_COUNTER.labels(endpoint="/predict", status="400").inc()
        raise HTTPException(status_code=400, detail="Only JPEG and PNG images are supported.")

    contents = await file.read()
    image = np.array(Image.open(io.BytesIO(contents)).convert("RGB"))
    tensor = inference_transforms(image=image)["image"].unsqueeze(0)

    t0 = time.perf_counter()
    with torch.no_grad():
        logits = _model(tensor)
        probs = F.softmax(logits, dim=1).squeeze(0)
    latency = time.perf_counter() - t0
    INFERENCE_LATENCY.observe(latency)

    predicted_idx = probs.argmax().item()
    confidence = round(probs[predicted_idx].item(), 6)
    probabilities = {name: round(probs[i].item(), 6) for i, name in enumerate(CLASS_NAMES)}
    PREDICTION_COUNTER.labels(predicted_class=CLASS_NAMES[predicted_idx]).inc()
    CONFIDENCE_HISTOGRAM.observe(confidence)
    REQUEST_COUNTER.labels(endpoint="/predict", status="200").inc()

    _metrics_logger.record_prediction(
        predicted_class=CLASS_NAMES[predicted_idx],
        confidence=confidence,
        latency_seconds=latency,
    )
    _drift_detector.update(confidence)

    return PredictionResponse(
        predicted_class=CLASS_NAMES[predicted_idx],
        confidence=confidence,
        probabilities=probabilities,
        model_version=MODEL_VERSION,
    )


@app.get("/stats")
def stats():
    summary = _metrics_logger.get_summary()
    summary["drift_events"] = _drift_detector.drift_count
    summary["drift_detection_active"] = _drift_detector.is_active
    return summary
