import io
from contextlib import asynccontextmanager
from pathlib import Path

import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException, UploadFile
from PIL import Image

from model.classifier import build_model
from serving.schemas import HealthResponse, PredictionResponse
from src.transforms import val_transforms

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

_model: torch.nn.Module | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model
    _model = build_model(num_classes=len(CLASS_NAMES), pretrained=False)
    state_dict = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    _model.load_state_dict(state_dict)
    _model.eval()
    yield
    _model = None


app = FastAPI(title="Skin Lesion Classifier", version=MODEL_VERSION, lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        model_loaded=_model is not None,
        model_version=MODEL_VERSION,
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile):
    if file.content_type not in ("image/jpeg", "image/png"):
        raise HTTPException(status_code=400, detail="Only JPEG and PNG images are supported.")

    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    tensor = val_transforms(image).unsqueeze(0)

    with torch.no_grad():
        logits = _model(tensor)
        probs = F.softmax(logits, dim=1).squeeze(0)

    predicted_idx = probs.argmax().item()
    probabilities = {name: round(probs[i].item(), 6) for i, name in enumerate(CLASS_NAMES)}

    return PredictionResponse(
        predicted_class=CLASS_NAMES[predicted_idx],
        confidence=round(probs[predicted_idx].item(), 6),
        probabilities=probabilities,
        model_version=MODEL_VERSION,
    )
