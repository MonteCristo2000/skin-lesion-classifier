from pathlib import Path

import albumentations as A
import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from PIL import Image

from model.classifier import build_model

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

val_transforms = A.Compose([
    A.Resize(384, 384),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])


def _load_model() -> torch.nn.Module:
    model = build_model(num_classes=len(CLASS_NAMES), pretrained=False)
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return model


_model = _load_model()


def predict(image: Image.Image) -> dict[str, float]:
    img_array = np.array(image.convert("RGB"))
    tensor = val_transforms(image=img_array)["image"].unsqueeze(0)

    with torch.no_grad():
        probs = F.softmax(_model(tensor), dim=1).squeeze(0)

    return {name: round(probs[i].item(), 4) for i, name in enumerate(CLASS_NAMES)}


TITLE = "Skin Lesion Classifier"

DESCRIPTION = """
Classifies dermoscopy images into **8 skin lesion categories** using an
**EfficientNet-B4** backbone fine-tuned on the
[ISIC 2019 Challenge dataset](https://challenge.isic-archive.com/landing/2019/).

**Classes:** Melanoma · Nevus (mole) · Basal Cell Carcinoma · Actinic Keratosis ·
Benign Keratosis · Dermatofibroma · Vascular Lesion · Squamous Cell Carcinoma

> ⚠️ **Disclaimer:** This tool is for **research and educational purposes only**.
> It is **NOT** a diagnostic tool and must **not** be used for medical decision-making.
> Always consult a qualified dermatologist for any skin concerns.
"""

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Dermoscopy image"),
    outputs=gr.Label(num_top_classes=4, label="Predictions"),
    title=TITLE,
    description=DESCRIPTION,
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
