# Skin Lesion Classifier

A deep learning pipeline for classifying skin lesions into 8 classes using EfficientNet-B4, served via FastAPI with Prometheus monitoring.

**Classes:** Melanoma, Nevus (mole), Basal Cell Carcinoma, Actinic Keratosis, Benign Keratosis, Dermatofibroma, Vascular Lesion, Squamous Cell Carcinoma

## Project Structure

```
skin-lesion-classifier/
├── src/
│   ├── dataset.py          # SkinLesionDataset — CSV-driven, PIL-based
│   ├── transforms.py       # Train (augmented) and val (deterministic) pipelines
│   ├── train.py            # Training loop — AdamW + CosineAnnealingLR, saves best.pth
│   ├── evaluate.py         # Classification report and confusion matrix
│   └── utils.py            # Logger, save/load checkpoint
│
├── model/
│   └── classifier.py       # EfficientNet-B4 backbone (timm), custom classification head
│
├── serving/
│   ├── app.py              # FastAPI app — /health, /predict, /metrics
│   └── schemas.py          # Pydantic models: PredictionResponse, HealthResponse
│
├── configs/
│   └── train.yaml          # Training hyperparameters and data paths
│
├── monitoring/
│   ├── metrics.py          # Prometheus counters, histograms, and gauges
│   └── prometheus.yml      # Prometheus scrape config
│
├── docker/
│   ├── Dockerfile          # Multi-stage build (builder + slim runtime)
│   └── docker-compose.yml  # API + Prometheus services
│
├── tests/
│   ├── test_schemas.py     # Pydantic schema unit tests
│   └── test_api.py         # FastAPI endpoint tests (mocked model)
│
├── .github/
│   └── workflows/
│       ├── ci.yml          # Lint (ruff) + test (pytest) on push and PRs
│       └── docker.yml      # Build and push to GHCR on master and version tags
│
├── requirements.txt
├── .gitignore
└── .dockerignore
```

## Setup

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Training

```bash
# Uses configs/train.yaml by default
python -m src.train

# Override individual fields
python -m src.train --epochs 50 --lr 3e-4 --output_dir outputs/run1
```

Expected CSV format (`data/train.csv`, `data/val.csv`):

```
image_path,label
images/ISIC_0024306.jpg,0
images/ISIC_0024307.jpg,1
```

## Evaluation

```bash
python -m src.evaluate \
  --csv data/val.csv \
  --checkpoint outputs/best.pth
```

## Serving

```bash
# Local
uvicorn serving.app:app --reload

# Docker
docker compose -f docker/docker-compose.yml up --build
```

Endpoints:

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Model status and version |
| POST | `/predict` | Upload JPEG/PNG, returns class + probabilities |
| GET | `/metrics` | Prometheus metrics |

## Monitoring

Prometheus scrapes `/metrics` every 15 seconds. Available metrics:

| Metric | Type | Description |
|--------|------|-------------|
| `skin_lesion_predictions_total` | Counter | Predictions per class |
| `skin_lesion_prediction_confidence` | Histogram | Confidence score distribution |
| `skin_lesion_inference_latency_seconds` | Histogram | Forward pass latency |
| `skin_lesion_requests_total` | Counter | Requests by endpoint and status |
| `skin_lesion_model_loaded` | Gauge | Model loaded (1) or not (0) |

Prometheus UI available at `http://localhost:9090` when running via Docker Compose.

## Example Prediction

Tested against [ISIC_0000000](https://api.isic-archive.com/api/v2/images/?isic_id=ISIC_0000000) — a dermoscopic image clinically diagnosed as **Nevus (mole)**:

```bash
curl -X POST http://localhost:8080/predict \
  -F "file=@ISIC_0000000.jpg"
```

```json
{
  "predicted_class": "Nevus (mole)",
  "confidence": 0.715271,
  "probabilities": {
    "Melanoma": 0.204866,
    "Nevus (mole)": 0.715271,
    "Basal Cell Carcinoma": 0.030693,
    "Actinic Keratosis": 0.009799,
    "Benign Keratosis": 0.010218,
    "Dermatofibroma": 0.012197,
    "Vascular Lesion": 0.014393,
    "Squamous Cell Carcinoma": 0.002562
  },
  "model_version": "1.0.0"
}
```

Prediction matches ground truth label. ✓

## Tests

```bash
pytest tests/ -v
```
