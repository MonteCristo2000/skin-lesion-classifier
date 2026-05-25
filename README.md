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

## Tests

```bash
pytest tests/ -v
```
