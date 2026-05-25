from prometheus_client import Counter, Histogram, Gauge

PREDICTION_COUNTER = Counter(
    "skin_lesion_predictions_total",
    "Total number of predictions by class",
    ["predicted_class"],
)

CONFIDENCE_HISTOGRAM = Histogram(
    "skin_lesion_prediction_confidence",
    "Distribution of prediction confidence scores",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0],
)

INFERENCE_LATENCY = Histogram(
    "skin_lesion_inference_latency_seconds",
    "Inference latency in seconds",
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
)

REQUEST_COUNTER = Counter(
    "skin_lesion_requests_total",
    "Total HTTP requests by endpoint and status",
    ["endpoint", "status"],
)

MODEL_LOADED = Gauge(
    "skin_lesion_model_loaded",
    "Whether the model is currently loaded (1=yes, 0=no)",
)
