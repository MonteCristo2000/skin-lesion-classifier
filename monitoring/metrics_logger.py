import threading
from collections import defaultdict
from typing import Any


class MetricsLogger:
    """In-process tracker for production request metrics.

    Thread-safe. Values reset on process restart (no persistence).
    For durable storage use the Prometheus counters in metrics.py.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._total_requests: int = 0
        self._class_counts: dict[str, int] = defaultdict(int)
        self._confidence_sum: float = 0.0
        self._latency_sum: float = 0.0
        self._prediction_count: int = 0

    def record_prediction(
        self,
        predicted_class: str,
        confidence: float,
        latency_seconds: float,
    ) -> None:
        with self._lock:
            self._total_requests += 1
            self._prediction_count += 1
            self._class_counts[predicted_class] += 1
            self._confidence_sum += confidence
            self._latency_sum += latency_seconds

    def get_summary(self) -> dict[str, Any]:
        with self._lock:
            n = self._prediction_count
            return {
                "total_requests": self._total_requests,
                "total_predictions": n,
                "per_class_counts": dict(self._class_counts),
                "average_confidence": round(self._confidence_sum / n, 4) if n else None,
                "average_latency_seconds": round(self._latency_sum / n, 6) if n else None,
            }
