import logging
import threading
from collections import deque

import numpy as np
from scipy.stats import ks_2samp

logger = logging.getLogger(__name__)


class DriftDetector:
    """
    Detects confidence distribution shift between a reference set (e.g. validation
    confidences) and a rolling window of live predictions using the KS test.

    Once the window fills, every new prediction triggers a KS test.  A warning is
    logged whenever p_value < p_threshold.  The window slides: the oldest observation
    is dropped when a new one arrives.
    """

    def __init__(
        self,
        reference: list[float],
        window_size: int = 100,
        p_threshold: float = 0.01,
    ) -> None:
        if reference and len(reference) < 2:
            raise ValueError("reference must contain at least 2 values (or be empty to disable drift detection)")
        self._reference = np.array(reference, dtype=np.float64)
        self._window_size = window_size
        self._p_threshold = p_threshold
        self._window: deque[float] = deque(maxlen=window_size)
        self._lock = threading.Lock()
        self._drift_count = 0

    @property
    def is_active(self) -> bool:
        """False when no reference data was provided; update() becomes a no-op."""
        return len(self._reference) >= 2

    @property
    def drift_count(self) -> int:
        with self._lock:
            return self._drift_count

    def update(self, confidence: float) -> bool:
        """
        Record a new confidence score.

        Returns True if drift was detected on this call, False otherwise.
        Returns False immediately if the detector has no reference data or
        the rolling window is not yet full.
        """
        if not self.is_active:
            return False

        with self._lock:
            self._window.append(confidence)
            if len(self._window) < self._window_size:
                return False

            stat, p_value = ks_2samp(self._reference, list(self._window))
            if p_value < self._p_threshold:
                self._drift_count += 1
                logger.warning(
                    "Confidence drift detected — KS statistic=%.4f  p-value=%.6f  "
                    "(threshold=%.3f, window=%d, drift_events=%d)",
                    stat,
                    p_value,
                    self._p_threshold,
                    self._window_size,
                    self._drift_count,
                )
                return True

            return False
