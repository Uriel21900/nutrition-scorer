"""
Structured JSON Logger & Prediction Telemetry Tracker
Designed for Cloud Logging (Google Cloud Logging / AWS CloudWatch / Azure Monitor)
"""

import json
import logging
import time
from typing import Any, Dict, Optional


class JSONFormatter(logging.Formatter):
    """
    Formats log records as structured JSON for easy parsing by cloud telemetry tools.
    """

    def format(self, record: logging.LogRecord) -> str:
        log_payload: Dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "severity": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Include custom telemetry fields if present in record attributes
        for field in [
            "event_type",
            "model_version",
            "latency_ms",
            "prediction",
            "grade",
            "status_code",
            "path",
            "client_ip",
            "barcode",
        ]:
            if hasattr(record, field):
                log_payload[field] = getattr(record, field)

        if record.exc_info:
            log_payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_payload, ensure_ascii=False)


def get_logger(name: str = "NutriScoreTelemetry") -> logging.Logger:
    """
    Returns a configured structured JSON logger.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
        logger.propagate = False
    return logger


class TelemetryTracker:
    """
    Tracks prediction requests, latency percentiles, grade distributions, and drift alerts.
    In-memory metrics store for /api/v1/metrics endpoint.
    """

    def __init__(self):
        self.total_requests: int = 0
        self.error_count: int = 0
        self.latencies_ms: list[float] = []
        self.grade_counts: Dict[str, int] = {"A": 0, "B": 0, "C": 0, "D": 0, "E": 0}
        self.model_version: str = "NutriScore-StackingRegressor-v1.0"
        self.logger = get_logger("NutriScoreTelemetry")

    def log_request(
        self,
        path: str,
        status_code: int,
        latency_ms: float,
        prediction: Optional[float] = None,
        grade: Optional[str] = None,
        barcode: Optional[str] = None,
    ) -> None:
        """
        Records request telemetry and updates aggregation metrics.
        """
        self.total_requests += 1
        self.latencies_ms.append(latency_ms)
        if len(self.latencies_ms) > 10000:
            self.latencies_ms.pop(0)

        if status_code >= 400:
            self.error_count += 1

        if grade and grade in self.grade_counts:
            self.grade_counts[grade] += 1

        extra_fields = {
            "event_type": "api_request",
            "path": path,
            "status_code": status_code,
            "latency_ms": round(latency_ms, 2),
            "model_version": self.model_version,
        }
        if prediction is not None:
            extra_fields["prediction"] = round(prediction, 2)
        if grade:
            extra_fields["grade"] = grade
        if barcode:
            extra_fields["barcode"] = barcode

        # Log structured record
        record = logging.LogRecord(
            name=self.logger.name,
            level=logging.INFO if status_code < 400 else logging.ERROR,
            pathname=__file__,
            lineno=0,
            msg=f"API Request: {path} [{status_code}] ({round(latency_ms, 2)}ms)",
            args=None,
            exc_info=None,
        )
        for key, val in extra_fields.items():
            setattr(record, key, val)
        self.logger.handle(record)

    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Returns JSON-serializable metrics summary for cloud Prometheus / monitoring scrapes.
        """
        sorted_latencies = sorted(self.latencies_ms)
        n = len(sorted_latencies)

        p50 = round(sorted_latencies[int(n * 0.50)], 2) if n > 0 else 0.0
        p95 = round(sorted_latencies[int(n * 0.95)], 2) if n > 0 else 0.0
        p99 = round(sorted_latencies[int(n * 0.99)], 2) if n > 0 else 0.0

        return {
            "model_version": self.model_version,
            "total_requests": self.total_requests,
            "error_count": self.error_count,
            "error_rate_pct": round(
                (self.error_count / self.total_requests) * 100, 2
            )
            if self.total_requests > 0
            else 0.0,
            "latency_ms": {
                "p50": p50,
                "p95": p95,
                "p99": p99,
                "mean": round(sum(sorted_latencies) / n, 2) if n > 0 else 0.0,
            },
            "grade_distribution": self.grade_counts,
        }


# Singleton global telemetry instance
telemetry = TelemetryTracker()
