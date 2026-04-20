"""
Structured Logging Module
JSON-based structured logging for production observability.
"""

import logging
import json
import sys
import os
from datetime import datetime
from typing import Any, Optional, Dict
from pythonjsonlogger import jsonlogger
from backend.config import get_settings

settings = get_settings()


class StructuredLogFormatter(jsonlogger.JsonFormatter):
    """Custom JSON formatter for structured logging."""

    def add_fields(self, log_record: Dict, record: logging.LogRecord, message_dict: Dict) -> None:
        super().add_fields(log_record, record, message_dict)
        
        # Add custom fields
        log_record["timestamp"] = datetime.utcnow().isoformat()
        log_record["level"] = record.levelname
        log_record["logger"] = record.name
        log_record["module"] = record.module
        
        # Add exception info if present
        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)


def setup_logging():
    """Configure structured logging for the application."""
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, settings.log_level))

    # Console handler (JSON format)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(
        StructuredLogFormatter(
            fmt="%(timestamp)s %(level)s %(name)s %(message)s",
            defaults={"environment": settings.environment}
        )
    )
    root_logger.addHandler(console_handler)

    # File handler (if configured)
    if settings.log_file:
        # Create logs directory if it doesn't exist
        log_dir = os.path.dirname(settings.log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        file_handler = logging.FileHandler(settings.log_file)
        file_handler.setFormatter(
            StructuredLogFormatter(
                fmt="%(timestamp)s %(level)s %(name)s %(message)s",
                defaults={"environment": settings.environment}
            )
        )
        root_logger.addHandler(file_handler)


class AppLogger:
    """Application-level structured logger."""

    def __init__(self, name: str):
        self.logger = logging.getLogger(name)

    def info(self, message: str, **kwargs):
        """Log info level."""
        self.logger.info(message, extra=kwargs)

    def error(self, message: str, exception: Optional[Exception] = None, **kwargs):
        """Log error level."""
        if exception:
            kwargs["exception_type"] = type(exception).__name__
            kwargs["exception_message"] = str(exception)
        self.logger.error(message, extra=kwargs)

    def warning(self, message: str, **kwargs):
        """Log warning level."""
        self.logger.warning(message, extra=kwargs)

    def debug(self, message: str, **kwargs):
        """Log debug level."""
        self.logger.debug(message, extra=kwargs)

    def audit_prediction(
        self,
        prediction_id: str,
        claim_id: str,
        model_version: str,
        fraud_score: float,
        inference_time_ms: float,
        **kwargs
    ):
        """Log prediction with audit trail."""
        self.logger.info(
            "Prediction made",
            extra={
                "event_type": "prediction",
                "prediction_id": prediction_id,
                "claim_id": claim_id,
                "model_version": model_version,
                "fraud_score": fraud_score,
                "inference_time_ms": inference_time_ms,
                **kwargs,
            }
        )

    def audit_drift_detected(
        self,
        drift_type: str,
        drift_score: float,
        threshold: float,
        **kwargs
    ):
        """Log drift detection event."""
        self.logger.warning(
            "Data drift detected",
            extra={
                "event_type": "drift_detected",
                "drift_type": drift_type,
                "drift_score": drift_score,
                "threshold": threshold,
                **kwargs,
            }
        )


# Module-level logger
logger = AppLogger(__name__)
