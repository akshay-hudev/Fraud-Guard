"""
Monitoring Module — Prometheus Metrics & Observability
Tracks model performance, API metrics, and drift indicators.
"""

from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry
import time
from datetime import datetime
import logging

log = logging.getLogger(__name__)

# ── Registry ──────────────────────────────────────────────────────────────────
registry = CollectorRegistry()

# ── Prediction Metrics ────────────────────────────────────────────────────────

predictions_total = Counter(
    'fraud_predictions_total',
    'Total fraud predictions made',
    ['model', 'prediction'],
    registry=registry
)

prediction_latency = Histogram(
    'fraud_prediction_latency_ms',
    'Prediction latency in milliseconds',
    ['model'],
    buckets=(10, 25, 50, 100, 250, 500, 1000),
    registry=registry
)

fraud_scores = Histogram(
    'fraud_prediction_scores',
    'Distribution of fraud scores predicted',
    ['model'],
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
    registry=registry
)

# ── API Metrics ───────────────────────────────────────────────────────────────

api_requests_total = Counter(
    'api_requests_total',
    'Total API requests',
    ['endpoint', 'method', 'status_code'],
    registry=registry
)

api_request_latency = Histogram(
    'api_request_latency_ms',
    'API request latency in milliseconds',
    ['endpoint', 'method'],
    buckets=(10, 50, 100, 250, 500, 1000, 2500),
    registry=registry
)

# ── Model Metrics ────────────────────────────────────────────────────────────

model_accuracy = Gauge(
    'model_accuracy_last_batch',
    'Model accuracy on last evaluation batch',
    ['model'],
    registry=registry
)

model_precision = Gauge(
    'model_precision_last_batch',
    'Model precision on last evaluation batch',
    ['model'],
    registry=registry
)

model_recall = Gauge(
    'model_recall_last_batch',
    'Model recall on last evaluation batch',
    ['model'],
    registry=registry
)

model_roc_auc = Gauge(
    'model_roc_auc_last_batch',
    'Model ROC-AUC on last evaluation batch',
    ['model'],
    registry=registry
)

# ── Error Metrics ────────────────────────────────────────────────────────────

prediction_errors = Counter(
    'fraud_prediction_errors_total',
    'Total prediction errors',
    ['model', 'error_type'],
    registry=registry
)

api_errors = Counter(
    'api_errors_total',
    'Total API errors',
    ['endpoint', 'error_type'],
    registry=registry
)

# ── Health Metrics ───────────────────────────────────────────────────────────

model_loaded = Gauge(
    'model_loaded',
    'Whether model is loaded (1=yes, 0=no)',
    ['model'],
    registry=registry
)

database_connected = Gauge(
    'database_connected',
    'Whether database is connected (1=yes, 0=no)',
    registry=registry
)

# ── Drift Metrics ────────────────────────────────────────────────────────────

fraud_score_mean = Gauge(
    'fraud_score_mean',
    'Mean fraud score (for drift detection)',
    ['model'],
    registry=registry
)

fraud_score_std = Gauge(
    'fraud_score_std',
    'Std dev of fraud scores (for drift detection)',
    ['model'],
    registry=registry
)

fraud_rate = Gauge(
    'fraud_rate_last_hour',
    'Fraud rate in last hour (model predictions)',
    ['model'],
    registry=registry
)


# ── Helper Functions ──────────────────────────────────────────────────────────

class MetricsCollector:
    """Helper class for recording metrics in context managers."""
    
    @staticmethod
    def record_prediction(model_name: str, fraud_score: float, predicted_fraud: bool, latency_ms: float):
        """Record a prediction with all metrics."""
        try:
            prediction = "fraud" if predicted_fraud else "legit"
            predictions_total.labels(model=model_name, prediction=prediction).inc()
            prediction_latency.labels(model=model_name).observe(latency_ms)
            fraud_scores.labels(model=model_name).observe(fraud_score)
        except Exception as e:
            log.error(f"Error recording prediction metrics: {e}")
    
    @staticmethod
    def record_api_request(endpoint: str, method: str, status_code: int, latency_ms: float):
        """Record an API request."""
        try:
            api_requests_total.labels(endpoint=endpoint, method=method, status_code=status_code).inc()
            api_request_latency.labels(endpoint=endpoint, method=method).observe(latency_ms)
        except Exception as e:
            log.error(f"Error recording API request metrics: {e}")
    
    @staticmethod
    def record_error(error_type: str, context: str = "prediction"):
        """Record an error."""
        try:
            if context == "prediction":
                prediction_errors.labels(model="gnn", error_type=error_type).inc()
            else:
                api_errors.labels(endpoint="unknown", error_type=error_type).inc()
        except Exception as e:
            log.error(f"Error recording error metrics: {e}")
    
    @staticmethod
    def set_model_metrics(model_name: str, accuracy: float, precision: float, recall: float, roc_auc: float):
        """Set model performance metrics."""
        try:
            model_accuracy.labels(model=model_name).set(accuracy)
            model_precision.labels(model=model_name).set(precision)
            model_recall.labels(model=model_name).set(recall)
            model_roc_auc.labels(model=model_name).set(roc_auc)
        except Exception as e:
            log.error(f"Error recording model metrics: {e}")
    
    @staticmethod
    def set_drift_metrics(model_name: str, mean: float, std: float, fraud_rate_pct: float):
        """Set drift detection metrics."""
        try:
            fraud_score_mean.labels(model=model_name).set(mean)
            fraud_score_std.labels(model=model_name).set(std)
            fraud_rate.labels(model=model_name).set(fraud_rate_pct)
        except Exception as e:
            log.error(f"Error recording drift metrics: {e}")
    
    @staticmethod
    def set_health_status(model_loaded_: bool, db_connected: bool):
        """Set system health metrics."""
        try:
            model_loaded.labels(model="gnn").set(1 if model_loaded_ else 0)
            database_connected.set(1 if db_connected else 0)
        except Exception as e:
            log.error(f"Error recording health metrics: {e}")


# ── Timing Context Manager ────────────────────────────────────────────────────

class TimingContext:
    """Context manager to measure execution time."""
    
    def __init__(self, metric: Histogram, labels: dict):
        self.metric = metric
        self.labels = labels
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed_ms = (time.time() - self.start_time) * 1000
        self.metric.labels(**self.labels).observe(elapsed_ms)
