"""
Database Models
SQLAlchemy ORM models for audit trail, predictions, and model metadata.
"""

from sqlalchemy import (
    Column, Integer, String, Float, DateTime, Boolean,
    Text, ForeignKey, Enum as SQLEnum, JSON
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
import enum
import json

Base = declarative_base()


class PredictionRecord(Base):
    """
    Audit trail: stores every prediction made.
    Used for:
      - Model drift detection (compare prediction distributions over time)
      - Ground truth comparison (when actual label is available)
      - Model rollback analysis
      - Feature importance tracking per claim type
    """
    __tablename__ = "prediction_records"

    id = Column(Integer, primary_key=True, index=True)
    prediction_id = Column(String(255), unique=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)

    # Request Info
    claim_id = Column(String(255), index=True)
    patient_id = Column(String(255), nullable=True)
    doctor_id = Column(String(255), nullable=True)
    hospital_id = Column(String(255), nullable=True)

    # Features (JSON for flexibility)
    features = Column(JSON)
    claim_amount = Column(Float)

    # Prediction Output
    model_version = Column(String(50), index=True)  # e.g., "gnn_v1.2.3"
    fraud_score = Column(Float)  # [0, 1]
    fraud_prediction = Column(Boolean)  # True = fraud, False = legit
    confidence = Column(Float)  # Model confidence

    # Explainability
    top_features = Column(JSON)  # [{feature: str, importance: float}, ...]
    shap_values = Column(JSON, nullable=True)  # Optional detailed explanation

    # Ground Truth (populated later by feedback system)
    actual_label = Column(Boolean, nullable=True)
    label_timestamp = Column(DateTime, nullable=True)

    # Metadata
    inference_time_ms = Column(Float)
    api_endpoint = Column(String(255))
    client_ip = Column(String(45), nullable=True)  # IPv4 + IPv6 support

    def to_dict(self):
        return {
            "id": self.id,
            "prediction_id": self.prediction_id,
            "timestamp": self.timestamp.isoformat(),
            "claim_id": self.claim_id,
            "model_version": self.model_version,
            "fraud_score": self.fraud_score,
            "fraud_prediction": self.fraud_prediction,
            "confidence": self.confidence,
            "inference_time_ms": self.inference_time_ms,
        }


class ModelVersion(Base):
    """
    Model registry: stores metadata for all trained model versions.
    Enables:
      - Model versioning + switching
      - Performance comparison
      - A/B testing
      - Rollback tracking
    """
    __tablename__ = "model_versions"

    id = Column(Integer, primary_key=True, index=True)
    version = Column(String(50), unique=True, index=True)  # e.g., "gnn_v1.2.3"
    model_type = Column(String(50))  # "gnn", "random_forest", "ensemble"
    
    # Performance Metrics
    f1_score = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    roc_auc = Column(Float)
    accuracy = Column(Float)

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    trained_on_samples = Column(Integer)  # Number of training samples
    feature_hash = Column(String(64))  # SHA256 hash of feature set used
    
    # Status
    is_active = Column(Boolean, default=False)
    promoted_at = Column(DateTime, nullable=True)
    demoted_at = Column(DateTime, nullable=True)

    # Drift Metrics
    last_drift_check = Column(DateTime, nullable=True)
    drift_score_kl = Column(Float, nullable=True)
    drift_score_js = Column(Float, nullable=True)
    
    # Metadata
    description = Column(Text, nullable=True)
    model_path = Column(String(512))  # Path to model artifacts (.pkl/.pt)
    training_config = Column(JSON)  # Hyperparameters used

    def to_dict(self):
        return {
            "version": self.version,
            "model_type": self.model_type,
            "f1_score": self.f1_score,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat(),
            "drift_score_kl": self.drift_score_kl,
        }


class DataDriftAlert(Base):
    """
    Tracks data/concept drift detections.
    Triggers:
      - Automated retraining
      - Notifications
      - Model rollback if necessary
    """
    __tablename__ = "data_drift_alerts"

    id = Column(Integer, primary_key=True, index=True)
    alert_timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Detection Details
    drift_type = Column(String(50))  # "data_drift", "concept_drift", "data_quality"
    drift_metric = Column(String(100))  # "kl_divergence", "js_divergence", "distribution_shift"
    drift_score = Column(Float)
    threshold = Column(Float)
    
    # Affected Features
    affected_features = Column(JSON)  # List of feature names
    
    # Action Taken
    action_taken = Column(String(100))  # "retrain_triggered", "alert_sent", "rollback"
    associated_model_version = Column(String(50), ForeignKey("model_versions.version"), nullable=True)
    
    # Resolution
    resolved = Column(Boolean, default=False)
    resolved_at = Column(DateTime, nullable=True)
    resolution_notes = Column(Text, nullable=True)

    def to_dict(self):
        return {
            "id": self.id,
            "alert_timestamp": self.alert_timestamp.isoformat(),
            "drift_type": self.drift_type,
            "drift_score": self.drift_score,
            "resolved": self.resolved,
        }


class FeatureSchema(Base):
    """
    Stores expected feature schema + statistics for validation.
    Used for:
      - Input validation (check incoming data matches schema)
      - Data quality checks
      - Outlier detection
      - Schema evolution tracking
    """
    __tablename__ = "feature_schemas"

    id = Column(Integer, primary_key=True, index=True)
    version = Column(String(50), unique=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Schema Definition
    feature_names = Column(JSON)  # List of expected feature names
    feature_types = Column(JSON)  # {feature: "float", "int", "bool", ...}
    
    # Statistics for Validation
    feature_stats = Column(JSON)  # {feature: {min, max, mean, std, null_pct}}
    
    # Validity Range (for outlier detection)
    valid_ranges = Column(JSON)  # {feature: {min_valid, max_valid}}

    def to_dict(self):
        return {
            "version": self.version,
            "feature_names": self.feature_names,
            "created_at": self.created_at.isoformat(),
        }
