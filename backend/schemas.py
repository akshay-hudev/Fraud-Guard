"""
Pydantic Request/Response Models
Comprehensive input validation for all API endpoints.
"""

from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class PredictionRequest(BaseModel):
    """Single claim prediction request."""
    
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "claim_id": "CLM_2024_001",
            "patient_id": "PAT_001",
            "doctor_id": "DOC_001",
            "hospital_id": "HOS_001",
            "claim_amount": 15000.50,
            "num_procedures": 3,
            "days_in_hospital": 2,
            "age": 45,
            "gender": "M",
            "insurance_type": "PPO",
            "specialty": "Cardiology",
            "explain": True,
        }
    })

    claim_id: str = Field(..., min_length=1, description="Unique claim identifier")
    patient_id: Optional[str] = Field(None, description="Patient ID")
    doctor_id: Optional[str] = Field(None, description="Doctor ID")
    hospital_id: Optional[str] = Field(None, description="Hospital ID")
    
    claim_amount: float = Field(..., ge=0, description="Claim amount in USD")
    num_procedures: int = Field(1, ge=0, le=100, description="Number of procedures")
    days_in_hospital: int = Field(0, ge=0, le=365, description="Days hospitalized")
    
    age: Optional[int] = Field(None, ge=0, le=150, description="Patient age")
    gender: Optional[str] = Field(None, description="Gender: M/F")
    insurance_type: Optional[str] = Field(None, description="Insurance type")
    specialty: Optional[str] = Field(None, description="Medical specialty")
    chronic_condition: Optional[str] = Field(None, description="Chronic condition")
    
    explain: bool = Field(False, description="Include SHAP explanation")

    @field_validator("claim_amount")
    @classmethod
    def validate_amount(cls, v):
        """Ensure claim amount is reasonable (basic sanity check)."""
        if v > 10_000_000:  # > $10M is likely invalid
            raise ValueError("Claim amount suspiciously high (> $10M)")
        return v


class BatchPredictionRequest(BaseModel):
    """Batch prediction request."""
    
    claims: List[PredictionRequest] = Field(..., max_items=1000, description="List of claims")
    explain: bool = Field(False, description="Include explanations for all")
    model_version: Optional[str] = Field(None, description="Specific model version")

    @field_validator("claims")
    @classmethod
    def validate_batch_size(cls, v):
        """Limit batch size to prevent DoS."""
        if len(v) > 1000:
            raise ValueError("Batch size limited to 1000 claims")
        return v


class PredictionResponse(BaseModel):
    """Single prediction response."""
    
    prediction_id: str = Field(..., description="Unique prediction ID (audit trail)")
    fraud_score: float = Field(..., ge=0, le=1, description="Fraud probability [0, 1]")
    fraud_prediction: bool = Field(..., description="Fraud (True) or Legit (False)")
    confidence: float = Field(..., ge=0, le=1, description="Model confidence")
    model_version: str = Field(..., description="Model version used")
    inference_time_ms: float = Field(..., description="Inference latency (milliseconds)")
    
    top_features: Optional[List[Dict[str, Any]]] = Field(None, description="Top contributing features (SHAP)")
    shap_values: Optional[List[float]] = Field(None, description="SHAP values for top features")
    
    error: Optional[str] = Field(None, description="Error message if prediction failed")


class BatchPredictionResponse(BaseModel):
    """Batch prediction response."""
    
    predictions: List[PredictionResponse] = Field(..., description="List of predictions")
    total_processed: int = Field(..., description="Total claims processed")
    successful: int = Field(..., description="Successfully predicted")
    failed: int = Field(..., description="Failed predictions")
    average_fraud_score: float = Field(..., ge=0, le=1, description="Average fraud score")
    fraud_count: int = Field(..., ge=0, description="Number of fraud predictions")


class ModelVersionInfo(BaseModel):
    """Model version metadata."""
    
    version: str = Field(..., description="Version string (e.g., gnn_v1.2.3)")
    model_type: str = Field(..., description="Model type: gnn, random_forest, ensemble")
    f1_score: float = Field(..., ge=0, le=1, description="F1 score on test set")
    precision: float = Field(..., ge=0, le=1)
    recall: float = Field(..., ge=0, le=1)
    roc_auc: float = Field(..., ge=0, le=1)
    accuracy: float = Field(..., ge=0, le=1)
    is_active: bool = Field(..., description="Currently active for predictions")
    created_at: datetime = Field(..., description="When model was trained")
    drift_score_kl: Optional[float] = Field(None, description="KL-divergence drift metric")


class HealthCheckResponse(BaseModel):
    """API health check response."""
    
    status: str = Field(..., description="health, degraded, unhealthy")
    models_available: bool
    database_available: bool
    active_model_version: Optional[str] = None
    uptime_seconds: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class DriftAlert(BaseModel):
    """Data drift alert."""
    
    alert_id: int
    detected_at: datetime
    drift_type: str  # data_drift, concept_drift
    drift_score: float
    affected_features: List[str]
    threshold: float
    resolved: bool


class APIErrorResponse(BaseModel):
    """Standard error response."""
    
    error: str = Field(..., description="Error message")
    status_code: int = Field(..., description="HTTP status code")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    request_id: Optional[str] = Field(None, description="For debugging/tracing")


class TokenRequest(BaseModel):
    """Token request for authentication."""
    
    api_key: str = Field(..., description="API key for authentication")


class TokenResponse(BaseModel):
    """JWT token response."""
    
    access_token: str = Field(..., description="JWT access token")
    token_type: str = Field(default="bearer", description="Token type")
    expires_in: int = Field(..., description="Token expiration time in seconds")
