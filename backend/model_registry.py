"""
Model Registry & Versioning
Manages model versions, enables switching, rollback, and A/B testing.
"""

from typing import Optional, List, Dict, Any
from datetime import datetime
from sqlalchemy.orm import Session
from backend.database.models import ModelVersion
import logging

logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Central model registry for version management.
    Features:
      - Register new trained models
      - Promote/demote versions
      - Retrieve model metadata
      - Track performance metrics
      - Support A/B testing
    """

    def __init__(self, db: Session):
        self.db = db

    def register_model(
        self,
        version: str,
        model_type: str,
        model_path: str,
        metrics: Dict[str, float],
        feature_hash: str,
        trained_on_samples: int,
        training_config: Dict[str, Any],
        description: Optional[str] = None,
    ) -> ModelVersion:
        """Register a newly trained model."""
        model_record = ModelVersion(
            version=version,
            model_type=model_type,
            model_path=model_path,
            f1_score=metrics.get("f1", 0.0),
            precision=metrics.get("precision", 0.0),
            recall=metrics.get("recall", 0.0),
            roc_auc=metrics.get("roc_auc", 0.0),
            accuracy=metrics.get("accuracy", 0.0),
            feature_hash=feature_hash,
            trained_on_samples=trained_on_samples,
            training_config=training_config,
            description=description,
            is_active=False,  # Must explicitly promote
        )
        self.db.add(model_record)
        self.db.commit()
        self.db.refresh(model_record)
        logger.info(f"Model registered: {version} (F1={metrics.get('f1')})")
        return model_record

    def promote_model(self, version: str) -> ModelVersion:
        """Promote a model to active (production)."""
        # Demote current active model
        active = self.db.query(ModelVersion).filter(ModelVersion.is_active == True).first()
        if active:
            active.is_active = False
            active.demoted_at = datetime.utcnow()
            logger.info(f"Model demoted: {active.version}")

        # Promote new model
        model = self.db.query(ModelVersion).filter(ModelVersion.version == version).first()
        if not model:
            raise ValueError(f"Model version {version} not found")
        
        model.is_active = True
        model.promoted_at = datetime.utcnow()
        self.db.commit()
        logger.info(f"Model promoted: {version}")
        return model

    def get_active_model(self) -> Optional[ModelVersion]:
        """Get currently active model."""
        return self.db.query(ModelVersion).filter(ModelVersion.is_active == True).first()

    def get_model_by_version(self, version: str) -> Optional[ModelVersion]:
        """Retrieve model by version string."""
        return self.db.query(ModelVersion).filter(ModelVersion.version == version).first()

    def list_models(self, limit: int = 10) -> List[ModelVersion]:
        """List all registered models (newest first)."""
        return self.db.query(ModelVersion).order_by(
            ModelVersion.created_at.desc()
        ).limit(limit).all()

    def compare_models(self, version1: str, version2: str) -> Dict[str, Any]:
        """Compare metrics between two model versions for A/B testing."""
        m1 = self.get_model_by_version(version1)
        m2 = self.get_model_by_version(version2)
        
        if not m1 or not m2:
            raise ValueError("One or both model versions not found")
        
        return {
            "version1": {
                "version": m1.version,
                "f1": m1.f1_score,
                "precision": m1.precision,
                "recall": m1.recall,
                "roc_auc": m1.roc_auc,
                "created_at": m1.created_at.isoformat(),
            },
            "version2": {
                "version": m2.version,
                "f1": m2.f1_score,
                "precision": m2.precision,
                "recall": m2.recall,
                "roc_auc": m2.roc_auc,
                "created_at": m2.created_at.isoformat(),
            },
            "delta_f1": round(m2.f1_score - m1.f1_score, 4),
            "better_model": m2.version if m2.f1_score > m1.f1_score else m1.version,
        }

    def record_drift_check(
        self,
        version: str,
        drift_score_kl: float,
        drift_score_js: float,
    ) -> ModelVersion:
        """Record drift detection metrics."""
        model = self.get_model_by_version(version)
        if not model:
            raise ValueError(f"Model version {version} not found")
        
        model.last_drift_check = datetime.utcnow()
        model.drift_score_kl = drift_score_kl
        model.drift_score_js = drift_score_js
        self.db.commit()
        logger.info(f"Drift recorded for {version}: KL={drift_score_kl:.4f}, JS={drift_score_js:.4f}")
        return model
