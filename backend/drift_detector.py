"""
Drift Detection Module
Statistical methods to detect data/concept drift.
"""

import numpy as np
from scipy.special import rel_entr
from scipy.stats import entropy
from typing import Dict, Tuple, List, Optional
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from backend.database.models import PredictionRecord, DataDriftAlert
import logging

logger = logging.getLogger(__name__)


class DriftDetector:
    """
    Detects data and concept drift using statistical methods.
    
    Methods:
      - KL Divergence: compares prediction distribution shift
      - JS Divergence: symmetric alternative to KL
      - Kolmogorov-Smirnov test: compares numerical feature distributions
    """

    def __init__(self, db: Session, threshold_kl: float = 0.15, threshold_js: float = 0.10):
        self.db = db
        self.threshold_kl = threshold_kl
        self.threshold_js = threshold_js

    def get_recent_predictions(self, days: int = 7) -> List[PredictionRecord]:
        """Fetch recent predictions for drift analysis."""
        cutoff = datetime.utcnow() - timedelta(days=days)
        return self.db.query(PredictionRecord).filter(
            PredictionRecord.timestamp >= cutoff
        ).all()

    def compute_prediction_distribution(
        self, 
        predictions: List[PredictionRecord],
        bins: int = 10
    ) -> np.ndarray:
        """
        Convert continuous fraud scores to a probability distribution.
        Returns histogram normalized to [0, 1].
        """
        scores = np.array([p.fraud_score for p in predictions])
        hist, _ = np.histogram(scores, bins=bins, range=(0, 1))
        return hist / hist.sum()

    def kl_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        """
        KL divergence: D_KL(P || Q)
        Measures how much distribution Q diverges from P.
        - 0 = identical distributions
        - Higher = more divergence
        """
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        p = p + epsilon
        q = q + epsilon
        p = p / p.sum()
        q = q / q.sum()
        return np.sum(p * (np.log(p) - np.log(q)))

    def js_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        """
        Jensen-Shannon divergence: symmetric version of KL.
        - Always between 0 and log(2)
        - Symmetric: JS(P,Q) = JS(Q,P)
        """
        epsilon = 1e-10
        p = p + epsilon
        q = q + epsilon
        p = p / p.sum()
        q = q / q.sum()
        m = 0.5 * (p + q)
        return 0.5 * self.kl_divergence(p, m) + 0.5 * self.kl_divergence(q, m)

    def check_prediction_drift(
        self, 
        baseline_predictions: List[PredictionRecord],
        current_predictions: List[PredictionRecord],
    ) -> Dict[str, float]:
        """
        Detect drift in prediction distribution.
        Compares baseline period vs current period.
        """
        if len(baseline_predictions) < 10 or len(current_predictions) < 10:
            logger.warning("Insufficient predictions for drift analysis")
            return {"has_drift": False, "reason": "insufficient_data"}

        baseline_dist = self.compute_prediction_distribution(baseline_predictions)
        current_dist = self.compute_prediction_distribution(current_predictions)

        kl_div = self.kl_divergence(baseline_dist, current_dist)
        js_div = self.js_divergence(baseline_dist, current_dist)

        has_kl_drift = kl_div > self.threshold_kl
        has_js_drift = js_div > self.threshold_js

        result = {
            "kl_divergence": round(kl_div, 4),
            "js_divergence": round(js_div, 4),
            "has_drift": has_kl_drift or has_js_drift,
            "drift_reason": "kl_divergence" if has_kl_drift else ("js_divergence" if has_js_drift else "none"),
        }

        if result["has_drift"]:
            logger.warning(f"Drift detected: KL={kl_div:.4f}, JS={js_div:.4f}")
            # Record alert
            self._create_drift_alert(result, baseline_predictions, current_predictions)

        return result

    def _create_drift_alert(
        self,
        drift_result: Dict,
        baseline: List[PredictionRecord],
        current: List[PredictionRecord],
    ):
        """Create database alert for drift."""
        alert = DataDriftAlert(
            drift_type="concept_drift",
            drift_metric="prediction_distribution",
            drift_score=drift_result.get("kl_divergence", 0),
            threshold=self.threshold_kl,
            affected_features=["fraud_score"],
            action_taken="alert_sent",
        )
        self.db.add(alert)
        self.db.commit()
        logger.info(f"Drift alert created: {alert.id}")

    def check_feature_drift(
        self,
        baseline_features: np.ndarray,
        current_features: np.ndarray,
        feature_names: List[str],
    ) -> Dict[str, any]:
        """
        Detect drift in input features using statistical tests.
        Checks for significant changes in feature distributions.
        """
        from scipy.stats import ks_2samp

        feature_shifts = {}
        for i, name in enumerate(feature_names):
            if len(baseline_features) > 5 and len(current_features) > 5:
                stat, pvalue = ks_2samp(baseline_features[:, i], current_features[:, i])
                feature_shifts[name] = {"ks_statistic": stat, "pvalue": pvalue}

        return feature_shifts
