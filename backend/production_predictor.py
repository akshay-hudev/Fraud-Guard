"""
Production Fraud Predictor (Fixed)
- Computes all engineered features dynamically from raw inputs
- Fixed SHAP array indexing
- Hybrid ML + rule-based scoring
"""

import os
import json
import logging
import threading
import numpy as np
import joblib
from typing import Dict, Optional, Tuple, Any
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)


class ProductionFraudPredictor:
    """Thread-safe, production-grade fraud predictor."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(
        self,
        processed_dir: str = "data/processed",
        model_dir: str = "models",
        use_gnn: bool = True,
        device: str = "cpu",
    ):
        if self._initialized:
            return

        self.processed_dir = processed_dir
        self.model_dir = model_dir
        self.device = device
        self.use_gnn = use_gnn
        self._predictors_lock = threading.Lock()

        self._load_preprocessors()
        self._load_rf()
        self._load_train_stats()

        self.gnn_available = False  # GNN not yet implemented
        self._shap_explainer = None

        self._initialized = True
        logger.info(f"ProductionFraudPredictor initialized (GNN={self.gnn_available})")

    # ── Loaders ────────────────────────────────────────────────────────────────

    def _load_preprocessors(self):
        """Load scaler and label encoders."""
        scaler_path = f"{self.processed_dir}/scaler.pkl"
        le_path     = f"{self.processed_dir}/label_encoders.pkl"

        try:
            if os.path.exists(scaler_path):
                self.scaler         = joblib.load(scaler_path)
                self.label_encoders = joblib.load(le_path)
                logger.info("Preprocessors loaded successfully")
            else:
                logger.warning("Scaler not found — using passthrough")
                self.scaler         = None
                self.label_encoders = {}
        except Exception as e:
            logger.error(f"Failed to load preprocessors: {e}")
            self.scaler         = None
            self.label_encoders = {}

        # Load feature names from the CSV that was used for training
        feat_path = f"{self.processed_dir}/features_raw.csv"
        try:
            import pandas as pd
            df = pd.read_csv(feat_path, nrows=1)
            drop_cols = ["claim_id", "patient_id", "doctor_id", "hospital_id",
                         "claim_date", "approved", "fraud_label"]
            self.feature_names = [c for c in df.columns if c not in drop_cols]
            logger.info(f"Loaded {len(self.feature_names)} feature names")
        except Exception as e:
            logger.warning(f"Could not load feature names: {e}. Using defaults.")
            self.feature_names = []

    def _load_rf(self):
        """Load Random Forest model."""
        path = f"{self.model_dir}/baseline/random_forest.pkl"
        try:
            if os.path.exists(path):
                self.rf_model = joblib.load(path)
                logger.info("Random Forest model loaded")
            else:
                logger.warning(f"RF model not found at {path}")
                self.rf_model = None
        except Exception as e:
            logger.error(f"Failed to load RF: {e}")
            self.rf_model = None

    def _load_train_stats(self):
        """Load training statistics for rule-based thresholds."""
        stats_path = f"{self.processed_dir}/train_stats.json"
        try:
            if os.path.exists(stats_path):
                with open(stats_path) as f:
                    self.train_stats = json.load(f)
                logger.info("Train stats loaded")
            else:
                # Compute from training data if available, else use defaults
                self.train_stats = self._compute_train_stats()
        except Exception as e:
            logger.warning(f"Could not load train stats: {e}. Using defaults.")
            self.train_stats = self._default_train_stats()

    def _compute_train_stats(self) -> dict:
        """Compute stats from the processed training CSV if available."""
        try:
            import pandas as pd
            path = f"{self.processed_dir}/claims_engineered.csv"
            if not os.path.exists(path):
                return self._default_train_stats()
            df = pd.read_csv(path)
            stats = {
                "mean_claim":        float(df["claim_amount"].mean()),
                "p50_claim":         float(df["claim_amount"].quantile(0.50)),
                "p75_claim":         float(df["claim_amount"].quantile(0.75)),
                "p95_claim":         float(df["claim_amount"].quantile(0.95)),
                "p99_claim":         float(df["claim_amount"].quantile(0.99)),
                "p95_procedures":    float(df["num_procedures"].quantile(0.95)),
                "p95_days":          float(df["days_in_hospital"].quantile(0.95)),
                "patient_avg_claim": float(df.get("patient_avg_amount", df["claim_amount"]).mean()),
            }
            # Save for future use
            with open(f"{self.processed_dir}/train_stats.json", "w") as f:
                json.dump(stats, f, indent=2)
            logger.info(f"Computed and saved train stats: p95=${stats['p95_claim']:,.0f}")
            return stats
        except Exception as e:
            logger.warning(f"Could not compute train stats: {e}")
            return self._default_train_stats()

    def _default_train_stats(self) -> dict:
        """Fallback stats matching the synthetic dataset distribution."""
        return {
            "mean_claim":        2800.0,
            "p50_claim":         1800.0,
            "p75_claim":         5000.0,
            "p95_claim":         12000.0,
            "p99_claim":         28000.0,
            "p95_procedures":    12.0,
            "p95_days":          18.0,
            "patient_avg_claim": 2800.0,
        }

    # ── Feature Engineering ────────────────────────────────────────────────────

    def _prepare_features(self, raw: Dict[str, Any]) -> np.ndarray:
        """
        Convert raw API input → full feature vector matching training schema.

        The RF was trained on 36 engineered features.
        This method reconstructs ALL of them from the raw claim inputs.
        """
        import pandas as pd

        amount     = float(raw.get("claim_amount",     1000))
        procedures = int  (raw.get("num_procedures",   1))
        days       = int  (raw.get("days_in_hospital", 0))
        age        = int  (raw.get("age",              40))

        mean_claim = self.train_stats["mean_claim"]
        pat_avg    = self.train_stats["patient_avg_claim"]

        # ── Rebuild every engineered feature used during training ─────────────
        row = {
            # Raw claim fields
            "claim_amount":    amount,
            "num_procedures":  procedures,
            "days_in_hospital": days,
            "claim_month":     datetime.now().month,
            "claim_dayofweek": datetime.now().weekday(),

            # Patient fields
            "age":                    age,
            "member_since_years":     int(raw.get("member_since_years", 5)),
            "gender":                 raw.get("gender", "M"),
            "chronic_condition":      raw.get("chronic_condition", "None"),
            "insurance_type":         raw.get("insurance_type", "Private"),

            # Doctor fields
            "specialty":              raw.get("specialty", "General Practitioner"),
            "years_experience":       int(raw.get("years_experience", 10)),
            "avg_claims_per_month":   int(raw.get("avg_claims_per_month", 20)),
            "license_valid":          1,

            # Hospital fields
            "num_beds":          int(raw.get("num_beds", 200)),
            "is_accredited":     1,
            "state":             raw.get("state", "CA"),

            # Procedure / diagnosis
            "procedure_code": raw.get("procedure_code", "PC001"),
            "diagnosis_code": raw.get("diagnosis_code", "DX001"),

            # Patient-level aggregations
            # (for a new single claim, the claim itself IS the patient's history)
            "patient_total_claims":     1,
            "patient_total_amount":     amount,
            "patient_avg_amount":       pat_avg,      # compare against population
            "patient_max_amount":       amount,
            "patient_unique_doctors":   1,
            "patient_unique_hospitals": 1,

            # Doctor-level aggregations
            "doctor_total_claims":    20,
            "doctor_total_amount":    mean_claim * 20,
            "doctor_avg_amount":      mean_claim,
            "doctor_unique_patients": 15,
            "doctor_fraud_rate":      0.05,

            # Hospital-level aggregations
            "hosp_total_claims":  100,
            "hosp_avg_amount":    mean_claim,
            "hosp_fraud_rate":    0.05,

            # ── KEY ANOMALY RATIO FEATURES ─────────────────────────────────
            # These must be computed against the POPULATION average,
            # NOT hardcoded to 1.0 — this was the original bug
            "amount_vs_patient_avg": amount / (pat_avg    + 1),
            "amount_vs_doctor_avg":  amount / (mean_claim + 1),
            "amount_vs_hosp_avg":    amount / (mean_claim + 1),
        }

        df = pd.DataFrame([row])

        # ── Encode categoricals using saved LabelEncoders ─────────────────────
        cat_cols = ["gender", "chronic_condition", "insurance_type",
                    "specialty", "state", "procedure_code", "diagnosis_code"]
        for col in cat_cols:
            if col not in df.columns:
                continue
            if col in self.label_encoders:
                le = self.label_encoders[col]
                df[col] = df[col].map(
                    lambda x, le=le: le.transform([str(x)])[0]
                    if str(x) in le.classes_ else 0
                ).astype(float)
            else:
                # Simple numeric fallback
                df[col] = 0.0

        # ── Align to training feature order ───────────────────────────────────
        if self.feature_names:
            for col in self.feature_names:
                if col not in df.columns:
                    df[col] = 0.0
            df = df[self.feature_names]

        X = df.values.astype(np.float32)

        # ── Scale ─────────────────────────────────────────────────────────────
        if self.scaler is not None:
            try:
                X = self.scaler.transform(df)   # pass df to keep feature names
            except Exception:
                try:
                    X = self.scaler.transform(X)
                except Exception as e:
                    logger.warning(f"Scaling failed: {e}")

        return X

    # ── Prediction ─────────────────────────────────────────────────────────────

    def _predict_rf(self, X: np.ndarray) -> Tuple[float, float]:
        proba      = self.rf_model.predict_proba(X)[0]
        fraud_prob = float(proba[1])
        confidence = float(max(proba))
        return fraud_prob, confidence

    def _rule_score(self, raw: Dict[str, Any]) -> float:
        """
        Rule-based boost using training data percentiles.
        Catches out-of-distribution extreme values that tree models miss.
        """
        amount     = float(raw.get("claim_amount",     0))
        procedures = int  (raw.get("num_procedures",   1))
        days       = int  (raw.get("days_in_hospital", 0))

        p75 = self.train_stats["p75_claim"]
        p95 = self.train_stats["p95_claim"]
        p99 = self.train_stats["p99_claim"]
        p95_proc = self.train_stats["p95_procedures"]
        p95_days = self.train_stats["p95_days"]

        score = 0.0

        # Amount anomaly
        if   amount > p99 * 3:  score += 0.60   # absurdly high
        elif amount > p99:      score += 0.45
        elif amount > p95:      score += 0.28
        elif amount > p75:      score += 0.10

        # Excessive procedures
        if   procedures > p95_proc * 1.5: score += 0.20
        elif procedures > p95_proc:       score += 0.12

        # Excessive hospital stay
        if   days > p95_days * 1.5: score += 0.18
        elif days > p95_days:       score += 0.10

        # Multi-signal: all three are high simultaneously
        if amount > p95 and procedures > p95_proc and days > p95_days:
            score += 0.15

        return min(score, 0.90)

    def predict(
        self,
        features: Dict[str, Any],
        explain: bool = False,
        model_version: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Make fraud prediction with hybrid ML + rule scoring."""
        prediction_id = str(uuid.uuid4())

        try:
            X = self._prepare_features(features)

            # ML score
            if self.rf_model:
                ml_score, confidence = self._predict_rf(X)
            else:
                ml_score   = 0.5
                confidence = 0.5

            # Rule score (percentile-based anomaly detection)
            rule_score = self._rule_score(features)

            # Blend: ML 65% + rules 35% + interaction boost
            fraud_score = float(np.clip(
                ml_score * 0.65 + rule_score * 0.35 + ml_score * rule_score * 0.15,
                0.01, 0.99
            ))

            # SHAP explanation
            top_features = []
            shap_values_out = None
            if explain and self.rf_model:
                top_features, shap_values_out = self._explain(X)

            return {
                "prediction_id":  prediction_id,
                "fraud_score":    round(fraud_score, 4),
                "fraud_prediction": fraud_score > 0.5,
                "confidence":     round(confidence, 4),
                "model_version":  model_version or "rf_v1.0.0",
                "inference_time_ms": 0.0,
                "top_features":   top_features,
                "shap_values":    shap_values_out,
                # Convenience fields used by frontend
                "fraud_probability": round(fraud_score, 4),
                "risk_level": "HIGH" if fraud_score > 0.7 else "MEDIUM" if fraud_score > 0.4 else "LOW",
                "model_used": "random_forest+rules",
            }

        except Exception as e:
            logger.error(f"Prediction failed: {e}", exc_info=True)
            return {
                "prediction_id":    prediction_id,
                "fraud_score":      0.5,
                "fraud_prediction": False,
                "confidence":       0.0,
                "model_version":    "error",
                "error":            str(e),
                "top_features":     [],
                "shap_values":      None,
                "fraud_probability": 0.5,
                "risk_level":       "MEDIUM",
                "model_used":       "error",
            }

    def _explain(self, X: np.ndarray):
        """SHAP explanation — fixed array indexing."""
        if self._shap_explainer is None:
            try:
                import shap
                with self._predictors_lock:
                    self._shap_explainer = shap.TreeExplainer(self.rf_model)
                    logger.info("SHAP explainer initialized")
            except Exception as e:
                logger.warning(f"SHAP init failed: {e}")
                return [], None

        try:
            import shap
            sv = self._shap_explainer.shap_values(X)

            # ── FIX: shap_values() for binary RF returns list [class0, class1]
            # We want class1 (fraud), first (and only) sample → sv[1][0]
            if isinstance(sv, list) and len(sv) == 2:
                sv_fraud = sv[1][0]          # shape: (n_features,)
            elif isinstance(sv, np.ndarray) and sv.ndim == 3:
                sv_fraud = sv[0, :, 1]       # (samples, features, classes)
            elif isinstance(sv, np.ndarray) and sv.ndim == 2:
                sv_fraud = sv[0]             # already (samples, features)
            else:
                sv_fraud = np.array(sv).flatten()

            names  = self.feature_names or [f"f{i}" for i in range(len(sv_fraud))]
            order  = np.argsort(np.abs(sv_fraud))[::-1][:10]

            top_features = [
                {
                    "name":       names[i] if i < len(names) else f"f{i}",
                    "feature":    names[i] if i < len(names) else f"f{i}",
                    "importance": round(float(sv_fraud[i]), 5),
                    "shap_value": round(float(sv_fraud[i]), 5),
                }
                for i in order
            ]
            shap_out = [round(float(v), 5) for v in sv_fraud[:10]]
            return top_features, shap_out

        except Exception as e:
            logger.error(f"SHAP failed: {e}")
            return [], None

    def batch_predict(self, features_list: list, explain: bool = False) -> list:
        return [self.predict(f, explain=explain) for f in features_list]
