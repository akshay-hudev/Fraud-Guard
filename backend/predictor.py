"""
Fraud Predictor
Unified inference interface used by the FastAPI backend.
Loads models once and caches them in memory.
"""

import os
import json
import logging
import numpy as np
import joblib
import torch

log = logging.getLogger(__name__)


class FraudPredictor:
    """
    Singleton-style predictor that:
    1. Loads scaler + label encoders
    2. Loads Random Forest (fast, always available)
    3. Optionally loads GNN for richer predictions
    4. Provides a unified .predict(claim_dict) interface
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(
        self,
        processed_dir: str = "data/processed",
        model_dir:     str = "models",
        use_gnn:       bool = False,
        device:        str = "cpu",
    ):
        if self._initialized:
            return
        self.processed_dir = processed_dir
        self.model_dir     = model_dir
        self.device        = device
        self.use_gnn       = use_gnn

        self._load_preprocessors()
        self._load_rf()
        if use_gnn:
            self._load_gnn()

        self._initialized = True
        log.info("FraudPredictor initialised (GNN=%s)", use_gnn)

    def _load_preprocessors(self):
        scaler_path = f"{self.processed_dir}/scaler.pkl"
        le_path     = f"{self.processed_dir}/label_encoders.pkl"

        if os.path.exists(scaler_path):
            self.scaler         = joblib.load(scaler_path)
            self.label_encoders = joblib.load(le_path)
        else:
            log.warning("Scaler not found. Using passthrough.")
            self.scaler         = None
            self.label_encoders = {}

        # Load feature names from processed features
        feat_path = f"{self.processed_dir}/features_raw.csv"
        if os.path.exists(feat_path):
            import pandas as pd
            df = pd.read_csv(feat_path, nrows=1)
            drop = ["claim_id", "patient_id", "doctor_id", "hospital_id",
                    "claim_date", "approved", "fraud_label"]
            self.feature_names = [c for c in df.columns if c not in drop]
        else:
            self.feature_names = []

    def _load_rf(self):
        path = f"{self.model_dir}/baseline/random_forest.pkl"
        if os.path.exists(path):
            self.rf_model = joblib.load(path)
            log.info("RF model loaded from %s", path)
        else:
            self.rf_model = None
            log.warning("RF model not found at %s", path)

    def _load_gnn(self):
        graph_path = f"{self.processed_dir}/hetero_graph.pt"
        ckpt_path  = f"{self.model_dir}/gnn/best_model.pt"
        if not os.path.exists(graph_path) or not os.path.exists(ckpt_path):
            log.warning("GNN artefacts not found. Skipping GNN.")
            self.gnn_model = None
            self.graph_data = None
            return

        from src.models.gnn import build_gnn_model
        self.graph_data = torch.load(graph_path, map_location=self.device)
        self.gnn_model  = build_gnn_model(self.graph_data)
        self.gnn_model.load_state_dict(
            torch.load(ckpt_path, map_location=self.device)
        )
        self.gnn_model.eval()
        log.info("GNN model loaded.")

    # ── Feature preparation ────────────────────────────────────────────────────

    def _prepare_features(self, claim: dict) -> np.ndarray:
        """Convert a raw claim dict → scaled feature vector."""
        import pandas as pd

        # Fill defaults for any missing fields
        defaults = {
            "claim_amount": 1000, "num_procedures": 1, "days_in_hospital": 0,
            "age": 40, "member_since_years": 5, "years_experience": 10,
            "avg_claims_per_month": 20, "num_beds": 200, "is_accredited": 1,
            "license_valid": 1, "patient_total_claims": 1, "patient_total_amount": 1000,
            "patient_avg_amount": 1000, "patient_max_amount": 1000,
            "patient_unique_doctors": 1, "patient_unique_hospitals": 1,
            "doctor_total_claims": 10, "doctor_total_amount": 10000,
            "doctor_avg_amount": 1000, "doctor_unique_patients": 10,
            "doctor_fraud_rate": 0.05, "hosp_total_claims": 100,
            "hosp_avg_amount": 1500, "hosp_fraud_rate": 0.05,
            "amount_vs_patient_avg": 1.0, "amount_vs_doctor_avg": 1.0,
            "amount_vs_hosp_avg": 1.0, "claim_month": 6, "claim_dayofweek": 2,
            "gender": "M", "chronic_condition": "None", "insurance_type": "Private",
            "specialty": "General Practitioner", "state": "CA",
            "procedure_code": "PC001", "diagnosis_code": "DX001",
        }
        row = {**defaults, **claim}
        df  = pd.DataFrame([row])

        # Encode categoricals
        cat_cols = ["gender", "chronic_condition", "insurance_type",
                    "specialty", "state", "procedure_code", "diagnosis_code"]
        for col in cat_cols:
            if col in df.columns and col in self.label_encoders:
                le = self.label_encoders[col]
                df[col] = df[col].map(
                    lambda x, le=le: le.transform([str(x)])[0]
                    if str(x) in le.classes_ else 0
                )

        # Keep only feature columns in the right order
        if self.feature_names:
            for col in self.feature_names:
                if col not in df.columns:
                    df[col] = 0
            df = df[self.feature_names]

        X = df.values.astype(np.float32)

        if self.scaler is not None:
            X = self.scaler.transform(X)

        return X

    # ── Public API ─────────────────────────────────────────────────────────────

    def predict(self, claim: dict, explain: bool = False) -> dict:
        """
        Parameters
        ----------
        claim   : raw claim dict (same schema as the claims CSV)
        explain : whether to include SHAP feature contributions

        Returns
        -------
        {
            "fraud_probability": float,
            "prediction":        0 | 1,
            "risk_level":        "LOW" | "MEDIUM" | "HIGH",
            "model_used":        str,
            "contributions":     [...] (if explain=True)
        }
        """
        X = self._prepare_features(claim)

        if self.rf_model is not None:
            proba = float(self.rf_model.predict_proba(X)[0, 1])
            pred  = int(proba >= 0.5)
            model_used = "random_forest"
        else:
            # Fallback: simple rule-based heuristic
            amount = float(claim.get("claim_amount", 1000))
            proba  = min(amount / 50_000, 1.0)
            pred   = int(proba >= 0.5)
            model_used = "heuristic"

        risk = "HIGH" if proba > 0.7 else "MEDIUM" if proba > 0.4 else "LOW"

        result: dict = {
            "fraud_probability": round(proba, 4),
            "prediction":        pred,
            "risk_level":        risk,
            "model_used":        model_used,
        }

        if explain and self.rf_model is not None:
            try:
                import shap
                explainer = shap.TreeExplainer(self.rf_model)
                sv = explainer.shap_values(X)
                if isinstance(sv, list):
                    sv = sv[1]
                sv = sv[0]
                names = self.feature_names or [f"f{i}" for i in range(len(sv))]
                order = np.argsort(np.abs(sv))[::-1][:10]
                result["contributions"] = [
                    {"feature": names[j], "shap_value": round(float(sv[j]), 4)}
                    for j in order
                ]
            except Exception as e:
                log.warning("SHAP failed: %s", e)

        return result

    def batch_predict(self, claims: list[dict]) -> list[dict]:
        return [self.predict(c) for c in claims]
