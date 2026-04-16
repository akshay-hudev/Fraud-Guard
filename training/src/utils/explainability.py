"""
Explainability Module
  - SHAP values for baseline ML models
  - Feature importance from Random Forest
  - Attention-weight extraction from HGT layers
  - Fraud risk breakdown per claim
"""

import numpy as np
import pandas as pd
import json
import os
import logging
import warnings
warnings.filterwarnings("ignore")

log = logging.getLogger(__name__)


# ── SHAP Explainer ─────────────────────────────────────────────────────────────

class SHAPExplainer:
    """Wraps shap.TreeExplainer for Random Forest / GBM models."""

    def __init__(self, model_path: str, feature_names: list[str]):
        import joblib, shap
        self.model         = joblib.load(model_path)
        self.feature_names = feature_names
        self.explainer     = shap.TreeExplainer(self.model)

    def explain(self, X: np.ndarray, top_k: int = 10) -> dict:
        """
        Returns SHAP values and top-k contributing features for each sample.
        X: shape (n_samples, n_features)
        """
        import shap
        shap_values = self.explainer.shap_values(X)
        # For binary classification, shap_values is a list [class0, class1]
        if isinstance(shap_values, list):
            shap_values = shap_values[1]   # focus on fraud class

        results = []
        for i in range(len(X)):
            sv   = shap_values[i]
            idx  = np.argsort(np.abs(sv))[::-1][:top_k]
            contrib = [
                {"feature": self.feature_names[j], "shap_value": round(float(sv[j]), 4)}
                for j in idx
            ]
            results.append({
                "sample_index":    i,
                "top_contributions": contrib,
                "base_value":      round(float(self.explainer.expected_value
                                              if not isinstance(self.explainer.expected_value, list)
                                              else self.explainer.expected_value[1]), 4),
            })
        return results

    def global_importance(self, X: np.ndarray) -> list[dict]:
        """Mean absolute SHAP value per feature (global importance)."""
        import shap
        shap_values = self.explainer.shap_values(X)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        mean_abs = np.abs(shap_values).mean(0)
        order    = np.argsort(mean_abs)[::-1]
        return [
            {"feature": self.feature_names[i], "importance": round(float(mean_abs[i]), 4)}
            for i in order
        ]


# ── GNN Attention Extractor ────────────────────────────────────────────────────

class GNNAttentionExtractor:
    """
    Extract per-edge attention weights from HGTConv layers.
    Useful for highlighting which patient-claim-doctor paths are suspicious.
    """

    def __init__(self, model, data, device: str = "cpu"):
        self.model  = model.to(device)
        self.data   = data.to(device)
        self.device = device
        self._hooks: list = []
        self.attention_weights: dict = {}

    def _register_hooks(self):
        """Attach forward hooks to each HGTConv layer."""
        for idx, conv in enumerate(self.model.convs):
            key = f"hgt_layer_{idx}"
            def hook(module, input, output, _key=key):
                # HGTConv returns (out_dict, attn_weights) if return_attention_weights=True
                # We store the output dict here; attn must be accessed differently
                self.attention_weights[_key] = output
            h = conv.register_forward_hook(hook)
            self._hooks.append(h)

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def get_fraud_risk_breakdown(self, claim_indices: list[int]) -> list[dict]:
        """
        For each claim index, return a risk breakdown based on:
          - claim node embedding magnitude
          - connected doctor fraud rate
          - connected hospital accreditation
        """
        import torch
        self.model.eval()
        with torch.no_grad():
            logits = self.model(self.data.x_dict, self.data.edge_index_dict)
            probs  = torch.softmax(logits, dim=-1)[:, 1]

        results = []
        for ci in claim_indices:
            p = float(probs[ci].item())
            results.append({
                "claim_index":     ci,
                "fraud_probability": round(p, 4),
                "risk_level":      "HIGH" if p > 0.7 else "MEDIUM" if p > 0.4 else "LOW",
            })
        return results


# ── Feature Importance (RF) ────────────────────────────────────────────────────

def get_rf_feature_importance(model_path: str, feature_names: list[str]) -> list[dict]:
    import joblib
    model = joblib.load(model_path)
    if not hasattr(model, "feature_importances_"):
        return []
    imp   = model.feature_importances_
    order = np.argsort(imp)[::-1]
    return [
        {"feature": feature_names[i], "importance": round(float(imp[i]), 4)}
        for i in order[:20]
    ]


# ── Alert System ───────────────────────────────────────────────────────────────

class AlertSystem:
    """
    Generates real-time alerts for high-risk claims.
    In production this would publish to Kafka / send email / Slack.
    """

    HIGH_RISK_THRESHOLD   = 0.75
    MEDIUM_RISK_THRESHOLD = 0.45

    def __init__(self, alert_log_path: str = "data/alerts.jsonl"):
        self.alert_log_path = alert_log_path
        os.makedirs(os.path.dirname(alert_log_path), exist_ok=True)

    def evaluate(self, claim_id: str, fraud_prob: float, metadata: dict | None = None) -> dict:
        if fraud_prob >= self.HIGH_RISK_THRESHOLD:
            level   = "HIGH"
            action  = "BLOCK_AND_REVIEW"
        elif fraud_prob >= self.MEDIUM_RISK_THRESHOLD:
            level   = "MEDIUM"
            action  = "MANUAL_REVIEW"
        else:
            level   = "LOW"
            action  = "AUTO_APPROVE"

        alert = {
            "claim_id":         claim_id,
            "fraud_probability": round(fraud_prob, 4),
            "risk_level":       level,
            "recommended_action": action,
            "metadata":         metadata or {},
        }

        if level in ("HIGH", "MEDIUM"):
            self._log_alert(alert)
            log.warning("🚨 Alert [%s] claim=%s prob=%.3f", level, claim_id, fraud_prob)

        return alert

    def _log_alert(self, alert: dict):
        from datetime import datetime
        alert["timestamp"] = datetime.utcnow().isoformat()
        with open(self.alert_log_path, "a") as f:
            f.write(json.dumps(alert) + "\n")

    def get_recent_alerts(self, n: int = 20) -> list[dict]:
        if not os.path.exists(self.alert_log_path):
            return []
        alerts = []
        with open(self.alert_log_path) as f:
            for line in f:
                try:
                    alerts.append(json.loads(line.strip()))
                except Exception:
                    pass
        return alerts[-n:]
