"""
Baseline ML Models
Logistic Regression and Random Forest for fraud detection.
Used as performance baselines against the GNN.
"""

import numpy as np
import joblib
import os
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    confusion_matrix,
)

log = logging.getLogger(__name__)


# ── Metrics helper ─────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    y_proba: np.ndarray | None = None) -> dict:
    metrics = {
        "accuracy":  round(accuracy_score (y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":    round(recall_score   (y_true, y_pred, zero_division=0), 4),
        "f1":        round(f1_score       (y_true, y_pred, zero_division=0), 4),
    }
    if y_proba is not None:
        metrics["roc_auc"] = round(roc_auc_score(y_true, y_proba), 4)
    return metrics


# ── Model definitions ──────────────────────────────────────────────────────────

BASELINE_MODELS = {
    "logistic_regression": LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=42
    ),
    "random_forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),
    "gradient_boosting": GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
    ),
}


class BaselineTrainer:
    """Train, evaluate, and persist all baseline models."""

    def __init__(self, model_dir: str = "models"):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        self.results: dict = {}

    def train_all(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val:   np.ndarray,
        y_val:   np.ndarray,
    ) -> dict:
        for name, model in BASELINE_MODELS.items():
            log.info("Training %s ...", name)
            model.fit(X_train, y_train)

            # Evaluate
            val_pred  = model.predict(X_val)
            val_proba = (
                model.predict_proba(X_val)[:, 1]
                if hasattr(model, "predict_proba") else None
            )
            metrics = compute_metrics(y_val, val_pred, val_proba)
            self.results[name] = metrics
            log.info("  %s — val metrics: %s", name, metrics)

            # Persist
            path = f"{self.model_dir}/{name}.pkl"
            joblib.dump(model, path)
            log.info("  Saved to %s", path)

        return self.results

    def evaluate_test(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> dict:
        test_results = {}
        for name in BASELINE_MODELS:
            path = f"{self.model_dir}/{name}.pkl"
            if not os.path.exists(path):
                log.warning("Model not found: %s", path)
                continue
            model = joblib.load(path)
            pred  = model.predict(X_test)
            proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
            metrics = compute_metrics(y_test, pred, proba)
            test_results[name] = metrics
            log.info("TEST  %s — %s", name, metrics)
            print(f"\n{'='*50}")
            print(f"Model: {name.upper()}")
            print(classification_report(y_test, pred, target_names=["Legit", "Fraud"]))
            print("Confusion Matrix:\n", confusion_matrix(y_test, pred))
        return test_results

    def predict_single(self, features: np.ndarray, model_name: str = "random_forest") -> dict:
        """Predict a single claim. features shape: (1, n_features)"""
        path = f"{self.model_dir}/{model_name}.pkl"
        model = joblib.load(path)
        pred  = model.predict(features)[0]
        proba = model.predict_proba(features)[0, 1] if hasattr(model, "predict_proba") else None
        return {"prediction": int(pred), "fraud_probability": float(proba or pred)}


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    processed_dir = "data/processed"
    X_train = np.load(f"{processed_dir}/X_train.npy")
    X_val   = np.load(f"{processed_dir}/X_val.npy")
    X_test  = np.load(f"{processed_dir}/X_test.npy")
    y_train = np.load(f"{processed_dir}/y_train.npy")
    y_val   = np.load(f"{processed_dir}/y_val.npy")
    y_test  = np.load(f"{processed_dir}/y_test.npy")

    trainer = BaselineTrainer(model_dir="models/baseline")
    trainer.train_all(X_train, y_train, X_val, y_val)
    trainer.evaluate_test(X_test, y_test)
