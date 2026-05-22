"""
Baseline ML Models
Logistic Regression and Random Forest for fraud detection.
Used as performance baselines against the GNN.
"""

import json
import numpy as np
import joblib
import os
import logging
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, PredefinedSplit
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)
from src.utils.metrics import (
    compute_binary_metrics,
    evaluate_per_ring,
    evaluate_ring_fraud_units,
    save_json,
)

log = logging.getLogger(__name__)


# ── Metrics helper ─────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    y_proba: np.ndarray | None = None) -> dict:
    return compute_binary_metrics(y_true, y_pred, y_proba)


# ── Model definitions ──────────────────────────────────────────────────────────

BASELINE_MODELS = {
    "logistic_regression": LogisticRegression(
        max_iter=3000, class_weight="balanced", random_state=42, solver="saga"
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

BASELINE_PARAM_GRIDS = {
    "gradient_boosting": {
        "n_estimators": [100, 150, 200],
        "learning_rate": [0.05, 0.1, 0.15],
        "max_depth": [3, 5, 7],
        "subsample": [0.7, 0.8, 0.9],
    },
    "random_forest": {
        "n_estimators": [100, 200, 300],
        "max_depth": [8, 12, 16],
    },
    "logistic_regression": {
        "C": [0.1, 1.0, 10.0],
        "penalty": ["l1", "l2"],
    },
}


class BaselineTrainer:
    """Train, evaluate, and persist all baseline models."""

    def __init__(self, model_dir: str = "models", log_dir: str = "logs"):
        self.model_dir = model_dir
        self.log_dir = log_dir
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        self.results: dict = {}
        self.best_params: dict = {}

    @staticmethod
    def _with_random_state(model, seed: int):
        params = model.get_params()
        if "random_state" in params:
            model.set_params(random_state=seed)
        return model

    def _build_model(self, name: str, seed: int = 42, params: dict | None = None):
        model = clone(BASELINE_MODELS[name])
        if params:
            model.set_params(**params)
        return self._with_random_state(model, seed)

    def tune_all(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> dict:
        """
        Equalize LR/RF/GB hyperparameter tuning.

        The held-out test set is never touched. The search pool is the
        train+validation portion, with 5-fold CV and refit=True.
        """
        split_index = [-1] * len(X_train) + [0] * len(X_val)
        X_search = np.vstack([X_train, X_val])
        y_search = np.concatenate([y_train, y_val])
        cv = PredefinedSplit(test_fold=split_index)

        best_params = {}
        searches = {}
        for name in ["logistic_regression", "random_forest", "gradient_boosting"]:
            log.info("Tuning %s with GridSearchCV(cv=5, scoring='f1') ...", name)
            search = GridSearchCV(
                estimator=self._build_model(name, seed=42),
                param_grid=BASELINE_PARAM_GRIDS[name],
                cv=cv,
                scoring="f1",
                refit=True,
                n_jobs=-1,
            )
            # Tuning uses predefined validation split — test set never touched
            search.fit(X_search, y_search)
            best_params[name] = {
                "best_params": search.best_params_,
                "best_score": round(float(search.best_score_), 4),
            }
            searches[name] = search
            path = f"{self.model_dir}/{name}.pkl"
            joblib.dump(search.best_estimator_, path)
            log.info("  Best %s params: %s", name, search.best_params_)
            log.info("  Saved tuned estimator to %s", path)

        self.best_params = {name: row["best_params"] for name, row in best_params.items()}
        save_json(best_params, f"{self.log_dir}/tabular_best_params.json")
        os.makedirs("logs", exist_ok=True)
        best_params_summary = {
            "GB": {
                "best_params": searches["gradient_boosting"].best_params_,
                "best_val_f1": round(float(searches["gradient_boosting"].best_score_), 4),
            },
            "RF": {
                "best_params": searches["random_forest"].best_params_,
                "best_val_f1": round(float(searches["random_forest"].best_score_), 4),
            },
            "LR": {
                "best_params": searches["logistic_regression"].best_params_,
                "best_val_f1": round(float(searches["logistic_regression"].best_score_), 4),
            },
        }
        with open("logs/tabular_best_params.json", "w") as f:
            json.dump(best_params_summary, f, indent=2)
        print("✓ Saved logs/tabular_best_params.json")
        return best_params

    def train_all(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val:   np.ndarray,
        y_val:   np.ndarray,
        tune: bool = True,
    ) -> dict:
        if tune:
            self.tune_all(X_train, y_train, X_val, y_val)
            train_pool_X = np.vstack([X_train, X_val])
            train_pool_y = np.concatenate([y_train, y_val])
        else:
            train_pool_X, train_pool_y = X_train, y_train

        for name in BASELINE_MODELS:
            if tune and os.path.exists(f"{self.model_dir}/{name}.pkl"):
                model = joblib.load(f"{self.model_dir}/{name}.pkl")
            else:
                model = self._build_model(name, params=self.best_params.get(name))
                log.info("Training %s ...", name)
                model.fit(train_pool_X, train_pool_y)

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
        preds_by_model = {}
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
            preds_by_model[name] = pred
            if name == "gradient_boosting":
                self._maybe_save_ring_evaluation(y_test, pred)
            log.info("TEST  %s — %s", name, metrics)
            print(f"\n{'='*50}")
            print(f"Model: {name.upper()}")
            print(classification_report(y_test, pred, target_names=["Legit", "Fraud"]))
            print("Confusion Matrix:\n", confusion_matrix(y_test, pred))

        test_claims = self._load_test_claims(len(y_test))
        if test_claims is not None:
            ring_claim_recall = {}
            ring_doctor_recall = {}
            for name, label in [("gradient_boosting", "GB"), ("random_forest", "RF")]:
                if name in preds_by_model:
                    report = evaluate_ring_fraud_units(
                        test_claims,
                        preds_by_model[name],
                        out_path=None,
                    )
                    ring_claim_recall[label] = float(report.get("ring_claim_recall", 0.0))
                    ring_doctor_recall[label] = float(report.get("ring_doctor_recall", 0.0))

            hgt_preds = self._load_gnn_predictions(len(y_test))
            if hgt_preds is not None:
                report = evaluate_ring_fraud_units(
                    test_claims,
                    hgt_preds,
                    out_path=None,
                )
                ring_claim_recall["HGT"] = float(report.get("ring_claim_recall", 0.0))
                ring_doctor_recall["HGT"] = float(report.get("ring_doctor_recall", 0.0))
            else:
                ring_claim_recall["HGT"] = 0.0
                ring_doctor_recall["HGT"] = 0.0

            os.makedirs("logs", exist_ok=True)
            with open("logs/ring_fraud_evaluation.json", "w") as f:
                json.dump({
                    "ring_claim_recall": {
                        "HGT": float(ring_claim_recall.get("HGT", 0.0)),
                        "GB": float(ring_claim_recall.get("GB", 0.0)),
                        "RF": float(ring_claim_recall.get("RF", 0.0)),
                    },
                    "ring_doctor_recall": {
                        "HGT": float(ring_doctor_recall.get("HGT", 0.0)),
                        "GB": float(ring_doctor_recall.get("GB", 0.0)),
                        "RF": float(ring_doctor_recall.get("RF", 0.0)),
                    },
                    "definition": {
                        "claim_recall": "fraction of ring-member claims correctly flagged",
                        "doctor_recall": "fraction of ring doctors where >=50% claims flagged",
                    },
                }, f, indent=2)
        return test_results

    def _maybe_save_ring_evaluation(self, y_test: np.ndarray, pred: np.ndarray) -> None:
        candidates = [
            "data/processed/features_raw.csv",
            "training/data/processed/features_raw.csv",
            "data/processed/claims_engineered.csv",
            "training/data/processed/claims_engineered.csv",
        ]
        path = next((p for p in candidates if os.path.exists(p)), None)
        if path is None:
            return
        try:
            import pandas as pd
            claims = pd.read_csv(path)
            if "claim_date" in claims.columns:
                claims["claim_date"] = pd.to_datetime(claims["claim_date"])
                claims = claims.sort_values("claim_date").reset_index(drop=True)
            test_claims = claims.iloc[-len(y_test):].reset_index(drop=True).copy()
            test_claims["y_true"] = y_test
            evaluate_ring_fraud_units(
                test_claims,
                pred,
                out_path=f"{self.log_dir}/ring_fraud_evaluation.json",
            )
            if "ring_id" in test_claims.columns:
                ring_labels = {
                    str(int(ring_id)): group.index.astype(int).tolist()
                    for ring_id, group in test_claims[
                        test_claims["ring_id"].fillna(0).astype(int) > 0
                    ].groupby("ring_id")
                }
                ring_claims = test_claims.copy()
                ring_claims["y_pred"] = pred
                evaluate_per_ring(
                    None,
                    ring_claims,
                    ring_labels,
                    out_path=f"{self.log_dir}/per_ring_recall.json",
                )
        except Exception as exc:
            log.warning("Ring fraud unit evaluation skipped: %s", exc)

    def evaluate_five_seeds(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        seeds: list[int] | None = None,
    ) -> dict:
        """Rerun post-tuning tabular evaluations with fixed best params."""
        seeds = seeds or [0, 1, 2, 3, 4]
        train_pool_X = np.vstack([X_train, X_val])
        train_pool_y = np.concatenate([y_train, y_val])
        seed_results: dict[str, list[dict]] = {name: [] for name in BASELINE_MODELS}
        test_claims = self._load_test_claims(len(y_test))
        ring_mask = None
        if test_claims is not None:
            if "ring_id" in test_claims.columns:
                ring_mask = test_claims["ring_id"].fillna(0).astype(int).to_numpy() > 0
            else:
                ring_mask = y_test.astype(int) == 1

        for seed in seeds:
            for name in BASELINE_MODELS:
                params = self.best_params.get(name)
                model = self._build_model(name, seed=seed, params=params)
                model.fit(train_pool_X, train_pool_y)
                pred = model.predict(X_test)
                proba = (
                    model.predict_proba(X_test)[:, 1]
                    if hasattr(model, "predict_proba") else None
                )
                metrics = compute_metrics(y_test, pred, proba)
                if ring_mask is not None:
                    ring_positions = ring_mask & (y_test.astype(int) == 1)
                    if ring_positions.sum() > 0:
                        metrics["ring_recall"] = float((pred[ring_positions] == 1).mean())
                    else:
                        metrics["ring_recall"] = 0.0
                metrics["seed"] = seed
                seed_results[name].append(metrics)

        save_json(seed_results, f"{self.log_dir}/tabular_5seed_results.json")
        return seed_results

    def predict_single(self, features: np.ndarray, model_name: str = "random_forest") -> dict:
        """Predict a single claim. features shape: (1, n_features)"""
        path = f"{self.model_dir}/{model_name}.pkl"
        model = joblib.load(path)
        pred  = model.predict(features)[0]
        proba = model.predict_proba(features)[0, 1] if hasattr(model, "predict_proba") else None
        return {"prediction": int(pred), "fraud_probability": float(proba or pred)}

    def _load_test_claims(self, n_test: int):
        candidates = [
            "data/processed/features_raw.csv",
            "training/data/processed/features_raw.csv",
            "data/processed/claims_engineered.csv",
            "training/data/processed/claims_engineered.csv",
        ]
        path = next((p for p in candidates if os.path.exists(p)), None)
        if path is None:
            return None
        import pandas as pd
        claims = pd.read_csv(path)
        if "claim_date" in claims.columns:
            claims["claim_date"] = pd.to_datetime(claims["claim_date"])
            claims = claims.sort_values("claim_date").reset_index(drop=True)
        if len(claims) < n_test:
            return None
        test_claims = claims.iloc[-n_test:].reset_index(drop=True).copy()
        return test_claims

    def _load_gnn_predictions(self, n_test: int):
        candidates = [
            "models/gnn/test_predictions.json",
            "training/models/gnn/test_predictions.json",
        ]
        path = next((p for p in candidates if os.path.exists(p)), None)
        if path is None:
            return None
        with open(path) as f:
            payload = json.load(f)
        preds = np.asarray(payload.get("y_pred", []), dtype=int)
        if len(preds) != n_test:
            return None
        return preds


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
