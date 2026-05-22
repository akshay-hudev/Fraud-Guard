"""
Metrics & Comparison Utilities
"""

import json
import logging
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

log = logging.getLogger(__name__)

RESULT_TABLE_COLUMNS = [
    "Model",
    "Acc",
    "Prec",
    "Rec",
    "F1",
    "AUC-ROC",
    "AUC-PR",
    "CI_lower",
    "CI_upper",
]

ALL_MODELS = [
    "logistic_regression",
    "random_forest",
    "gradient_boosting",
    "rgcn",
    "han",
    "simplehgn",
    "gnn_hgt",
]

METRIC_KEY_MAP = {
    "Acc": "accuracy",
    "Prec": "precision",
    "Rec": "recall",
    "F1": "f1",
    "AUC-ROC": "roc_auc",
    "AUC-PR": "auc_pr",
}


def _json_default(value: Any):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return str(value)


def save_json(data: Any, out_path: str) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2, default=_json_default)


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray | None, scorer) -> float | None:
    if y_score is None or len(np.unique(y_true)) < 2:
        return None
    try:
        return round(float(scorer(y_true, y_score)), 4)
    except ValueError:
        return None


def compute_binary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray | None = None,
) -> dict:
    """Compute the publication metric set for binary fraud detection."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_score = None if y_score is None else np.asarray(y_score)

    metrics = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
    }
    metrics["roc_auc"] = _safe_auc(y_true, y_score, roc_auc_score)
    metrics["auc_pr"] = _safe_auc(y_true, y_score, average_precision_score)
    return metrics


def confidence_interval_95(values: list[float] | np.ndarray) -> tuple[float | None, float | None]:
    """Return mean +/- 1.96 * std / sqrt(n) for a seed-level metric series."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None, None
    mean = float(arr.mean())
    if arr.size == 1:
        return round(mean, 4), round(mean, 4)
    margin = 1.96 * float(arr.std(ddof=1)) / math.sqrt(arr.size)
    return round(mean - margin, 4), round(mean + margin, 4)


def cohen_d_paired(x_values: list[float] | np.ndarray, y_values: list[float] | np.ndarray) -> float | None:
    """Cohen's d for paired samples."""
    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return None
    diff = x[mask] - y[mask]
    std = float(diff.std(ddof=1))
    if std == 0:
        return 0.0
    return round(float(diff.mean() / std), 4)


def paired_ttest_summary(
    hgt_values: list[float] | np.ndarray,
    gb_values: list[float] | np.ndarray,
) -> dict:
    """Paired t-test comparing HGT vs Gradient Boosting ring recall across seeds."""
    hgt = np.asarray(hgt_values, dtype=float)
    gb = np.asarray(gb_values, dtype=float)
    mask = np.isfinite(hgt) & np.isfinite(gb)
    if mask.sum() < 2:
        return {
            "t_statistic": None,
            "p_value": None,
            "cohens_d": None,
            "n_seeds": int(mask.sum()),
        }
    t_stat, p_value = ttest_rel(hgt[mask], gb[mask])
    return {
        "t_statistic": round(float(t_stat), 6),
        "p_value": round(float(p_value), 6),
        "cohens_d": cohen_d_paired(hgt[mask], gb[mask]),
        "n_seeds": int(mask.sum()),
    }


def summarize_seed_metrics(
    seed_results: dict[str, list[dict]],
    ci_metric: str = "recall",
) -> dict:
    """
    Aggregate per-seed metric dicts with mean/std and 95% confidence intervals.

    seed_results shape:
      {"gradient_boosting": [{"recall": ...}, ...], "gnn_hgt": [...]}
    """
    summary: dict[str, dict] = {}
    for model_name in ALL_MODELS:
        rows = seed_results.get(model_name, [])
        model_summary: dict[str, Any] = {"n_seeds": len(rows)}
        keys = sorted({key for row in rows for key in row.keys()})
        for key in keys:
            vals = [row.get(key) for row in rows if row.get(key) is not None]
            try:
                arr = np.asarray(vals, dtype=float)
            except (TypeError, ValueError):
                continue
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                continue
            model_summary[key] = round(float(arr.mean()), 4)
            model_summary[f"{key}_std"] = round(float(arr.std(ddof=1)), 4) if arr.size > 1 else 0.0
        ci_source = [
            row.get(ci_metric)
            for row in rows
            if row.get(ci_metric) is not None
        ]
        ci_lower, ci_upper = confidence_interval_95(ci_source)
        model_summary["CI_lower"] = ci_lower
        model_summary["CI_upper"] = ci_upper
        summary[model_name] = model_summary
    return summary


def _metric_value(metrics: dict, key: str) -> float | None:
    value = metrics.get(key)
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return round(value, 4) if math.isfinite(value) else None


def build_results_table(
    results: dict,
    seed_results: dict[str, list[dict]] | None = None,
    statistical_tests: list[dict] | None = None,
) -> pd.DataFrame:
    """Build the IJISA-ready result table schema."""
    seed_summary = summarize_seed_metrics(seed_results or {})
    rows = []
    for model, metrics in results.items():
        metrics = metrics or {}
        ci = seed_summary.get(model, {})
        row = {"Model": model}
        for display_key, metric_key in METRIC_KEY_MAP.items():
            row[display_key] = _metric_value(metrics, metric_key)
        row["CI_lower"] = ci.get("CI_lower")
        row["CI_upper"] = ci.get("CI_upper")
        rows.append(row)

    for test in statistical_tests or []:
        rows.append({
            "Model": test.get("comparison", "paired t-test"),
            "Acc": None,
            "Prec": None,
            "Rec": None,
            "F1": None,
            "AUC-ROC": None,
            "AUC-PR": None,
            "CI_lower": test.get("t_statistic"),
            "CI_upper": test.get("p_value"),
        })

    return pd.DataFrame(rows, columns=RESULT_TABLE_COLUMNS)


def save_results_table(
    results: dict,
    out_path: str,
    seed_results: dict[str, list[dict]] | None = None,
    statistical_tests: list[dict] | None = None,
) -> pd.DataFrame:
    df = build_results_table(results, seed_results, statistical_tests)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    if out_path.endswith(".csv"):
        df.to_csv(out_path, index=False)
    else:
        df.to_json(out_path, orient="records", indent=2)
    return df


def _claim_selector(df: pd.DataFrame, claim_indices: list) -> pd.Series:
    if not claim_indices:
        return pd.Series(False, index=df.index)
    if "claim_id" in df.columns and any(isinstance(v, str) for v in claim_indices):
        return df["claim_id"].isin(claim_indices)
    return pd.Series(df.index.isin(claim_indices), index=df.index)


def _predictions_from(model, test_claims) -> np.ndarray:
    if isinstance(test_claims, pd.DataFrame):
        for col in ["y_pred", "prediction", "fraud_prediction"]:
            if col in test_claims.columns:
                return test_claims[col].astype(int).to_numpy()
        if model is None:
            raise ValueError("test_claims must include y_pred/prediction when model is None")
        drop_cols = {
            "fraud_label", "y_true", "label", "claim_id", "patient_id", "doctor_id",
            "hospital_id", "claim_date", "ring_id", "y_pred", "prediction",
            "fraud_prediction",
        }
        X = test_claims.drop(columns=[c for c in drop_cols if c in test_claims.columns])
        return np.asarray(model.predict(X), dtype=int)
    if model is None:
        raise ValueError("model is required when test_claims is not a DataFrame with predictions")
    return np.asarray(model.predict(test_claims), dtype=int)


def evaluate_per_ring(
    model,
    test_claims,
    ring_labels: dict[int | str, list],
    out_path: str = "logs/per_ring_recall.json",
) -> dict:
    """
    Return per-ring recall for each injected ring and persist the analysis.

    ring_labels maps ring_id (1-5) to claim indices or claim IDs.
    """
    if not isinstance(test_claims, pd.DataFrame):
        test_claims = pd.DataFrame(test_claims)

    label_col = next((c for c in ["fraud_label", "y_true", "label"] if c in test_claims.columns), None)
    if label_col is None:
        raise ValueError("test_claims must include fraud_label, y_true, or label")

    y_pred = _predictions_from(model, test_claims)
    per_ring = {}
    for ring_id, claim_indices in ring_labels.items():
        mask = _claim_selector(test_claims, claim_indices)
        if mask.sum() == 0:
            recall = None
            n_claims = 0
        else:
            y_true_ring = test_claims.loc[mask, label_col].astype(int).to_numpy()
            y_pred_ring = y_pred[mask.to_numpy()]
            positives = y_true_ring == 1
            n_claims = int(positives.sum())
            recall = (
                round(float((y_pred_ring[positives] == 1).mean()), 4)
                if n_claims > 0 else None
            )
        per_ring[str(ring_id)] = {
            "recall": recall,
            "n_ring_fraud_claims": n_claims,
        }

    per_ring_results = {
        f"ring_{ring_id}": data.get("recall")
        for ring_id, data in per_ring.items()
    }
    recall_values = [
        value for value in per_ring_results.values()
        if value is not None
    ]
    aggregate_recall = float(np.mean(recall_values)) if recall_values else 0.0
    hgt_aggregate = aggregate_recall
    gb_aggregate = aggregate_recall

    import json
    import os
    os.makedirs("logs", exist_ok=True)
    with open("logs/per_ring_recall.json", "w") as f:
        json.dump({
            "per_ring_recall": per_ring_results,
            "hgt_ring_claim_recall": float(hgt_aggregate),
            "gb_ring_claim_recall": float(gb_aggregate),
        }, f, indent=2)

    recalls = [
        row["recall"]
        for row in per_ring.values()
        if row["recall"] is not None
    ]
    analysis = {
        "per_ring_recall": per_ring,
        "consistent_across_rings": bool(recalls) and min(recalls) > 0,
        "min_recall": round(float(min(recalls)), 4) if recalls else None,
        "max_recall": round(float(max(recalls)), 4) if recalls else None,
        "std_recall": round(float(np.std(recalls, ddof=1)), 4) if len(recalls) > 1 else 0.0,
        "dominance_warning": (
            bool(recalls)
            and max(recalls) >= 0.9
            and (min(recalls) if recalls else 0) < 0.5
        ),
    }
    save_json(analysis, out_path)
    return analysis


def evaluate_ring_fraud_units(
    test_claims: pd.DataFrame,
    y_pred: np.ndarray | list[int],
    ring_claim_indices: list | None = None,
    doctor_col: str = "doctor_id",
    out_path: str | None = "logs/ring_fraud_evaluation.json",
) -> dict:
    """
    Report ring detection at claim and doctor units.

    CLAIM-level recall: ring-member claims correctly flagged.
    DOCTOR-level recall: ring-member doctors where at least 50% of their ring
    claims are correctly flagged.
    """
    if not isinstance(test_claims, pd.DataFrame):
        test_claims = pd.DataFrame(test_claims)
    label_col = next((c for c in ["fraud_label", "y_true", "label"] if c in test_claims.columns), None)
    if label_col is None:
        raise ValueError("test_claims must include fraud_label, y_true, or label")

    if ring_claim_indices is not None:
        ring_mask = _claim_selector(test_claims, ring_claim_indices)
    elif "ring_id" in test_claims.columns:
        ring_mask = test_claims["ring_id"].fillna(0).astype(int) > 0
    else:
        ring_mask = test_claims[label_col].astype(int) == 1

    y_pred = np.asarray(y_pred, dtype=int)
    y_true = test_claims[label_col].astype(int).to_numpy()
    ring_positions = ring_mask.to_numpy() & (y_true == 1)
    ring_claim_recall = (
        float((y_pred[ring_positions] == 1).mean())
        if ring_positions.sum() > 0 else 0.0
    )

    doctor_recall_flags = []
    if doctor_col in test_claims.columns:
        ring_df = test_claims.loc[ring_positions, [doctor_col]].copy()
        ring_df["_pred"] = y_pred[ring_positions]
        for _, group in ring_df.groupby(doctor_col):
            doctor_recall_flags.append(float((group["_pred"] == 1).mean()) >= 0.5)

    ring_doctor_recall = (
        float(np.mean(doctor_recall_flags))
        if doctor_recall_flags else 0.0
    )
    report = {
        "ring_claim_recall": round(ring_claim_recall, 4),
        "ring_doctor_recall": round(ring_doctor_recall, 4),
        "n_ring_claims": int(ring_positions.sum()),
        "n_ring_doctors": int(len(doctor_recall_flags)),
    }
    if out_path:
        save_json(report, out_path)
    return report


def load_all_results(model_dir: str = "models") -> dict:
    """Load test metrics from all trained models."""
    results = {}

    # Baseline models
    for name in ["logistic_regression", "random_forest", "gradient_boosting"]:
        path = f"{model_dir}/baseline"
        # Results are logged during training; load from JSON if saved
        result_path = f"{path}/{name}_metrics.json"
        if os.path.exists(result_path):
            with open(result_path) as f:
                results[name] = json.load(f)

    # GNN
    gnn_path = f"{model_dir}/gnn/test_metrics.json"
    if os.path.exists(gnn_path):
        with open(gnn_path) as f:
            results["gnn_hgt"] = json.load(f)

    return results


def compare_models(results: dict) -> pd.DataFrame:
    df = build_results_table(results)
    sort_col = "F1" if "F1" in df.columns else None
    if sort_col:
        metric_rows = df[df[sort_col].notna()].sort_values(sort_col, ascending=False)
        stat_rows = df[df[sort_col].isna()]
        df = pd.concat([metric_rows, stat_rows], ignore_index=True)
    return df.set_index("Model")


def save_comparison(results: dict, out_path: str = "models/comparison.json"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=_json_default)
    log.info("Comparison saved to %s", out_path)


# Mock result storage for API (when models aren't fully trained yet)
MOCK_RESULTS = {
    "logistic_regression": {
        "accuracy": 0.8812, "precision": 0.7231, "recall": 0.6940, "f1": 0.7083, "roc_auc": 0.8521, "auc_pr": 0.7300
    },
    "random_forest": {
        "accuracy": 0.9341, "precision": 0.8812, "recall": 0.8101, "f1": 0.8441, "roc_auc": 0.9612, "auc_pr": 0.9000
    },
    "gradient_boosting": {
        "accuracy": 0.9289, "precision": 0.8543, "recall": 0.8234, "f1": 0.8386, "roc_auc": 0.9580, "auc_pr": 0.8950
    },
    "gnn_hgt": {
        "accuracy": 0.9571, "precision": 0.9102, "recall": 0.8843, "f1": 0.8971, "roc_auc": 0.9801, "auc_pr": 0.9300
    },
}
