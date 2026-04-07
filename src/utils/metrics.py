"""
Metrics & Comparison Utilities
"""

import json
import os
import numpy as np
import pandas as pd
import logging

log = logging.getLogger(__name__)


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
    rows = []
    for model, metrics in results.items():
        rows.append({"Model": model, **metrics})
    df = pd.DataFrame(rows).set_index("Model")
    df = df.sort_values("f1", ascending=False)
    return df


def save_comparison(results: dict, out_path: str = "models/comparison.json"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info("Comparison saved to %s", out_path)


# Mock result storage for API (when models aren't fully trained yet)
MOCK_RESULTS = {
    "logistic_regression": {
        "accuracy": 0.8812, "precision": 0.7231, "recall": 0.6940, "f1": 0.7083, "roc_auc": 0.8521
    },
    "random_forest": {
        "accuracy": 0.9341, "precision": 0.8812, "recall": 0.8101, "f1": 0.8441, "roc_auc": 0.9612
    },
    "gradient_boosting": {
        "accuracy": 0.9289, "precision": 0.8543, "recall": 0.8234, "f1": 0.8386, "roc_auc": 0.9580
    },
    "gnn_hgt": {
        "accuracy": 0.9571, "precision": 0.9102, "recall": 0.8843, "f1": 0.8971, "roc_auc": 0.9801
    },
}
