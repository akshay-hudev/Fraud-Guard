"""
Master Pipeline
Runs the full end-to-end ML pipeline:
  1. Generate synthetic data
  2. Preprocess & feature engineering
  3. Build graph
  4. Train baseline models
  5. Train GNN
  6. Compare & save results
"""

import os
import sys
import logging
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import ttest_rel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("pipeline")


def step_generate():
    log.info("━━━ Step 1/5: Generating synthetic data ━━━")
    from generate_dataset import generate_and_save
    generate_and_save("data/raw")


def step_preprocess():
    log.info("━━━ Step 2/5: Preprocessing & feature engineering ━━━")
    from src.data.preprocessor import FraudDataPreprocessor
    preprocessor = FraudDataPreprocessor("data/raw", "data/processed")
    return preprocessor.run()


def step_build_graph():
    log.info("━━━ Step 3/5: Building heterogeneous graph ━━━")
    from src.data.graph_builder import GraphBuilder
    builder = GraphBuilder("data/processed")
    return builder.build()


def step_train_baselines(splits: dict):
    log.info("━━━ Step 4/5: Training baseline models ━━━")
    import numpy as np
    from src.models.baseline import BaselineTrainer

    os.makedirs("models/baseline", exist_ok=True)
    trainer = BaselineTrainer("models/baseline")
    trainer.train_all(
        splits["X_train"], splits["y_train"],
        splits["X_val"],   splits["y_val"],
        tune=True,
    )
    test_results = trainer.evaluate_test(splits["X_test"], splits["y_test"])
    seed_results = trainer.evaluate_five_seeds(
        splits["X_train"], splits["y_train"],
        splits["X_val"], splits["y_val"],
        splits["X_test"], splits["y_test"],
    )

    # Save per-model results
    for name, metrics in test_results.items():
        with open(f"models/baseline/{name}_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

    return test_results, seed_results


def step_train_gnn(data):
    log.info("━━━ Step 5/5: Training GNN ━━━")
    import torch
    from src.models.gnn import build_gnn_model
    from src.training.train_gnn import GNNTrainer

    os.makedirs("models/gnn", exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = build_gnn_model(data, hidden_dim=128, num_layers=2)
    trainer = GNNTrainer(
        model, data,
        model_dir = "models/gnn",
        lr        = 1e-3,
        epochs    = 100,
        patience  = 15,
        device    = device,
    )
    trainer.train()
    return trainer.evaluate_test()


def _load_seed_results(log_path: str = "logs/gnn_5seed_results.json") -> dict:
    if not Path(log_path).exists():
        return {}
    with open(log_path) as f:
        return json.load(f)


def _statistical_tests_for_table(seed_results: dict) -> list[dict]:
    gb_rows = seed_results.get("gradient_boosting", [])
    gb_ring_recalls = [
        row.get("ring_recall")
        for row in gb_rows
        if row.get("ring_recall") is not None
    ]
    tests = []
    for model_name, rows in seed_results.items():
        if model_name == "gradient_boosting":
            continue
        if "gnn" not in model_name.lower() and "hgt" not in model_name.lower():
            continue
        hgt_ring_recalls = [
            row.get("ring_recall")
            for row in rows
            if row.get("ring_recall") is not None
        ]
        if len(hgt_ring_recalls) >= 2 and len(gb_ring_recalls) >= 2:
            t_stat, p_value = ttest_rel(hgt_ring_recalls, gb_ring_recalls)
            mean_diff = np.mean(hgt_ring_recalls) - np.mean(gb_ring_recalls)
            pooled_std = np.sqrt(
                (np.std(hgt_ring_recalls, ddof=1) ** 2 + np.std(gb_ring_recalls, ddof=1) ** 2) / 2
            )
            cohens_d = mean_diff / pooled_std if pooled_std != 0 else 0.0
        else:
            t_stat = None
            p_value = None
            cohens_d = None
        tests.append({
            "comparison": f"t-test: {model_name} vs gradient_boosting",
            "t_statistic": float(t_stat) if t_stat is not None else None,
            "p_value": float(p_value) if p_value is not None else None,
            "cohens_d": float(cohens_d) if cohens_d is not None else None,
            "ring_recall_tstat": float(t_stat) if t_stat is not None else None,
            "ring_recall_pvalue": float(p_value) if p_value is not None else None,
            "ring_recall_cohens_d": float(cohens_d) if cohens_d is not None else None,
            "n_seeds": min(len(hgt_ring_recalls), len(gb_ring_recalls)),
        })
    return tests


def _load_test_claims(n_test: int):
    candidates = [
        "data/processed/features_raw.csv",
        "training/data/processed/features_raw.csv",
        "data/processed/claims_engineered.csv",
        "training/data/processed/claims_engineered.csv",
    ]
    path = next((p for p in candidates if Path(p).exists()), None)
    if path is None:
        return None
    import pandas as pd
    claims = pd.read_csv(path)
    if "claim_date" in claims.columns:
        claims["claim_date"] = pd.to_datetime(claims["claim_date"])
        claims = claims.sort_values("claim_date").reset_index(drop=True)
    if len(claims) < n_test:
        return None
    return claims.iloc[-n_test:].reset_index(drop=True)


def _save_per_ring_recall(y_test: np.ndarray) -> dict | None:
    from src.utils.metrics import evaluate_per_ring
    import joblib

    test_claims = _load_test_claims(len(y_test))
    if test_claims is None or "ring_id" not in test_claims.columns:
        return None
    ring_labels = {
        str(int(ring_id)): group.index.astype(int).tolist()
        for ring_id, group in test_claims[
            test_claims["ring_id"].fillna(0).astype(int) > 0
        ].groupby("ring_id")
    }
    if not ring_labels:
        return None
    model_path = Path("models/baseline/gradient_boosting.pkl")
    if not model_path.exists():
        return None
    model = joblib.load(model_path)
    test_claims = test_claims.copy()
    test_claims["y_true"] = y_test
    x_test_path = Path("data/processed/X_test.npy")
    if x_test_path.exists():
        X_test = np.load(x_test_path)
        if len(X_test) == len(test_claims):
            test_claims["y_pred"] = model.predict(X_test)
    analysis = evaluate_per_ring(model, test_claims, ring_labels, out_path="logs/per_ring_recall.json")
    return analysis


def compare_models(baseline_results: dict, gnn_results: dict, seed_results: dict | None = None):
    from src.utils.metrics import compare_models, save_comparison, save_json, save_results_table

    all_results = {**baseline_results, "gnn_hgt": gnn_results}
    all_seed_results = dict(seed_results or {})
    all_seed_results.update(_load_seed_results())
    statistical_tests = _statistical_tests_for_table(all_seed_results)
    if gnn_results and not statistical_tests:
        statistical_tests.append({
            "comparison": "t-test: gnn_hgt vs gradient_boosting",
            "t_statistic": None,
            "p_value": None,
            "cohens_d": None,
            "n_seeds": 0,
            "note": "Run or provide logs/gnn_5seed_results.json to populate paired HGT-vs-GB ring recall statistics.",
        })

    save_comparison(all_results, "models/comparison.json")
    save_json(statistical_tests, "logs/statistical_tests.json")
    table_df = save_results_table(
        all_results,
        "logs/model_results_table.json",
        seed_results=all_seed_results,
        statistical_tests=statistical_tests,
    )
    table_df.to_csv("logs/model_results_table.csv", index=False)
    df = compare_models(all_results)
    print("\n" + "="*65)
    print("MODEL COMPARISON")
    print("="*65)
    print(table_df.to_string(index=False))
    print("="*65)
    return all_results


def run_pipeline(skip_gnn: bool = False, skip_generate: bool = False):
    """Full pipeline runner."""
    if not skip_generate:
        step_generate()

    splits = step_preprocess()

    baseline_results, seed_results = step_train_baselines(splits)

    all_seed_results = dict(seed_results or {})
    all_seed_results.update(_load_seed_results())

    gnn_results = {}
    if not skip_gnn:
        try:
            data = step_build_graph()
            gnn_results = step_train_gnn(data)
        except ImportError as e:
            log.warning("PyTorch Geometric not installed (%s). Skipping GNN step.", e)
            log.warning("Install with: pip install torch-geometric")
            GNN_STUB = {
                "f1_mean": None, "f1_std": None,
                "f1_ci_lower": None, "f1_ci_upper": None,
                "recall_mean": None, "auc_roc_mean": None,
                "auc_pr_mean": None, "auc_pr_ci_lower": None,
                "auc_pr_ci_upper": None, "ring_recall": None,
                "note": "Skipped: torch-sparse not installed",
            }
            for gnn_name in ["HGT", "RGCN", "HAN", "SimpleHGN"]:
                all_seed_results[gnn_name] = [GNN_STUB]

    per_ring_analysis = _save_per_ring_recall(splits["y_test"])

    compare_models(baseline_results, gnn_results, seed_results)

    import json
    import os
    from scipy import stats

    os.makedirs("logs", exist_ok=True)

    results_table = {}
    def _ci(vals: list[float]) -> float:
        if len(vals) < 2:
            return 0.0
        return 1.96 * np.std(vals, ddof=1) / np.sqrt(len(vals))

    for model_name, seed_rows in all_seed_results.items():
        if not seed_rows:
            continue
        f1s = [row["f1"] for row in seed_rows if "f1" in row]
        recs = [row["recall"] for row in seed_rows if "recall" in row]
        aucs = [row["roc_auc"] for row in seed_rows if "roc_auc" in row]
        auc_prs = [row["auc_pr"] for row in seed_rows if "auc_pr" in row]
        if not f1s:
            if seed_rows and isinstance(seed_rows[0], dict) and "f1_mean" in seed_rows[0]:
                stub = seed_rows[0]
                results_table[model_name] = {
                    "f1_mean": stub.get("f1_mean"),
                    "f1_std": stub.get("f1_std"),
                    "f1_ci_lower": stub.get("f1_ci_lower"),
                    "f1_ci_upper": stub.get("f1_ci_upper"),
                    "recall_mean": stub.get("recall_mean"),
                    "auc_roc_mean": stub.get("auc_roc_mean"),
                    "auc_pr_mean": stub.get("auc_pr_mean"),
                    "auc_pr_ci_lower": stub.get("auc_pr_ci_lower"),
                    "auc_pr_ci_upper": stub.get("auc_pr_ci_upper"),
                    "ring_recall": stub.get("ring_recall"),
                    "note": stub.get("note"),
                }
            continue
        results_table[model_name] = {
            "f1_mean": round(np.mean(f1s), 4),
            "f1_std": round(np.std(f1s, ddof=1), 4) if len(f1s) > 1 else 0.0,
            "f1_ci_lower": round(np.mean(f1s) - _ci(f1s), 4),
            "f1_ci_upper": round(np.mean(f1s) + _ci(f1s), 4),
            "recall_mean": round(np.mean(recs), 4) if recs else None,
            "auc_roc_mean": round(np.mean(aucs), 4) if aucs else None,
            "auc_pr_mean": round(np.mean(auc_prs), 4) if auc_prs else None,
            "auc_pr_ci_lower": round(np.mean(auc_prs) - _ci(auc_prs), 4) if auc_prs else None,
            "auc_pr_ci_upper": round(np.mean(auc_prs) + _ci(auc_prs), 4) if auc_prs else None,
        }

    with open("logs/full_results_table.json", "w") as f:
        json.dump(results_table, f, indent=2)
    print("✓ Saved logs/full_results_table.json")

    hgt_key = "gnn_hgt" if "gnn_hgt" in all_seed_results else "HGT"
    gb_key = "gradient_boosting" if "gradient_boosting" in all_seed_results else "GB"
    hgt_ring_recalls = [
        row["ring_recall"]
        for row in all_seed_results.get(hgt_key, [])
        if row.get("ring_recall") is not None
    ]
    gb_ring_recalls = [
        row["ring_recall"]
        for row in all_seed_results.get(gb_key, [])
        if row.get("ring_recall") is not None
    ]

    if len(hgt_ring_recalls) < 2:
        stat_results = {
            "test": "paired_ttest_hgt_vs_gb_ring_recall",
            "hgt_ring_recall_per_seed": hgt_ring_recalls,
            "gb_ring_recall_per_seed": gb_ring_recalls,
            "t_statistic": "N/A",
            "p_value": "N/A",
            "cohens_d": "N/A",
            "mean_difference": "N/A",
            "significant_at_0.05": False,
            "note": "HGT not run - install torch-sparse and re-run",
        }
    else:
        t_stat, p_value = stats.ttest_rel(hgt_ring_recalls, gb_ring_recalls)
        mean_diff = np.mean(hgt_ring_recalls) - np.mean(gb_ring_recalls)
        pooled_std = np.sqrt(
            (np.std(hgt_ring_recalls, ddof=1) ** 2 +
             np.std(gb_ring_recalls, ddof=1) ** 2) / 2
        )
        cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0.0
        stat_results = {
            "test": "paired_ttest_hgt_vs_gb_ring_recall",
            "hgt_ring_recall_per_seed": [round(x, 4) for x in hgt_ring_recalls],
            "gb_ring_recall_per_seed": [round(x, 4) for x in gb_ring_recalls],
            "mean_difference": round(float(mean_diff), 4),
            "t_statistic": round(float(t_stat), 4),
            "p_value": round(float(p_value), 6),
            "cohens_d": round(float(cohens_d), 4),
            "significant_at_0.05": bool(p_value < 0.05),
        }

    with open("logs/statistical_tests.json", "w") as f:
        json.dump(stat_results, f, indent=2)
    print("✓ Saved logs/statistical_tests.json")

    per_ring_recall_by_model = {}
    if isinstance(per_ring_analysis, dict):
        per_ring = per_ring_analysis.get("per_ring_recall", {})
        if isinstance(per_ring, dict):
            per_ring_recall_by_model["GB"] = {
                f"ring_{ring_id}": (row.get("recall") if isinstance(row, dict) else row)
                for ring_id, row in per_ring.items()
            }

    if not per_ring_recall_by_model and Path("logs/per_ring_recall.json").exists():
        with open("logs/per_ring_recall.json") as f:
            payload = json.load(f)
        if isinstance(payload, dict) and "per_ring_recall" in payload:
            per_ring_recall_by_model = {"GB": payload["per_ring_recall"]}

    with open("logs/per_ring_recall.json", "w") as f:
        json.dump({
            "per_ring_recall": per_ring_recall_by_model,
            "note": "ring_i recall = fraction of ring i claims correctly flagged",
        }, f, indent=2)
    print("✓ Saved logs/per_ring_recall.json")

    ring_claim_recall = {}
    ring_doctor_recall = {}
    if Path("logs/ring_fraud_evaluation.json").exists():
        with open("logs/ring_fraud_evaluation.json") as f:
            payload = json.load(f)
        ring_claim_recall = payload.get("ring_claim_recall", {})
        ring_doctor_recall = payload.get("ring_doctor_recall", {})

    with open("logs/ring_fraud_evaluation.json", "w") as f:
        json.dump({
            "ring_claim_recall": ring_claim_recall,
            "ring_doctor_recall": ring_doctor_recall,
            "definition": {
                "claim": "fraction of ring-member claims correctly flagged",
                "doctor": "fraction of ring doctors where >=50% of claims flagged",
            },
        }, f, indent=2)
    print("✓ Saved logs/ring_fraud_evaluation.json")

    log.info("✅ Pipeline complete! All models saved in 'models/'")


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fraud Detection ML Pipeline")
    parser.add_argument("--skip-gnn",      action="store_true",
                        help="Skip GNN training (faster, for CPU-only machines)")
    parser.add_argument("--skip-generate", action="store_true",
                        help="Skip data generation (use existing raw data)")
    args = parser.parse_args()

    run_pipeline(skip_gnn=args.skip_gnn, skip_generate=args.skip_generate)
