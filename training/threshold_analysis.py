"""
Threshold sensitivity analysis for hybrid GB -> HGT routing.

Sweeps the Gradient Boosting escalation threshold and reports how much traffic
is escalated to HGT and the ring-fraud recall of the cascade (a ring claim is
caught only if GB escalates it AND HGT flags it).

Requirements (the script refuses to run without them, rather than substituting
other scores or assumed numbers):
  * models/gnn/test_predictions.json  — HGT test-set scores from a real run
  * data/processed/split_claim_ids.json — claim IDs aligning X_test rows
Mean latency is reported only if logs/latency_benchmark.json exists with
measured per-model P50 latencies ({"gradient_boosting": {"p50_ms": ...}, "gnn_hgt": {...}}).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TRAINING_DIR = Path(__file__).resolve().parent
REPO_ROOT = TRAINING_DIR.parent
sys.path.insert(0, str(TRAINING_DIR))

from src.utils.metrics import load_split_claims

DATA_DIR = REPO_ROOT / "data" / "processed"
MODEL_DIR = REPO_ROOT / "models"
LOG_DIR = REPO_ROOT / "logs"
FIGURE_DIR = REPO_ROOT / "figures"

THRESHOLDS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]


def _load_hgt_scores(n_test: int) -> np.ndarray:
    path = MODEL_DIR / "gnn" / "test_predictions.json"
    if not path.exists():
        raise SystemExit(f"{path} not found: train HGT first; GB scores are never substituted.")
    with open(path) as f:
        scores = np.asarray(json.load(f).get("y_prob", []), dtype=float)
    if len(scores) != n_test:
        raise SystemExit(f"HGT predictions have {len(scores)} rows, expected {n_test}.")
    return scores


def _load_measured_latency() -> dict | None:
    path = LOG_DIR / "latency_benchmark.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def run_threshold_sweep() -> list[dict]:
    X_test = np.load(DATA_DIR / "X_test.npy")
    y_test = np.load(DATA_DIR / "y_test.npy")
    gb_model = joblib.load(MODEL_DIR / "baseline" / "gradient_boosting.pkl")
    gb_scores = gb_model.predict_proba(X_test)[:, 1]
    hgt_pred = _load_hgt_scores(len(y_test)) >= 0.5

    test_claims = load_split_claims("test", str(DATA_DIR))
    if test_claims is None or len(test_claims) != len(y_test):
        raise SystemExit("split_claim_ids.json missing or inconsistent with X_test.")
    ring_mask = (test_claims["ring_id"].fillna(0).astype(int).to_numpy() > 0) & (y_test == 1)
    latency = _load_measured_latency()

    rows = []
    for threshold in THRESHOLDS:
        escalated = gb_scores > threshold
        caught = escalated & hgt_pred
        row = {
            "threshold": threshold,
            "escalation_rate": round(float(escalated.mean()), 4),
            "cascade_ring_recall": round(float(caught[ring_mask].mean()), 4) if ring_mask.any() else None,
        }
        if latency is not None:
            gb_ms = latency["gradient_boosting"]["p50_ms"]
            hgt_ms = latency["gnn_hgt"]["p50_ms"]
            row["mean_latency_ms"] = round(gb_ms + row["escalation_rate"] * hgt_ms, 2)
        rows.append(row)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_DIR / "threshold_sweep.json", "w") as f:
        json.dump(rows, f, indent=2)
    _plot(rows)
    return rows


def _plot(rows: list[dict]) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    thresholds = [row["threshold"] for row in rows]
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(thresholds, [row["escalation_rate"] for row in rows], marker="o", color="#1f77b4", label="Escalation rate")
    ax1.set_xlabel("GB escalation threshold")
    ax1.set_ylabel("Escalation rate", color="#1f77b4")
    ax1.set_ylim(0, 1)
    ax2 = ax1.twinx()
    ax2.plot(thresholds, [row["cascade_ring_recall"] or 0 for row in rows], marker="s", color="#d62728", label="Cascade ring recall")
    ax2.set_ylabel("Cascade ring recall", color="#d62728")
    ax2.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "threshold_sensitivity.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    print(json.dumps(run_threshold_sweep(), indent=2))
