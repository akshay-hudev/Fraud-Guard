"""
Threshold sensitivity analysis for hybrid GB -> HGT routing.

Sweeps the Gradient Boosting escalation threshold and reports how much traffic
is escalated to HGT, the ring-fraud recall on escalated claims, and the blended
system latency.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TRAINING_DIR = Path(__file__).resolve().parent
REPO_ROOT = TRAINING_DIR.parent
DATA_DIR = TRAINING_DIR / "data" / "processed"
MODEL_DIR = TRAINING_DIR / "models"
LOG_DIR = REPO_ROOT / "logs"
FIGURE_DIR = REPO_ROOT / "figures"

THRESHOLDS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
GB_P50_MS = 30.0
HGT_P50_MS = 50.0


def _load_hgt_scores(n_test: int) -> np.ndarray | None:
    candidates = [
        MODEL_DIR / "gnn" / "test_predictions.json",
        REPO_ROOT / "models" / "gnn" / "test_predictions.json",
    ]
    for path in candidates:
        if not path.exists():
            continue
        with open(path) as f:
            payload = json.load(f)
        scores = np.asarray(payload.get("y_prob", []), dtype=float)
        if len(scores) == n_test:
            return scores
    return None


def _load_test_claims(n_test: int) -> pd.DataFrame:
    candidates = [
        DATA_DIR / "features_raw.csv",
        DATA_DIR / "claims_engineered.csv",
        TRAINING_DIR / "data" / "raw" / "claims.csv",
    ]
    for path in candidates:
        if not path.exists():
            continue
        claims = pd.read_csv(path)
        if "claim_date" in claims.columns:
            claims["claim_date"] = pd.to_datetime(claims["claim_date"])
            claims = claims.sort_values("claim_date").reset_index(drop=True)
        if len(claims) >= n_test:
            return claims.iloc[-n_test:].reset_index(drop=True)
    return pd.DataFrame(index=range(n_test))


def _ring_mask(test_claims: pd.DataFrame, y_test: np.ndarray) -> np.ndarray:
    if "ring_id" in test_claims.columns:
        return test_claims["ring_id"].fillna(0).astype(int).to_numpy() > 0
    return y_test.astype(int) == 1


def run_threshold_sweep() -> list[dict]:
    X_test = np.load(DATA_DIR / "X_test.npy")
    y_test = np.load(DATA_DIR / "y_test.npy")
    gb_model = joblib.load(MODEL_DIR / "baseline" / "gradient_boosting.pkl")
    gb_scores = gb_model.predict_proba(X_test)[:, 1]
    hgt_scores = _load_hgt_scores(len(y_test))
    if hgt_scores is None:
        hgt_scores = gb_scores

    test_claims = _load_test_claims(len(y_test))
    ring_mask = _ring_mask(test_claims, y_test)
    hgt_pred = hgt_scores >= 0.5

    rows = []
    for threshold in THRESHOLDS:
        escalated = gb_scores > threshold
        escalation_rate = float(escalated.mean())
        escalated_ring = escalated & ring_mask & (y_test.astype(int) == 1)
        if escalated_ring.sum() == 0:
            ring_recall = 0.0
        else:
            ring_recall = float((hgt_pred[escalated_ring] == 1).mean())
        system_latency = (1 - escalation_rate) * GB_P50_MS + escalation_rate * HGT_P50_MS
        rows.append({
            "threshold": threshold,
            "escalation_rate": round(escalation_rate, 4),
            "system_ring_recall": round(ring_recall, 4),
            "system_mean_latency_ms": round(system_latency, 2),
        })

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_DIR / "threshold_sweep.json", "w") as f:
        json.dump(rows, f, indent=2)

    import matplotlib.pyplot as plt
    import os
    os.makedirs("figures", exist_ok=True)

    thresholds = [entry["threshold"] for entry in rows]
    esc_rates = [entry["escalation_rate"] for entry in rows]
    ring_recalls = [entry["system_ring_recall"] for entry in rows]
    latencies = [
        entry.get("system_mean_latency", entry.get("system_mean_latency_ms"))
        for entry in rows
    ]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.set_xlabel("GB Escalation Threshold")
    ax1.set_ylabel("Escalation Rate", color="steelblue")
    ax1.plot(thresholds, esc_rates, "o-", color="steelblue",
             label="Escalation Rate")
    ax1.tick_params(axis="y", labelcolor="steelblue")
    ax1.set_ylim(0, 1)

    ax2 = ax1.twinx()
    ax2.set_ylabel("System Ring Recall", color="darkorange")
    ax2.plot(thresholds, ring_recalls, "s--", color="darkorange",
             label="Ring Recall")
    ax2.tick_params(axis="y", labelcolor="darkorange")
    ax2.set_ylim(0, 1)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")
    plt.title("Hybrid Pipeline: Threshold vs Escalation Rate and Ring Recall")
    plt.tight_layout()
    plt.savefig("figures/threshold_sensitivity.png", dpi=150)
    plt.close()
    print("Figure saved to figures/threshold_sensitivity.png")
    _plot(rows)
    return rows


def _plot(rows: list[dict]) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    thresholds = [row["threshold"] for row in rows]
    escalation = [row["escalation_rate"] for row in rows]
    recall = [row["system_ring_recall"] for row in rows]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(thresholds, escalation, marker="o", color="#1f77b4", label="Escalation rate")
    ax1.set_xlabel("GB escalation threshold")
    ax1.set_ylabel("Escalation rate", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.set_ylim(0, 1)

    ax2 = ax1.twinx()
    ax2.plot(thresholds, recall, marker="s", color="#d62728", label="Ring recall")
    ax2.set_ylabel("Ring recall", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    ax2.set_ylim(0, 1)

    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "threshold_sensitivity.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    run_threshold_sweep()
