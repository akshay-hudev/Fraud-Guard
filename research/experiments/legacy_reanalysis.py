"""Re-analysis of a small fixed benchmark: the 2,100-claim dataset of the repository's simpler generator.

Data: data/raw/{hospitals,doctors,patients,claims}.csv, written by training/generate_dataset.py
(seed 42); their SHA-256 is checked against archive/legacy_unverified/logs/dataset_config.json.
The dataset is evaluated with its own split (claims sorted by date, positional 60/20/20), and
the script asks whether this dataset can separate coordination from claim-level anomaly:
* composition and ring topology (members, block density, active span);
* how anomalous ring claims are on their own (standardised log-amount shift, single-feature ROC-AUC);
* how many ring claims the test split holds and whether their doctors or patients were already
  ring members with labelled claims in training;
* gradient boosting on three raw claim attributes only (no aggregates, no graph), with the
  benchmark's tuned GB-C configuration: ring AP and ring recall on the test split, with exact
  binomial and bootstrap intervals.
The same claim-level statistics are computed on benchmark test periods for comparison.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

from ..ledger import REPO_ROOT, RESULTS_DIR, provenance, write_json
from ..metrics import _ap, evaluate_scores
from ..models.train import fit_tabular
from ..synth.features import claim_level
from ..synth.generator import GeneratorConfig, generate
from .synthetic import load_hps

RAW = REPO_ROOT / "data" / "raw"
CONFIG = REPO_ROOT / "archive" / "legacy_unverified" / "logs" / "dataset_config.json"
OUT = RESULTS_DIR / "legacy"
RAW_FEATURES = ["log_amount", "num_procedures", "days_in_hospital"]


def dataset_sha256() -> str:
    h = hashlib.sha256()
    for name in ["hospitals.csv", "doctors.csv", "patients.csv", "claims.csv"]:
        h.update((RAW / name).read_bytes())
    return h.hexdigest()


def separability(log_amount: np.ndarray, frame: pd.DataFrame, fraud_type: np.ndarray) -> dict:
    ring, legit = fraud_type == "ring", fraud_type == "legit"
    keep = ring | legit
    out = {"log_amount_shift_sd": float((log_amount[ring].mean() - log_amount[legit].mean()) / log_amount[legit].std())}
    for col in ["claim_amount", "num_procedures", "days_in_hospital"]:
        out[f"auc_{col}"] = float(roc_auc_score(ring[keep], frame[col].to_numpy()[keep]))
    return out


def clopper_pearson(k: int, n: int, level: float = 0.95) -> tuple[float, float]:
    a = (1 - level) / 2
    lo = stats.beta.ppf(a, k, n - k + 1) if k > 0 else 0.0
    hi = stats.beta.ppf(1 - a, k + 1, n - k) if k < n else 1.0
    return float(lo), float(hi)


def legacy() -> dict:
    claims = pd.read_csv(RAW / "claims.csv")
    claims = claims.sort_values("claim_date", kind="stable").reset_index(drop=True)
    ft = np.where(claims["ring_id"] > 0, "ring", np.where(claims["fraud_label"] == 1, "isolated", "legit"))
    n = len(claims)
    cut1, cut2 = int(0.6 * n), int(0.8 * n)
    split = np.array(["train"] * cut1 + ["val"] * (cut2 - cut1) + ["test"] * (n - cut2))
    log_amount = np.log(claims["claim_amount"].to_numpy())

    ring = claims[ft == "ring"]
    dates = pd.to_datetime(ring["claim_date"])
    topology = []
    for r, g in ring.groupby("ring_id"):
        span = pd.to_datetime(g["claim_date"])
        topology.append({"ring_id": int(r), "claims": int(len(g)), "doctors": int(g["doctor_id"].nunique()),
                         "patients": int(g["patient_id"].nunique()),
                         "block_density": float(g[["patient_id", "doctor_id"]].drop_duplicates().shape[0]
                                                / (g["patient_id"].nunique() * g["doctor_id"].nunique())),
                         "active_months": float((span.max() - span.min()).days / 30.44)})

    comp = {s: {k: int(((split == s) & (ft == k)).sum()) for k in ["legit", "isolated", "ring"]}
            for s in ["train", "val", "test"]}
    train_ring = claims[(split == "train") & (ft == "ring")]
    test_ring = claims[(split == "test") & (ft == "ring")]
    seen = {"doctor": float(test_ring["doctor_id"].isin(train_ring["doctor_id"]).mean()),
            "patient": float(test_ring["patient_id"].isin(train_ring["patient_id"]).mean())}

    X_all = np.column_stack([log_amount, claims["num_procedures"], claims["days_in_hospital"]]).astype(float)
    X = {s: X_all[split == s] for s in ["train", "val", "test"]}
    y = {s: claims["fraud_label"].to_numpy()[split == s] for s in ["train", "val", "test"]}
    hp = load_hps(RESULTS_DIR / "tuning" / "best_hyperparameters.json")["GB-C"]
    res = fit_tabular("gb", X, y, hp, 0)
    ft_test = ft[split == "test"]
    m = evaluate_scores(y["val"], res["val_scores"], y["test"], res["test_scores"], ft_test)
    n_ring = int((ft_test == "ring").sum())
    k_ring = int(round(m["ring_recall"] * n_ring))
    rng = np.random.default_rng(0)
    keep = (ft_test == "legit") | (ft_test == "ring")
    yr, sr = (ft_test[keep] == "ring").astype(int), res["test_scores"][keep]
    boot = []
    for _ in range(2000):
        i = rng.integers(0, len(yr), len(yr))
        boot.append(_ap(yr[i], sr[i]))
    return {
        "sha256": dataset_sha256(), "sha256_expected": json.load(open(CONFIG))["dataset_sha256"],
        "n_claims": n, "n_patients": int(claims["patient_id"].nunique()), "n_doctors": int(claims["doctor_id"].nunique()),
        "composition": {k: int((ft == k).sum()) for k in ["legit", "isolated", "ring"]},
        "ring_topology": topology,
        "ring_span_months": float((dates.max() - dates.min()).days / 30.44),
        "separability": separability(log_amount, claims, ft),
        "split_composition": comp,
        "test_ring_claims_per_ring": test_ring.groupby("ring_id").size().astype(int).to_dict(),
        "test_ring_members_seen_in_training_rings": seen,
        "gb_raw_claim_features": {
            "features": RAW_FEATURES, "hyperparameters": hp,
            **{k: m[k] for k in ["ap", "ring_ap", "isolated_ap", "ring_recall", "isolated_recall", "f1", "threshold"]},
            "ring_recall_counts": [k_ring, n_ring], "ring_recall_ci": clopper_pearson(k_ring, n_ring),
            "ring_ap_bootstrap_ci": [float(np.nanpercentile(boot, 2.5)), float(np.nanpercentile(boot, 97.5))]},
    }


def benchmark(shifts=(0.0, 1.0, 2.0), seeds=range(1, 7), density: float = 0.5) -> dict:
    out = {}
    for shift in shifts:
        stats_ = []
        for seed in seeds:
            w = generate(GeneratorConfig(ring_shift=shift, ring_density=density, seed=seed))
            test = w.split == "test"
            ft = w.truth["fraud_type"].to_numpy()[test]
            frame = w.claims[test].reset_index(drop=True)
            s = separability(claim_level(w)["log_amount"].to_numpy()[test], frame, ft)
            s["test_ring_claims"] = int((ft == "ring").sum())
            s["n_claims"], s["n_test_claims"] = int(len(w.claims)), int(test.sum())
            stats_.append(s)
        out[f"delta_{shift:g}"] = {k: float(np.mean([s[k] for s in stats_])) for k in stats_[0]}
    return out


def main():
    result = {"legacy": legacy(), "benchmark_s0.5": benchmark()}
    write_json(OUT / "legacy_reanalysis.json", {"provenance": provenance({"analysis": "legacy_reanalysis"}), **result})
    print(json.dumps(result, indent=1, default=str))


if __name__ == "__main__":
    main()
