"""Real-data experiment: provider-level fraud on the Kaggle healthcare provider dataset.

Protocol
* Evaluation: stratified 5-fold rotation over providers (fold k = test, fold
  k+1 = validation, remaining folds = training), so every provider is tested once.
* Hyperparameters: one separate stratified 60/20/20 split (seed 1000); every
  method gets the same budget of configurations, scored by validation AP.
* All providers' claims define the graph (labels of validation/test providers
  are hidden). Label-neighbourhood features (L) use training labels only.
"""

from __future__ import annotations

import argparse
import copy
import json
import time

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split

from ..ledger import RESULTS_DIR, provenance, stable_hash, write_json
from ..metrics import evaluate_scores
from ..models.train import fit_graph, fit_mlp, fit_tabular
from ..realdata.kaggle import KaggleData, build_graph, build_metapath_graph, label_features, load
from .tune import SPACES, candidate_configs

OUT = RESULTS_DIR / "kaggle"
BUDGET = 6
TABULAR = {"LR-P": ("lr", "P"), "RF-P": ("rf", "P"), "GB-P": ("gb", "P"), "GB-PG": ("gb", "PG"), "GB-PGL": ("gb", "PGL")}
GRAPH = {"SAGE": ("sage", "P"), "RGCN": ("rgcn", "P"), "HAN": ("han", "P"), "SimpleHGN": ("simplehgn", "P"),
         "HGT": ("hgt", "P"), "SAGE+G": ("sage", "PG"), "HGT+G": ("hgt", "PG")}
ALL_METHODS = list(TABULAR) + ["MLP-P"] + list(GRAPH)
HYBRID_BASE = {"SAGE+G": "SAGE", "HGT+G": "HGT"}
SPACE_OF = {"LR-P": "LR-CA", "RF-P": "RF-CA", "GB-P": "GB-CA", "GB-PG": "GB-CA", "GB-PGL": "GB-CA",
            "MLP-P": "MLP-C", "SAGE": "SAGE", "RGCN": "RGCN", "HAN": "HAN", "SimpleHGN": "SimpleHGN",
            "HGT": "HGT", "SAGE+G": "SAGE+AG", "HGT+G": "HGT+AG"}


def splits() -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    data_y = _DATA.providers["y"].to_numpy()
    fold = np.empty(len(data_y), int)
    for k, (_, test) in enumerate(StratifiedKFold(5, shuffle=True, random_state=0).split(data_y, data_y)):
        fold[test] = k
    out = []
    for k in range(5):
        test = np.flatnonzero(fold == k)
        val = np.flatnonzero(fold == (k + 1) % 5)
        train = np.flatnonzero((fold != k) & (fold != (k + 1) % 5))
        out.append((train, val, test))
    return out


def tuning_split():
    y = _DATA.providers["y"].to_numpy()
    idx = np.arange(len(y))
    tr, rest = train_test_split(idx, test_size=0.4, stratify=y, random_state=1000)
    va, te = train_test_split(rest, test_size=0.5, stratify=y[rest], random_state=1000)
    return np.sort(tr), np.sort(va), np.sort(te)


def features(data: KaggleData, groups: str, train: np.ndarray) -> np.ndarray:
    parts = []
    if "P" in groups:
        parts.append(data.P.to_numpy(float))
    if "G" in groups:
        parts.append(data.G.to_numpy(float))
    if "L" in groups:
        parts.append(label_features(data, train).to_numpy(float))
    X = np.hstack(parts)
    mu, sd = X[train].mean(0), X[train].std(0)
    return (X - mu) / np.where(sd > 0, sd, 1.0)


def _masked(g, rows, n):
    g2 = copy.copy(g)
    m = torch.zeros(n, dtype=torch.bool)
    m[rows] = True
    g2["provider"].target_mask = m
    return g2


def run_method(name: str, hp: dict, train, val, test, seed: int = 0) -> dict:
    data = _DATA
    y = data.providers["y"].to_numpy()
    for idx in (train, val, test):
        assert (np.diff(idx) > 0).all(), "split indices must be sorted to align with node order"
    if name in TABULAR:
        fam, groups = TABULAR[name]
        X = features(data, groups, train)
        return fit_tabular(fam, {"train": X[train], "val": X[val], "test": X[test]},
                           {"train": y[train], "val": y[val], "test": y[test]}, hp, seed)
    if name == "MLP-P":
        X = features(data, "P", train)
        return fit_mlp({"train": X[train], "val": X[val], "test": X[test]},
                       {"train": y[train], "val": y[val], "test": y[test]}, hp, seed)
    kind, groups = GRAPH[name]
    X = features(data, groups, train)
    g = (build_metapath_graph if kind == "han" else build_graph)(data, X, y)
    graphs = {k: _masked(g, rows, len(y)) for k, rows in [("train", train), ("val", val), ("test", test)]}
    return fit_graph(kind, graphs, hp, seed, max_epochs=100, patience=10, target="provider")


def tune(methods):
    tr, va, te = tuning_split()
    trials = {}
    for name in methods:
        trials[name] = []
        for hp in candidate_configs(SPACE_OF[name], budget=BUDGET):
            t0 = time.time()
            res = run_method(name, hp, tr, va, te)
            trials[name].append({"hp": hp, "val_ap": res["best_val_ap"], "seconds": time.time() - t0})
            print(f"{name:10s} val AP {res['best_val_ap']:.4f} ({time.time() - t0:.0f}s) {hp}", flush=True)
    return trials


def main():
    global _DATA
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["tune", "merge", "eval"], required=True)
    ap.add_argument("--methods", nargs="+", default=ALL_METHODS)
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--tag", default="all")
    args = ap.parse_args()
    if args.stage == "merge":
        trials = {}
        for f in sorted((OUT / "tuning").glob("trials_*.json")):
            trials.update(json.load(open(f))["trials"])
        best = {m: max(t, key=lambda r: r["val_ap"])["hp"] for m, t in trials.items()}
        for hybrid, base in HYBRID_BASE.items():          # hybrids inherit their base configuration
            if base in best:
                best[hybrid] = dict(best[base])
        write_json(OUT / "tuning" / "best_hyperparameters.json",
                   {"provenance": provenance({"budget": BUDGET, "split_seed": 1000}), "best": best})
        print(json.dumps(best, indent=1))
        return
    _DATA = load()
    data_meta = {"n_providers": len(_DATA.providers), "n_fraud": int(_DATA.providers["y"].sum()),
                 "P_hash": stable_hash(_DATA.P.round(8).to_numpy().tolist()), "files": "data/external/kaggle_provider_fraud"}
    if args.stage == "tune":
        trials = tune([m for m in args.methods if m not in HYBRID_BASE])
        write_json(OUT / "tuning" / f"trials_{args.tag}.json",
                   {"provenance": provenance({"budget": BUDGET, "split_seed": 1000, "methods": args.methods}, data=data_meta),
                    "trials": trials})
        return
    best = json.load(open(OUT / "tuning" / "best_hyperparameters.json"))["best"]
    y = _DATA.providers["y"].to_numpy()
    for k in args.folds:
        train, val, test = splits()[k]
        rows, preds = [], {"provider_id": _DATA.providers["provider_id"].to_numpy()[test], "y": y[test]}
        for name in args.methods:
            res = run_method(name, best[name], train, val, test)
            m = evaluate_scores(y[val], res["val_scores"], y[test], res["test_scores"], budget=0.10)
            rows.append({"method": name, "fold": k, **m, "best_val_ap": res["best_val_ap"],
                         "train_seconds": res["train_seconds"], "infer_seconds": res["infer_seconds"],
                         "best_epoch": res.get("best_epoch")})
            preds[name] = res["test_scores"].astype(np.float32)
            print(f"fold {k} {name:10s} AP {m['ap']:.4f} ROC {m['roc_auc']:.4f} F1 {m['f1']:.4f}", flush=True)
        (OUT / "predictions").mkdir(parents=True, exist_ok=True)
        np.savez_compressed(OUT / "predictions" / f"fold{k}_{args.tag}.npz", **preds)
        write_json(OUT / "runs" / f"fold{k}_{args.tag}.json",
                   {"provenance": provenance({"fold": k, "methods": args.methods},
                                             hyperparameters={m: best[m] for m in args.methods}, data=data_meta),
                    "rows": rows})


_DATA: KaggleData | None = None

if __name__ == "__main__":
    main()
