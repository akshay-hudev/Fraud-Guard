"""Real-data experiment: provider-level fraud on the Kaggle healthcare provider dataset.

Protocol
* Evaluation: stratified 5-fold rotation over providers (fold k = test, fold
  k+1 = validation, remaining folds = training), so every provider is tested once.
* Hyperparameters: one separate stratified 60/20/20 split (seed 1000); every
  method gets the same budget of configurations, scored by validation AP.
* All providers' claims define the graph (labels of validation/test providers
  are hidden). Label-neighbourhood features (L) use training labels only.
* Robustness check (``--stage nested``, tabular and MLP models): the configuration
  is re-selected inside every rotation on that rotation's validation fold, so no
  provider used for selection is ever a test provider of the same rotation.
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
OUT_NESTED = RESULTS_DIR / "kaggle_nested"
OUT_REPEATED = RESULTS_DIR / "kaggle_repeated"
NESTED_METHODS = ["LR-P", "RF-P", "GB-P", "GB-PG", "GB-PGL", "MLP-P"]
BUDGET = 6
TABULAR = {"LR-P": ("lr", "P"), "RF-P": ("rf", "P"), "GB-P": ("gb", "P"), "GB-PG": ("gb", "PG"), "GB-PGL": ("gb", "PGL")}
GRAPH = {"SAGE": ("sage", "P"), "RGCN": ("rgcn", "P"), "HAN": ("han", "P"), "SimpleHGN": ("simplehgn", "P"),
         "HGT": ("hgt", "P"), "SAGE+G": ("sage", "PG"), "HGT+G": ("hgt", "PG"),
         # Fraud-specialised GNNs on the provider-only meta-path graph that HAN uses.
         "BWGNN": ("bwgnn", "P"), "CARE-GNN": ("care", "P")}
ALL_METHODS = list(TABULAR) + ["MLP-P"] + list(GRAPH)
HYBRID_BASE = {"SAGE+G": "SAGE", "HGT+G": "HGT"}
SPACE_OF = {"LR-P": "LR-CA", "RF-P": "RF-CA", "GB-P": "GB-CA", "GB-PG": "GB-CA", "GB-PGL": "GB-CA",
            "MLP-P": "MLP-C", "SAGE": "SAGE", "RGCN": "RGCN", "HAN": "HAN", "SimpleHGN": "SimpleHGN",
            "HGT": "HGT", "SAGE+G": "SAGE+AG", "HGT+G": "HGT+AG", "BWGNN": "BWGNN", "CARE-GNN": "CARE-GNN"}


def splits(repeat: int = 0) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Five rotations over stratified provider folds; ``repeat`` reshuffles the folds (0 = the reported split)."""
    data_y = _DATA.providers["y"].to_numpy()
    fold = np.empty(len(data_y), int)
    for k, (_, test) in enumerate(StratifiedKFold(5, shuffle=True, random_state=repeat).split(data_y, data_y)):
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
    g = (build_metapath_graph if kind in ("han", "bwgnn", "care") else build_graph)(data, X, y)
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


def nested(methods, folds, data_meta: dict, tag: str = "all") -> None:
    """Per-rotation selection on the rotation's own validation fold, then one test evaluation.

    Hybrids inherit the configuration selected for their base GNN in the same rotation when that base is
    in ``methods`` (as in the shared tuning); otherwise they get their own budget.
    """
    y = _DATA.providers["y"].to_numpy()
    for k in folds:
        train, val, test = splits()[k]
        rows, chosen = [], {}
        for name in sorted(methods, key=lambda m: m in HYBRID_BASE):        # base models before hybrids
            base = HYBRID_BASE.get(name)
            configs = [chosen[base]] if base in chosen else candidate_configs(SPACE_OF[name], budget=BUDGET)
            best, best_hp = None, None
            for hp in configs:
                res = run_method(name, hp, train, val, test)
                if best is None or res["best_val_ap"] > best["best_val_ap"] + 1e-12:
                    best, best_hp = res, hp
            chosen[name] = best_hp
            m = evaluate_scores(y[val], best["val_scores"], y[test], best["test_scores"], budget=0.10)
            rows.append({"method": name, "fold": k, "hp": best_hp, **m, "best_val_ap": best["best_val_ap"]})
            print(f"nested fold {k} {name:8s} AP {m['ap']:.4f} {best_hp}", flush=True)
        write_json(OUT_NESTED / "runs" / (f"fold{k}.json" if tag == "all" else f"fold{k}_{tag}.json"),
                   {"provenance": provenance({"fold": k, "methods": methods, "budget": BUDGET}, data=data_meta),
                    "rows": rows})


def main():
    global _DATA
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["tune", "merge", "eval", "nested"], required=True)
    ap.add_argument("--methods", nargs="+", default=ALL_METHODS)
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--tag", default="all")
    ap.add_argument("--repeat", type=int, default=0,
                    help="with --stage eval: reshuffle the folds (seed = repeat); 0 = the reported split")
    ap.add_argument("--nested-gnns", action="store_true",
                    help="with --stage nested: also re-select the GNNs and hybrids per rotation (about 25 CPU-hours)")
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
    if args.stage == "nested":
        allowed = ALL_METHODS if args.nested_gnns else NESTED_METHODS
        nested([m for m in args.methods if m in allowed], args.folds, data_meta, tag=args.tag)
        return
    if args.stage == "tune":
        trials = tune([m for m in args.methods if m not in HYBRID_BASE])
        write_json(OUT / "tuning" / f"trials_{args.tag}.json",
                   {"provenance": provenance({"budget": BUDGET, "split_seed": 1000, "methods": args.methods}, data=data_meta),
                    "trials": trials})
        return
    best = json.load(open(OUT / "tuning" / "best_hyperparameters.json"))["best"]
    y = _DATA.providers["y"].to_numpy()
    # Repeated cross-validation: reshuffled folds go to their own directory; repeat 0 keeps the reported paths.
    out, prefix = (OUT, "") if args.repeat == 0 else (OUT_REPEATED, f"r{args.repeat}_")
    for k in args.folds:
        train, val, test = splits(args.repeat)[k]
        rows, preds = [], {"provider_id": _DATA.providers["provider_id"].to_numpy()[test], "y": y[test]}
        for name in args.methods:
            res = run_method(name, best[name], train, val, test)
            m = evaluate_scores(y[val], res["val_scores"], y[test], res["test_scores"], budget=0.10)
            rows.append({"method": name, "fold": k, "repeat": args.repeat, **m, "best_val_ap": res["best_val_ap"],
                         "train_seconds": res["train_seconds"], "infer_seconds": res["infer_seconds"],
                         "best_epoch": res.get("best_epoch")})
            preds[name] = res["test_scores"].astype(np.float32)
            print(f"fold {k} {name:10s} AP {m['ap']:.4f} ROC {m['roc_auc']:.4f} F1 {m['f1']:.4f}", flush=True)
        (out / "predictions").mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out / "predictions" / f"{prefix}fold{k}_{args.tag}.npz", **preds)
        write_json(out / "runs" / f"{prefix}fold{k}_{args.tag}.json",
                   {"provenance": provenance({"fold": k, "repeat": args.repeat, "methods": args.methods},
                                             hyperparameters={m: best[m] for m in args.methods}, data=data_meta),
                    "rows": rows})


_DATA: KaggleData | None = None

if __name__ == "__main__":
    main()
