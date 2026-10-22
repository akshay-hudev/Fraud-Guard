"""Equal-budget hyperparameter selection on held-out development worlds.

Every method gets the same budget (up to ``BUDGET`` random configurations,
the first being the default) and is scored by mean validation average
precision over development worlds whose generator seeds are never used for
evaluation. The test split of the development worlds is not used.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..ledger import RESULTS_DIR, provenance, write_json
from ..models.train import fit_graph, fit_mlp, fit_tabular
from ..synth.generator import GeneratorConfig, generate
from .synthetic import (ALL_METHODS, GRAPH_METHODS, HYBRID_METHODS, TABULAR_METHODS, augment_claims,
                        feature_matrix, prepare)
from ..synth.graph import build_metapath_graphs

BUDGET = 10
HYBRID_BASE = {"SAGE+AG": "SAGE", "HGT+AG": "HGT"}
DEV_SEEDS = [901, 902]
DEV_REGIME = {"ring_shift": 0.5, "ring_density": 0.5}

GB_SPACE = {"learning_rate": [0.03, 0.1, 0.2], "max_leaf_nodes": [15, 31, 63], "min_samples_leaf": [10, 20, 50],
            "l2": [0.0, 1.0], "max_iter": [200, 400]}
GNN_SPACE = {"hidden": [32, 64], "layers": [2, 3], "dropout": [0.1, 0.3], "lr": [1e-3, 3e-3, 1e-2],
             "weight_decay": [1e-5, 1e-4, 1e-3]}
SPACES = {
    "LR-CA": {"C": [1.0, 0.001, 0.01, 0.1, 10.0, 100.0]},
    "RF-CA": {"n_estimators": [300], "max_depth": [None, 8, 16], "min_samples_leaf": [3, 1, 10],
              "max_features": ["sqrt", 0.3]},
    "GB-C": GB_SPACE, "GB-CA": GB_SPACE, "GB-CAG": GB_SPACE,
    "MLP-C": {"hidden": [64, 128], "layers": [2, 3], "dropout": [0.1, 0.3], "lr": [3e-3, 1e-3],
              "weight_decay": [1e-4, 1e-5]},
    "SAGE": {**GNN_SPACE, "aggr": ["sum", "mean"]},
    "RGCN": {**GNN_SPACE, "aggr": ["sum", "mean"]},
    "HAN": {**GNN_SPACE, "heads": [2, 4]},
    "SimpleHGN": {**GNN_SPACE, "heads": [2, 4]},
    "HGT": {**GNN_SPACE, "heads": [2, 4]},
    "SAGE+AG": {**GNN_SPACE, "aggr": ["sum", "mean"]},
    "HGT+AG": {**GNN_SPACE, "heads": [2, 4]},
}
DEFAULTS = {
    "GNN": {"hidden": 64, "layers": 2, "dropout": 0.1, "lr": 3e-3, "weight_decay": 1e-4},
}


def candidate_configs(method: str, budget: int = BUDGET, seed: int = 0) -> list[dict]:
    space = SPACES[method]
    keys = sorted(space)
    n_total = int(np.prod([len(space[k]) for k in keys]))
    first = {k: space[k][0] for k in keys}
    if method in GRAPH_METHODS or method in HYBRID_METHODS:
        first.update({k: v for k, v in DEFAULTS["GNN"].items() if k in space})
    configs, seen = [first], {json.dumps(first, sort_keys=True, default=str)}
    rng = np.random.default_rng(seed)
    while len(configs) < min(budget, n_total):
        cand = {k: space[k][rng.integers(len(space[k]))] for k in keys}
        key = json.dumps(cand, sort_keys=True, default=str)
        if key not in seen:
            seen.add(key)
            configs.append(cand)
    return configs


def evaluate_config(method: str, hp: dict, prepared: dict) -> float:
    world, tab, cols, graphs = prepared["world"], prepared["tab"], prepared["cols"], prepared["graphs"]
    y = {k: world.y[world.split == k] for k in ["train", "val", "test"]}
    if method in TABULAR_METHODS:
        fam, groups = TABULAR_METHODS[method]
        return fit_tabular(fam, feature_matrix(tab, cols, groups), y, hp, 0)["best_val_ap"]
    if method == "MLP-C":
        return fit_mlp(feature_matrix(tab, cols, "C"), y, hp, 0)["best_val_ap"]
    if method in GRAPH_METHODS:
        g = graphs
        if GRAPH_METHODS[method] == "han":
            prepared.setdefault("meta", build_metapath_graphs(world, graphs))
            g = prepared["meta"]
        return fit_graph(GRAPH_METHODS[method], g, hp, 0)["best_val_ap"]
    if method in HYBRID_METHODS:
        prepared.setdefault("hybrid", augment_claims(graphs, tab, cols))
        return fit_graph(HYBRID_METHODS[method], prepared["hybrid"], hp, 0)["best_val_ap"]
    raise ValueError(method)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=[m for m in ALL_METHODS if m in SPACES and m not in HYBRID_BASE])
    ap.add_argument("--tag", default="all")
    ap.add_argument("--merge", action="store_true", help="merge trial files into best_hyperparameters.json")
    args = ap.parse_args()
    out_dir = RESULTS_DIR / "tuning"
    if args.merge:
        trials = {}
        for f in sorted(out_dir.glob("trials_*.json")):
            trials.update(json.load(open(f))["trials"])
        best = {m: max(t, key=lambda r: r["mean_val_ap"])["hp"] for m, t in trials.items()}
        # Hybrids inherit the tuned configuration of their base architecture (no separate budget).
        for hybrid, base in HYBRID_BASE.items():
            if base in best:
                best[hybrid] = dict(best[base])
        write_json(out_dir / "best_hyperparameters.json", {
            "provenance": provenance({"budget": BUDGET, "dev_seeds": DEV_SEEDS, "dev_regime": DEV_REGIME, "spaces": SPACES,
                                      "hybrids_inherit": HYBRID_BASE}),
            "best": best, "best_mean_val_ap": {m: max(r["mean_val_ap"] for r in t) for m, t in trials.items()
                                               if m not in HYBRID_BASE}})
        print(json.dumps(best, indent=1))
        return
    prepared = []
    for seed in DEV_SEEDS:
        world = generate(GeneratorConfig(seed=seed, **DEV_REGIME))
        tab, cols, graphs = prepare(world)
        prepared.append({"world": world, "tab": tab, "cols": cols, "graphs": graphs})
    trials = {}
    for method in args.methods:
        trials[method] = []
        for hp in candidate_configs(method):
            aps = [evaluate_config(method, hp, p) for p in prepared]
            trials[method].append({"hp": hp, "val_ap": aps, "mean_val_ap": float(np.mean(aps))})
            print(f"{method:10s} {np.mean(aps):.4f} {hp}", flush=True)
        write_json(out_dir / f"trials_{args.tag}.json", {
            "provenance": provenance({"budget": BUDGET, "dev_seeds": DEV_SEEDS, "dev_regime": DEV_REGIME, "methods": args.methods}),
            "trials": trials})


if __name__ == "__main__":
    main()
