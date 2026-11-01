"""Fifth experimental round (paper/hiss_review/analysis_plan_round5.md).

Two questions left open by the earlier rounds:

* ``tune`` / ``merge`` / ``eval``: does the GNNs' deficit survive a search far wider than the ten candidates reused by
  every earlier tuning? Fresh candidates from a wider grid, more epochs, and selection and early stopping on
  validation ring AP (the primary metric) instead of overall AP. GB-CAG gets the same budget. Development worlds
  (seeds 901, 902) are drawn at the evaluation regime, so the search is as favourable as the design allows.
* ``closure``: is the shared-patient closure, a count of four-cycles that message passing cannot compute in general,
  what the pure GNNs lack? Two closure columns are given to a GNN and to GB-CA, and removed from GB-CAG.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
from sklearn.metrics import average_precision_score

from ..ledger import RESULTS_DIR, frame_hash, provenance, write_json
from ..metrics import evaluate_scores
from ..models.train import fit_graph, fit_tabular
from ..synth.features import A_COLUMNS, G_COLUMNS, c_columns
from ..synth.generator import GeneratorConfig, generate
from .synthetic import GRAPH_METHODS, augment_claims, feature_matrix, prepare, run_tag
from .tune import DEV_SEEDS, GB_SPACE, candidate_configs

OUT = RESULTS_DIR / "expanded"
OUT_CLOSURE = RESULTS_DIR / "closure_columns"
METHODS = ["GB-CAG", "SAGE", "RGCN", "SimpleHGN", "HGT"]
# Evaluation cells: (claims per ring per month, structural density), all at delta = 0.
REGIMES = {"v4_s0.5": (4.0, 0.5), "v4_s0.9": (4.0, 0.9), "v16_s0.9": (16.0, 0.9)}
EVAL_SEEDS = [1, 2, 3, 4, 5, 6]
BUDGET = 30
TOP = 6                     # candidates that advance from the first development world to the second
SAMPLING_SEED = 1
MAX_EPOCHS, PATIENCE = 300, 30
GNN_WIDE = {"hidden": [16, 32, 64, 128], "layers": [2, 3, 4], "dropout": [0.0, 0.1, 0.3],
            "lr": [3e-4, 1e-3, 3e-3, 1e-2], "weight_decay": [1e-5, 1e-4, 1e-3]}
WIDE_SPACES = {
    "GB-CAG": {"learning_rate": [0.01, 0.03, 0.1, 0.2], "max_leaf_nodes": [7, 15, 31, 63, 127],
               "min_samples_leaf": [5, 10, 20, 50], "l2": [0.0, 1.0, 10.0], "max_iter": [200, 400, 800]},
    "SAGE": {**GNN_WIDE, "aggr": ["sum", "mean"]},
    "RGCN": {**GNN_WIDE, "aggr": ["sum", "mean"]},
    "SimpleHGN": {**GNN_WIDE, "heads": [2, 4, 8]},
    "HGT": {**GNN_WIDE, "heads": [2, 4, 8]},
}
CLOSURE = ["claim_closure", "claim_closure_norm"]
CLOSURE_VOLUMES, CLOSURE_DENSITIES = [8.0, 4.0], [0.5, 0.9]


def expanded_candidates(method: str, main_hp: dict, budget: int = BUDGET, seed: int = SAMPLING_SEED) -> list[dict]:
    """The main configuration, then fresh draws from the wide grid that no earlier tuning offered."""
    space = WIDE_SPACES[method]
    keys = sorted(space)
    fmt = lambda hp: json.dumps(hp, sort_keys=True, default=str)
    seen = {fmt(c) for c in candidate_configs(method)} | {fmt(main_hp)}
    configs, rng = [dict(main_hp)], np.random.default_rng(seed)
    while len(configs) < budget:
        cand = {k: space[k][rng.integers(len(space[k]))] for k in keys}
        if fmt(cand) not in seen:
            seen.add(fmt(cand))
            configs.append(cand)
    return configs


def ring_val_keep(world) -> np.ndarray:
    """Validation claims that count for ring AP: ring and legitimate, without isolated fraud."""
    return world.truth["fraud_type"].to_numpy()[world.split == "val"] != "isolated"


def fit_method(method: str, world, tab, cols, graphs, hp: dict, model_seed: int = 0) -> dict:
    """Fit one method; ``best_val_ap`` is validation ring AP."""
    y = {k: world.y[world.split == k] for k in ["train", "val", "test"]}
    keep = ring_val_keep(world)
    if method == "GB-CAG":
        res = fit_tabular("gb", feature_matrix(tab, cols, "CAG"), y, hp, model_seed)
        res["best_val_ap"] = float(average_precision_score(y["val"][keep], res["val_scores"][keep]))
        return res
    return fit_graph(GRAPH_METHODS[method], graphs, hp, model_seed, max_epochs=MAX_EPOCHS, patience=PATIENCE, val_keep=keep)


def dev_world(regime: str, seed: int) -> tuple:
    volume, density = REGIMES[regime]
    world = generate(GeneratorConfig(seed=seed, ring_shift=0.0, ring_density=density, ring_claims_per_month=volume))
    return (world, *prepare(world))


def tune(regime: str, method: str, main_hps: dict) -> None:
    """Every candidate on the first development world; the best ``TOP`` also on the second; restartable."""
    path = OUT / "tuning" / f"{regime}__{method}.json"
    trials = json.load(open(path))["trials"] if path.exists() else []
    meta = {"regime": regime, "method": method, "budget": BUDGET, "top": TOP, "sampling_seed": SAMPLING_SEED,
            "dev_seeds": DEV_SEEDS, "space": WIDE_SPACES[method], "max_epochs": MAX_EPOCHS, "patience": PATIENCE}

    def save():
        write_json(path, {"provenance": provenance(meta), "trials": trials})

    first = dev_world(regime, DEV_SEEDS[0])
    for hp in expanded_candidates(method, main_hps[method])[len(trials):]:
        res = fit_method(method, *first, hp)
        trials.append({"hp": hp, "val_ring_ap": [res["best_val_ap"]], "best_epoch": res.get("best_epoch"),
                       "train_seconds": res["train_seconds"]})
        print(f"{regime} {method} {len(trials)}/{BUDGET} val ring AP {res['best_val_ap']:.4f} "
              f"({res['train_seconds']:.0f}s) {hp}", flush=True)
        save()
    second = dev_world(regime, DEV_SEEDS[1])
    for i in np.argsort([-t["val_ring_ap"][0] for t in trials], kind="stable")[:TOP]:
        t = trials[int(i)]
        if len(t["val_ring_ap"]) == 1:
            res = fit_method(method, *second, t["hp"])
            t["val_ring_ap"].append(res["best_val_ap"])
            print(f"{regime} {method} second world {res['best_val_ap']:.4f} {t['hp']}", flush=True)
            save()


def merge() -> None:
    best = {}
    for regime in REGIMES:
        for method in METHODS:
            t = json.load(open(OUT / "tuning" / f"{regime}__{method}.json"))["trials"]
            done = [r for r in t if len(r["val_ring_ap"]) == 2]
            r = max(done, key=lambda x: np.mean(x["val_ring_ap"]))
            best.setdefault(regime, {})[method] = {"hp": r["hp"], "mean_val_ring_ap": float(np.mean(r["val_ring_ap"])),
                                                   "main_config": r is t[0], "n_trials": len(t)}
    write_json(OUT / "best_hyperparameters.json", {"provenance": provenance({"budget": BUDGET, "top": TOP}), "best": best})
    print(json.dumps(best, indent=1))


def evaluate(regime: str, methods: list[str], seeds: list[int]) -> None:
    best = json.load(open(OUT / "best_hyperparameters.json"))["best"][regime]
    volume, density = REGIMES[regime]
    for seed in seeds:
        run = OUT / "runs" / f"{regime}__g{seed}__{'_'.join(methods)}.json"
        if run.exists():
            continue
        cfg = GeneratorConfig(seed=seed, ring_shift=0.0, ring_density=density, ring_claims_per_month=volume)
        world = generate(cfg)
        tab, cols, graphs = prepare(world)
        y = {k: world.y[world.split == k] for k in ["val", "test"]}
        ft = world.truth["fraud_type"].to_numpy()[world.split == "test"]
        rows, preds = [], {"claim_id": world.claims["claim_id"].to_numpy()[world.split == "test"], "y": y["test"]}
        for method in methods:
            res = fit_method(method, world, tab, cols, graphs, best[method]["hp"])
            m = evaluate_scores(y["val"], res["val_scores"], y["test"], res["test_scores"], ft)
            rows.append({"method": method, "regime": regime, **cfg.as_dict(), **m, "best_val_ring_ap": res["best_val_ap"],
                         "best_epoch": res.get("best_epoch"), "train_seconds": res["train_seconds"]})
            preds[method] = res["test_scores"].astype(np.float32)
            print(f"{regime} g{seed} {method} ring AP {m['ring_ap']:.3f}", flush=True)
        (OUT / "predictions").mkdir(parents=True, exist_ok=True)
        np.savez_compressed(OUT / "predictions" / f"{run.stem}.npz", **preds)
        write_json(run, {"provenance": provenance(cfg.as_dict(), hyperparameters={m: best[m]["hp"] for m in methods},
                                                  claims_hash=frame_hash(world.claims)), "rows": rows})


def closure_worlds(hps: dict, seeds: list[int]) -> None:
    """Closure columns added to SAGE, HGT and GB-CA, and removed from GB-CAG, with the main hyperparameters."""
    for volume in CLOSURE_VOLUMES:
        for density in CLOSURE_DENSITIES:
            for seed in seeds:
                cfg = GeneratorConfig(seed=seed, ring_shift=0.0, ring_density=density, ring_claims_per_month=volume)
                run = OUT_CLOSURE / "runs" / f"{run_tag(cfg, 0)}.json"
                if run.exists():
                    continue
                world = generate(cfg)
                tab, cols, graphs = prepare(world)
                y = {k: world.y[world.split == k] for k in ["train", "val", "test"]}
                ft = world.truth["fraud_type"].to_numpy()[world.split == "test"]
                extra = augment_claims(graphs, tab, {"A": CLOSURE, "G": []})
                plain = [c for c in G_COLUMNS if c not in CLOSURE]
                groups = {"C": c_columns(world), "A": A_COLUMNS, "K": CLOSURE, "P": plain}
                fits = {"SAGE+Cl": lambda: fit_graph("sage", extra, hps["SAGE"], 0),
                        "HGT+Cl": lambda: fit_graph("hgt", extra, hps["HGT"], 0),
                        "GB-CA+Cl": lambda: fit_tabular("gb", feature_matrix(tab, groups, "CAK"), y, hps["GB-CAG"], 0),
                        "GB-CAG-Cl": lambda: fit_tabular("gb", feature_matrix(tab, groups, "CAP"), y, hps["GB-CAG"], 0)}
                rows = []
                for name, fit in fits.items():
                    res = fit()
                    m = evaluate_scores(y["val"], res["val_scores"], y["test"], res["test_scores"], ft)
                    rows.append({"method": name, **cfg.as_dict(), **m})
                    print(f"v{volume:g} s{density} g{seed} {name} ring AP {m['ring_ap']:.3f}", flush=True)
                write_json(run, {"provenance": provenance(cfg.as_dict(), closure=CLOSURE), "rows": rows})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["tune", "merge", "eval", "closure"], required=True)
    ap.add_argument("--regime", choices=list(REGIMES))
    ap.add_argument("--methods", nargs="+", default=METHODS)
    ap.add_argument("--seeds", type=int, nargs="+", default=EVAL_SEEDS)
    ap.add_argument("--hps", default=str(RESULTS_DIR / "tuning" / "best_hyperparameters.json"))
    args = ap.parse_args()
    if args.stage == "merge":
        merge()
        return
    hps = json.load(open(args.hps))["best"]
    if args.stage == "tune":
        for method in args.methods:
            tune(args.regime, method, hps)
    elif args.stage == "eval":
        evaluate(args.regime, args.methods, args.seeds)
    else:
        closure_worlds(hps, args.seeds)


if __name__ == "__main__":
    main()
