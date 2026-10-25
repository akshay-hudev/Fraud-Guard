"""Synthetic regime experiments.

One "dataset" = one generator seed at one regime (ring_shift, ring_density,
protocol). For each dataset every method is trained on the training window,
early-stopped / thresholded on the validation window and scored on the test
window. Metrics and test-set predictions are written under results/synthetic/.
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import copy

import numpy as np
import pandas as pd
import torch

from ..ledger import RESULTS_DIR, frame_hash, provenance, write_json
from ..metrics import evaluate_scores
from ..models.train import fit_graph, fit_mlp, fit_tabular
from ..synth.features import A_COLUMNS, G_COLUMNS, build_tabular, c_columns, claim_level
from ..synth.generator import GeneratorConfig, generate
from ..synth.graph import build_graphs, build_metapath_graphs

TABULAR_METHODS = {          # name: (family, feature groups)
    "LR-CA": ("lr", "CA"),
    "RF-CA": ("rf", "CA"),
    "GB-C": ("gb", "C"),
    "GB-CA": ("gb", "CA"),
    "GB-CAG": ("gb", "CAG"),
}
GRAPH_METHODS = {"SAGE": "sage", "RGCN": "rgcn", "HAN": "han", "SimpleHGN": "simplehgn", "HGT": "hgt"}
# Hybrids: graph models whose claim nodes also receive the engineered A+G features.
HYBRID_METHODS = {"SAGE+AG": "sage", "HGT+AG": "hgt"}
ALL_METHODS = (list(TABULAR_METHODS) + ["MLP-C"] + list(GRAPH_METHODS) + list(HYBRID_METHODS)
               + ["Overlap-score"])


def prepare(world):
    tab = build_tabular(world)
    cols = {"C": c_columns(world), "A": A_COLUMNS, "G": G_COLUMNS}
    graphs = build_graphs(world, claim_level(world))
    return tab, cols, graphs


def feature_matrix(tab: dict, cols: dict, groups: str) -> dict:
    use = [c for g in groups for c in cols[g]]
    return {k: v[use].to_numpy(float) for k, v in tab.items()}


def augment_claims(graphs: dict, tab: dict, cols: dict) -> dict:
    """Copies of the period graphs with A+G features appended to claim nodes (train-standardised)."""
    X = feature_matrix(tab, cols, "AG")
    mu, sd = X["train"].mean(0), X["train"].std(0)
    sd = np.where(sd > 0, sd, 1.0)
    out = {}
    for split, glist in graphs.items():
        extra = torch.tensor((X[split] - mu) / sd, dtype=torch.float32)
        start, out[split] = 0, []
        for g in glist:
            n = g["claim"].num_nodes
            g2 = copy.copy(g)
            g2["claim"].x = torch.cat([g["claim"].x, extra[start:start + n]], dim=1)
            start += n
            out[split].append(g2)
        assert start == len(extra)
    return out


def run_dataset(cfg: GeneratorConfig, methods: list[str], hps: dict, model_seed: int = 0,
                out_dir: Path | None = None, verbose: bool = True, group: str = "all") -> list[dict]:
    world = generate(cfg)
    tab, cols, graphs = prepare(world)
    meta = hybrid = None
    y = {k: world.y[world.split == k] for k in ["train", "val", "test"]}
    ft_test = world.truth["fraud_type"].to_numpy()[world.split == "test"]
    claim_ids_test = world.claims["claim_id"].to_numpy()[world.split == "test"]
    for k in y:   # alignment check: features, graphs and labels share claim order
        assert (tab[k]["claim_id"].to_numpy() == world.claims["claim_id"].to_numpy()[world.split == k]).all()
        assert (torch.cat([g["claim"].y for g in graphs[k]]).numpy() == y[k]).all()
    rows, preds = [], {"claim_id": claim_ids_test, "y": y["test"], "fraud_type": ft_test}
    for name in methods:
        t0 = time.time()
        if name in TABULAR_METHODS:
            fam, groups = TABULAR_METHODS[name]
            res = fit_tabular(fam, feature_matrix(tab, cols, groups), y, hps[name], model_seed)
        elif name == "MLP-C":
            res = fit_mlp(feature_matrix(tab, cols, "C"), y, hps[name], model_seed)
        elif name in GRAPH_METHODS:
            g = graphs
            if GRAPH_METHODS[name] == "han":
                meta = meta or build_metapath_graphs(world, graphs)
                g = meta
            res = fit_graph(GRAPH_METHODS[name], g, hps[name], model_seed)
        elif name in HYBRID_METHODS:
            hybrid = hybrid or augment_claims(graphs, tab, cols)
            res = fit_graph(HYBRID_METHODS[name], hybrid, hps[name], model_seed)
        elif name == "Overlap-score":   # unsupervised: shared-patient closure around the claim
            res = {"val_scores": tab["val"]["claim_closure_norm"].to_numpy(float),
                   "test_scores": tab["test"]["claim_closure_norm"].to_numpy(float),
                   "train_seconds": 0.0, "infer_seconds": 0.0, "best_val_ap": float("nan")}
        else:
            raise ValueError(name)
        m = evaluate_scores(y["val"], res["val_scores"], y["test"], res["test_scores"], ft_test)
        row = {"method": name, **cfg.as_dict(), "protocol": cfg.protocol, "model_seed": model_seed, **m,
               "best_val_ap": res.get("best_val_ap"), "train_seconds": res["train_seconds"],
               "infer_seconds": res["infer_seconds"], "best_epoch": res.get("best_epoch")}
        rows.append(row)
        preds[name] = res["test_scores"].astype(np.float32)
        if verbose:
            print(f"  {name:13s} AP={m['ap']:.3f} ringAP={m.get('ring_ap', float('nan')):.3f} "
                  f"isoAP={m.get('isolated_ap', float('nan')):.3f} ({time.time() - t0:.1f}s)", flush=True)
    if out_dir is not None:
        tag = f"{run_tag(cfg, model_seed)}__{group}"
        (out_dir / "predictions").mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_dir / "predictions" / f"{tag}.npz", **preds)
        write_json(out_dir / "runs" / f"{tag}.json", {
            "provenance": provenance(cfg.as_dict(), model_seed=model_seed, hyperparameters={m: hps.get(m) for m in methods},
                                     claims_hash=frame_hash(world.claims), truth_hash=frame_hash(world.truth)),
            "rows": rows})
    return rows


def run_tag(cfg: GeneratorConfig, model_seed: int) -> str:
    # Ring volume appears in the tag only when it differs from the default, so existing tags are unchanged.
    volume = "" if cfg.ring_claims_per_month == GeneratorConfig.ring_claims_per_month else f"_v{cfg.ring_claims_per_month:g}"
    return f"{cfg.protocol}_d{cfg.ring_shift}_s{cfg.ring_density}{volume}_g{cfg.seed}_m{model_seed}"


def load_hps(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)["best"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shifts", type=float, nargs="+", default=[0.0, 0.5, 1.0, 2.0])
    ap.add_argument("--densities", type=float, nargs="+", default=[0.1, 0.5, 0.9])
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(1, 11)))
    ap.add_argument("--persistent", type=int, default=0, help="number of persistent rings (0 = emergent protocol)")
    ap.add_argument("--ring-claims-per-month", type=float, default=GeneratorConfig.ring_claims_per_month,
                    help="ring claim volume (default 16; the low-volume stress test uses 4)")
    ap.add_argument("--methods", nargs="+", default=ALL_METHODS)
    ap.add_argument("--hps", default=str(RESULTS_DIR / "tuning" / "best_hyperparameters.json"))
    ap.add_argument("--out", default=str(RESULTS_DIR / "synthetic"))
    ap.add_argument("--group", default="all", help="name of this method group (one run file per dataset and group)")
    args = ap.parse_args()
    hps, out = load_hps(Path(args.hps)), Path(args.out)
    for shift, dens, seed in itertools.product(args.shifts, args.densities, args.seeds):
        cfg = GeneratorConfig(ring_shift=shift, ring_density=dens, seed=seed, persistent_rings=args.persistent,
                              ring_claims_per_month=args.ring_claims_per_month)
        if (out / "runs" / f"{run_tag(cfg, 0)}__{args.group}.json").exists():
            continue
        print(f"[{run_tag(cfg, 0)} {args.group}]", flush=True)
        run_dataset(cfg, args.methods, hps, out_dir=out, group=args.group)


if __name__ == "__main__":
    main()
