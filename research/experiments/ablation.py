"""Mechanism ablations on the synthetic benchmark (delta = 0: ring claims look legitimate).

Variants
* full                 : tuned configuration
* rewired              : claim->doctor assignments permuted within each audit period
                         (preserves every doctor's and patient's claim volume, destroys
                         which patients co-occur with which doctors, i.e. the ring block)
* no_visits            : patient-doctor 'visits' relation removed (graph models only)
* no_degree            : degree encodings removed from node features (graph models only)
* layers_1/2/3         : message-passing depth (graph models only)

Pitfall demonstrations (reproducing defects of the legacy pipeline on the corrected benchmark)
* GB-CA|ring_id_leak   : GB-CA plus the generator's ring identifier as a feature
* HGT|no_reverse       : HGT on a graph without reverse relations (claims then receive
                         messages only from patients; doctor context cannot reach them)

If a method's ring detection relies on coordination structure, 'rewired' should
collapse it towards the volume-only signal.
"""

from __future__ import annotations

import argparse
import copy
import itertools

import numpy as np
import torch

from ..ledger import RESULTS_DIR, provenance, write_json
from ..metrics import evaluate_scores
from ..models.train import fit_graph, fit_tabular
from ..synth.generator import GeneratorConfig, generate
from .synthetic import feature_matrix, load_hps, prepare

OUT = RESULTS_DIR / "ablation"


def rewire(world, seed: int):
    """Permute doctor endpoints of claims within each period (degree-preserving)."""
    rng = np.random.default_rng(seed)
    w = copy.copy(world)
    claims = world.claims.copy()
    for p in np.unique(world.period):
        rows = np.flatnonzero(world.period == p)
        perm = rng.permutation(rows)
        claims.loc[rows, ["doctor_id", "hospital_id"]] = world.claims.loc[perm, ["doctor_id", "hospital_id"]].to_numpy()
    w.claims = claims
    return w


def drop_relation(graphs, rel: str):
    out = {}
    for split, glist in graphs.items():
        out[split] = []
        for g in glist:
            g2 = copy.copy(g)
            for et in list(g2.edge_types):
                if et[1] in (rel, f"rev_{rel}"):
                    del g2[et]
            out[split].append(g2)
    return out


def drop_reverse(graphs):
    """Keep only forward relations; node types left without incoming edges are given a self-loop
    relation so the convolution can run, which reproduces the legacy one-directional graph."""
    out = {}
    for split, glist in graphs.items():
        out[split] = []
        for g in glist:
            g2 = copy.copy(g)
            for et in list(g2.edge_types):
                if et[1].startswith("rev_"):
                    del g2[et]
            for t in g2.node_types:
                if t not in {et[2] for et in g2.edge_types}:
                    n = g2[t].num_nodes
                    g2[t, "self", t].edge_index = torch.arange(n).repeat(2, 1)
            out[split].append(g2)
    return out


def drop_degree(graphs, n_static: dict):
    out = {}
    for split, glist in graphs.items():
        out[split] = []
        for g in glist:
            g2 = copy.copy(g)
            for t, k in n_static.items():
                g2[t].x = g[t].x[:, :k]
            out[split].append(g2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--densities", type=float, nargs="+", default=[0.9])
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(1, 7)))
    ap.add_argument("--hps", default=str(RESULTS_DIR / "tuning" / "best_hyperparameters.json"))
    args = ap.parse_args()
    hps = load_hps(args.hps)
    for dens, seed in itertools.product(args.densities, args.seeds):
        path = OUT / "runs" / f"d0.0_s{dens}_g{seed}.json"
        if path.exists():
            continue
        cfg = GeneratorConfig(ring_shift=0.0, ring_density=dens, seed=seed)
        world = generate(cfg)
        rows = []
        for variant, w in [("full", world), ("rewired", rewire(world, seed))]:
            tab, cols, graphs = prepare(w)
            y = {k: w.y[w.split == k] for k in ["train", "val", "test"]}
            ft = w.truth["fraud_type"].to_numpy()[w.split == "test"]
            n_static = {"patient": 3, "doctor": 1 + w.config.n_specialties, "hospital": 1}
            runs = [("GB-CAG", lambda: fit_tabular("gb", feature_matrix(tab, cols, "CAG"), y, hps["GB-CAG"], 0)),
                    ("GB-CA", lambda: fit_tabular("gb", feature_matrix(tab, cols, "CA"), y, hps["GB-CA"], 0))]
            if variant == "full":
                ring_id = {k: w.truth["ring_id"].to_numpy()[w.split == k][:, None] for k in y}
                leaky = {k: np.hstack([v, ring_id[k]]) for k, v in feature_matrix(tab, cols, "CA").items()}
                runs.append(("GB-CA|ring_id_leak", lambda: fit_tabular("gb", leaky, y, hps["GB-CA"], 0)))
            for method, kind in [("SAGE", "sage"), ("HGT", "hgt")]:
                runs.append((method, lambda kind=kind, method=method: fit_graph(kind, graphs, hps[method], 0)))
                if variant == "full" and method == "HGT":
                    runs.append(("HGT|no_reverse", lambda: fit_graph("hgt", drop_reverse(graphs), hps["HGT"], 0)))
                if variant == "full":
                    runs.append((f"{method}|no_visits", lambda kind=kind, method=method:
                                 fit_graph(kind, drop_relation(graphs, "visits"), hps[method], 0)))
                    runs.append((f"{method}|no_degree", lambda kind=kind, method=method:
                                 fit_graph(kind, drop_degree(graphs, n_static), hps[method], 0)))
                    for L in [1, 2, 3]:
                        if L != hps[method]["layers"]:
                            runs.append((f"{method}|layers_{L}", lambda kind=kind, method=method, L=L:
                                         fit_graph(kind, graphs, {**hps[method], "layers": L}, 0)))
            for name, fn in runs:
                res = fn()
                m = evaluate_scores(y["val"], res["val_scores"], y["test"], res["test_scores"], ft)
                base, _, mod = name.partition("|")
                rows.append({"method": base, "variant": variant if not mod else mod, "ring_density": dens,
                             "seed": seed, **m, "train_seconds": res["train_seconds"]})
                print(f"s={dens} g={seed} {variant:8s} {name:16s} ringAP={m['ring_ap']:.3f}", flush=True)
        write_json(path, {"provenance": provenance(cfg.as_dict(), hyperparameters=hps), "rows": rows})


if __name__ == "__main__":
    main()
