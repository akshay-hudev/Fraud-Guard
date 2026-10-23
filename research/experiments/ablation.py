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

HGT single-factor family (``--family arch``, results/ablation_arch/; the tuned
configuration has 3 layers and 4 heads; every variant changes one factor):
* layers_4, heads_1, heads_2, heads_8
* no_files, no_billed_by     : relation (and its reverse) removed
* no_hospital                : hospital nodes and the affiliated relation removed
* no_claim_attributes        : claim-node inputs set to zero
* no_entity_attributes       : static patient/doctor/hospital attributes set to zero (degree encodings kept)
* degree_only                : both of the above, so only degree encodings and the graph remain
The family's 'full' reference is the HGT 'full' run of the main ablation (same worlds, seeds
and configuration).
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
OUT_ARCH = RESULTS_DIR / "ablation_arch"


def rewire(world, seed: int, periods=None):
    """Permute doctor endpoints of claims within each period (degree-preserving); all periods by default."""
    rng = np.random.default_rng(seed)
    w = copy.copy(world)
    claims = world.claims.copy()
    for p in (np.unique(world.period) if periods is None else periods):
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


def drop_node_type(graphs, node_type: str):
    """Remove a node type and every relation that touches it."""
    out = {}
    for split, glist in graphs.items():
        out[split] = []
        for g in glist:
            g2 = copy.copy(g)
            for et in list(g2.edge_types):
                if node_type in (et[0], et[2]):
                    del g2[et]
            del g2[node_type]
            out[split].append(g2)
    return out


def zero_inputs(graphs, n_static: dict, claim: bool = False, static: bool = False):
    """Set claim-node inputs and/or the static entity attributes (first n_static columns) to zero."""
    out = {}
    for split, glist in graphs.items():
        out[split] = []
        for g in glist:
            g2 = copy.copy(g)
            if claim:
                g2["claim"].x = torch.zeros_like(g["claim"].x)
            if static:
                for t, k in n_static.items():
                    x = g[t].x.clone()
                    x[:, :k] = 0.0
                    g2[t].x = x
            out[split].append(g2)
    return out


def n_static_columns(world) -> dict:
    return {"patient": 3, "doctor": 1 + world.config.n_specialties, "hospital": 1}


def arch_variants(graphs, hp: dict, n_static: dict) -> dict:
    """{variant: (graphs, hyperparameters)} for the HGT single-factor family."""
    return {
        "layers_4": (graphs, {**hp, "layers": 4}),
        "heads_1": (graphs, {**hp, "heads": 1}),
        "heads_2": (graphs, {**hp, "heads": 2}),
        "heads_8": (graphs, {**hp, "heads": 8}),
        "no_files": (drop_relation(graphs, "files"), hp),
        "no_billed_by": (drop_relation(graphs, "billed_by"), hp),
        "no_hospital": (drop_node_type(graphs, "hospital"), hp),
        "no_claim_attributes": (zero_inputs(graphs, n_static, claim=True), hp),
        "no_entity_attributes": (zero_inputs(graphs, n_static, static=True), hp),
        "degree_only": (zero_inputs(graphs, n_static, claim=True, static=True), hp),
    }


def main_arch(args, hps: dict) -> None:
    for dens, seed in itertools.product(args.densities, args.seeds):
        path = OUT_ARCH / "runs" / f"d0.0_s{dens}_g{seed}.json"
        if path.exists():
            continue
        cfg = GeneratorConfig(ring_shift=0.0, ring_density=dens, seed=seed)
        world = generate(cfg)
        _, _, graphs = prepare(world)
        y = {k: world.y[world.split == k] for k in ["train", "val", "test"]}
        ft = world.truth["fraud_type"].to_numpy()[world.split == "test"]
        rows = []
        for variant, (g, hp) in arch_variants(graphs, hps["HGT"], n_static_columns(world)).items():
            res = fit_graph("hgt", g, hp, 0)
            m = evaluate_scores(y["val"], res["val_scores"], y["test"], res["test_scores"], ft)
            rows.append({"method": "HGT", "variant": variant, "ring_density": dens, "seed": seed, **m,
                         "train_seconds": res["train_seconds"], "best_epoch": res["best_epoch"]})
            print(f"s={dens} g={seed} HGT|{variant:20s} ringAP={m['ring_ap']:.3f} isoAP={m['isolated_ap']:.3f}", flush=True)
        write_json(path, {"provenance": provenance(cfg.as_dict(), family="arch", hyperparameters=hps["HGT"]),
                          "rows": rows})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=["mechanism", "arch"], default="mechanism")
    ap.add_argument("--densities", type=float, nargs="+", default=[0.9])
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(1, 7)))
    ap.add_argument("--hps", default=str(RESULTS_DIR / "tuning" / "best_hyperparameters.json"))
    args = ap.parse_args()
    hps = load_hps(args.hps)
    if args.family == "arch":
        main_arch(args, hps)
        return
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
            n_static = n_static_columns(w)
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
