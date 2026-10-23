"""Cost micro-benchmark for GB-CAG, HGT and GB-CAG -> HGT routing (one process, one thread).

Measured on this machine, per world:
* batch costs of one audit period: building the tabular features (C, A, G) and the
  period graphs, GB-CAG scoring of the test period, HGT scoring of the full test graph;
* per-claim costs: GB-CAG on one precomputed feature row, and HGT on the claim's exact
  L-hop computation subgraph (extraction + forward pass), checked against full-graph scores;
* routing in batch mode: HGT on the union of the escalated claims' computation subgraphs.

Not measured (no such infrastructure exists in this repository): incremental maintenance of
aggregates, graph features and degree counts as claims arrive, data retrieval from a store,
network overhead and concurrency. Online latency figures therefore exclude feature upkeep.

Run alone (no parallel jobs):
    OMP_NUM_THREADS=1 RESEARCH_THREADS=1 python -m research.experiments.latency
"""

from __future__ import annotations

import argparse
import resource
import time

import numpy as np
import torch
from torch_geometric.utils import k_hop_subgraph

from ..ledger import RESULTS_DIR, provenance, write_json
from ..models.gnn import homogeneous_structure
from ..models.train import fit_graph, fit_tabular
from ..synth.features import build_tabular, claim_level
from ..synth.generator import GeneratorConfig, generate
from ..synth.graph import build_graphs
from .cascade import ESCALATION
from .synthetic import feature_matrix, load_hps, prepare

OUT = RESULTS_DIR / "latency"


def computation_subgraph(g, claim_ids, hops: int):
    """Induced subgraph of every node within ``hops`` of the given claims, and the claims' new positions.

    With every relation paired with its reverse, this contains everything an L-layer model needs
    to compute the claims' outputs exactly.
    """
    offsets, total, edge_index, _ = homogeneous_structure(g)
    centers = torch.as_tensor(np.atleast_1d(claim_ids), dtype=torch.long) + offsets["claim"]
    subset, _, _, _ = k_hop_subgraph(centers, hops, edge_index, num_nodes=total)
    parts = {}
    for t in g.node_types:
        lo, hi = offsets[t], offsets[t] + g[t].num_nodes
        parts[t] = subset[(subset >= lo) & (subset < hi)] - lo
    sub = g.subgraph(parts)
    pos = torch.searchsorted(parts["claim"], centers - offsets["claim"])
    return sub, pos


def _median_seconds(fn, repeats: int) -> float:
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def _pcts(x) -> dict:
    x = np.asarray(x) * 1000
    return {"p50_ms": float(np.percentile(x, 50)), "p99_ms": float(np.percentile(x, 99)), "mean_ms": float(x.mean())}


def benchmark_world(cfg: GeneratorConfig, hps: dict, n_claims: int, repeats: int) -> dict:
    world = generate(cfg)
    n_periods = cfg.n_periods
    feat_s = _median_seconds(lambda: build_tabular(world), 3) / n_periods
    graph_s = _median_seconds(lambda: build_graphs(world, claim_level(world)), 3) / n_periods
    tab, cols, graphs = prepare(world)
    y = {k: world.y[world.split == k] for k in ["train", "val", "test"]}
    X = feature_matrix(tab, cols, "CAG")
    gb = fit_tabular("gb", X, y, hps["GB-CAG"], 0)
    hgt = fit_graph("hgt", graphs, hps["HGT"], 0)
    model, g_test = hgt["model"], graphs["test"][0]
    hops = hps["HGT"]["layers"]
    with torch.no_grad():
        full = torch.sigmoid(model(g_test)).numpy()
    out = {"config": cfg.as_dict(), "n_test_claims": int(len(y["test"])),
           "test_graph_nodes": int(sum(g_test[t].num_nodes for t in g_test.node_types)),
           "test_graph_edges": int(sum(g_test[et].num_edges for et in g_test.edge_types)),
           "hgt_parameters": int(sum(p.numel() for p in model.parameters())),
           "batch": {"features_per_period_s": feat_s, "graph_per_period_s": graph_s,
                     "gb_score_period_s": _median_seconds(lambda: gb["model"].predict_proba(X["test"]), repeats),
                     "hgt_score_period_s": _median_seconds(lambda: model(g_test), repeats)}}
    rng = np.random.default_rng(cfg.seed)
    claims = rng.choice(len(y["test"]), size=min(n_claims, len(y["test"])), replace=False)
    gb_t, ext_t, fwd_t, sizes, err = [], [], [], [], []
    with torch.no_grad():
        for c in claims:
            row = X["test"][c:c + 1]
            t0 = time.perf_counter()
            gb["model"].predict_proba(row)
            gb_t.append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            sub, pos = computation_subgraph(g_test, c, hops)
            t1 = time.perf_counter()
            p = torch.sigmoid(model(sub)[pos]).item()
            t2 = time.perf_counter()
            ext_t.append(t1 - t0)
            fwd_t.append(t2 - t1)
            sizes.append(sum(sub[t].num_nodes for t in sub.node_types))
            err.append(abs(p - full[c]))
    total = np.asarray(ext_t) + np.asarray(fwd_t)
    out["per_claim"] = {"n_claims": int(len(claims)), "gb": _pcts(gb_t), "hgt_extract": _pcts(ext_t),
                        "hgt_forward": _pcts(fwd_t), "hgt_total": _pcts(total),
                        "gb_throughput_per_s": float(1 / np.mean(gb_t)), "hgt_throughput_per_s": float(1 / total.mean()),
                        "subgraph_nodes_median": float(np.median(sizes)),
                        "subgraph_share_of_graph_median": float(np.median(sizes) / out["test_graph_nodes"]),
                        "max_abs_diff_vs_full_graph": float(np.max(err))}
    g_val_scores, g_test_scores = gb["val_scores"], gb["test_scores"]
    routing = []
    with torch.no_grad():
        for e in [x for x in ESCALATION if x < 1]:
            esc = np.flatnonzero(g_test_scores >= np.quantile(g_val_scores, 1 - e))
            t0 = time.perf_counter()
            sub, pos = computation_subgraph(g_test, esc, hops)
            model(sub)
            routing.append({"escalation": e, "escalated_test": float(len(esc) / len(g_test_scores)),
                            "union_subgraph_share": float(sum(sub[t].num_nodes for t in sub.node_types)
                                                          / out["test_graph_nodes"]),
                            "seconds": time.perf_counter() - t0})
    out["batch_routing"] = routing
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--densities", type=float, nargs="+", default=[0.1, 0.9])
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--n-claims", type=int, default=300)
    ap.add_argument("--repeats", type=int, default=20)
    args = ap.parse_args()
    torch.set_num_threads(1)
    hps = load_hps(RESULTS_DIR / "tuning" / "best_hyperparameters.json")
    worlds = []
    for dens in args.densities:
        for seed in args.seeds:
            cfg = GeneratorConfig(ring_shift=0.0, ring_density=dens, seed=seed)
            worlds.append(benchmark_world(cfg, hps, args.n_claims, args.repeats))
            w = worlds[-1]
            print(f"s={dens} g={seed}: GB {w['per_claim']['gb']['p50_ms']:.2f} ms, HGT {w['per_claim']['hgt_total']['p50_ms']:.1f}/"
                  f"{w['per_claim']['hgt_total']['p99_ms']:.1f} ms (p50/p99), subgraph "
                  f"{w['per_claim']['subgraph_share_of_graph_median']:.0%} of graph", flush=True)
    write_json(OUT / "latency.json", {
        "provenance": provenance({"benchmark": "latency", **vars(args)}, threads=torch.get_num_threads()),
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,   # bytes on macOS
        "worlds": worlds})


if __name__ == "__main__":
    main()
