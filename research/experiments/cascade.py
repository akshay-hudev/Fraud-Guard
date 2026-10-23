"""Two-stage routing on the synthetic benchmark, and what the trained models rely on.

For every world, GB-CAG and the GNNs are retrained exactly as in synthetic.py
(same hyperparameters and seeds). Their test scores are compared with the saved
predictions of the main runs (a reproducibility check of the committed code),
and their validation scores, which the main runs did not keep, are saved.

Routing. Stage 1 scores every claim with GB-CAG. Claims whose GB-CAG score is at
least the (1 - e) quantile of the GB-CAG *validation* scores are escalated to a
GNN; e is chosen from ESCALATION by validation AP, so test claims never affect
the choice. Escalated claims are ordered by
  * 'rerank': the GNN score alone (GB-CAG acts as a filter), or
  * 'fuse'  : the mean of the two scores' validation percentiles,
and rank above every non-escalated claim, which keeps its GB-CAG score.
e = 1 escalates everything: 'rerank' is then the GNN alone and 'fuse' a
rank-average ensemble. A claim's routed score depends only on its own two
scores and on validation statistics.

Reliance (test period only; models are not retrained):
  * GB-CAG: grouped permutation importance of C, A, G and of each G feature
  * GNNs: permutation of claim attributes, static entity attributes and degree encodings
  * all : degree-preserving rewiring of the test period's claim-doctor links,
          with features and graphs rebuilt from the rewired claims
"""

from __future__ import annotations

import argparse
import copy
import itertools
import time
from pathlib import Path

import numpy as np
import torch

from ..ledger import RESULTS_DIR, frame_hash, provenance, write_json
from ..metrics import _ap, precision_at_budget, recall_at_budget
from ..models.train import _scores, fit_graph, fit_tabular
from ..synth.features import A_COLUMNS, G_COLUMNS
from ..synth.generator import GeneratorConfig, generate
from .ablation import n_static_columns, rewire
from .synthetic import feature_matrix, load_hps, prepare, run_tag

OUT = RESULTS_DIR / "cascade"
ESCALATION = [0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
MODES = ["rerank", "fuse"]
GNNS = {"HGT": "hgt", "SAGE": "sage"}
BUDGET = 0.05
N_PERMUTATIONS = 5
N_REWIRINGS = 3
# Saved test predictions of the main runs: method -> run group.
SAVED_GROUP = {"GB-CAG": "tab", "SAGE": "tab", "HGT": "hgt", "SAGE+AG": "hybrid", "HGT+AG": "hybrid"}


def percentile(ref: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Share of the reference (validation) scores that are <= x."""
    return np.searchsorted(np.sort(ref), x, side="right") / len(ref)


def route(g: np.ndarray, h: np.ndarray, g_val: np.ndarray, h_val: np.ndarray, e: float, mode: str):
    """Routed scores for claims with GB-CAG score g and GNN score h; returns (scores, escalated)."""
    tau = -np.inf if e >= 1 else np.quantile(g_val, 1 - e)
    escalated = g >= tau
    second = h if mode == "rerank" else (percentile(g_val, g) + percentile(h_val, h)) / 2
    return np.where(escalated, 1.0 + second, g), escalated


def select_escalation(y_val: np.ndarray, g_val: np.ndarray, h_val: np.ndarray, mode: str) -> tuple[float, float]:
    """Escalation fraction with the highest validation AP (ties: the smaller fraction)."""
    best_e, best_ap = None, -np.inf
    for e in ESCALATION:
        ap = _ap(y_val, route(g_val, h_val, g_val, h_val, e, mode)[0])
        if ap > best_ap + 1e-12:
            best_e, best_ap = e, ap
    return best_e, best_ap


def score_metrics(y: np.ndarray, s: np.ndarray, fraud_type: np.ndarray, ring_id: np.ndarray,
                  budget: float = BUDGET) -> dict:
    """Threshold-free metrics on one test period. Unit: claim, except the ring-level entries."""
    pct = int(budget * 100)
    k = max(1, int(round(budget * len(s))))
    top = np.zeros(len(s), bool)
    top[np.argsort(-s, kind="stable")[:k]] = True
    out = {"ap": _ap(y, s), f"precision_at_{pct}pct": precision_at_budget(y, s, budget),
           f"recall_at_{pct}pct": recall_at_budget(y, s, budget)}
    for kind in ["ring", "isolated"]:
        keep = (fraud_type == "legit") | (fraud_type == kind)
        out[f"{kind}_ap"] = _ap((fraud_type[keep] == kind).astype(int), s[keep])
        out[f"{kind}_recall_at_{pct}pct"] = float(top[fraud_type == kind].mean())
    rings = np.unique(ring_id[ring_id > 0])
    share = np.array([top[ring_id == r].mean() for r in rings])
    out["ring_share_in_budget"] = share.round(4).tolist()
    out["rings_with_any_claim_in_budget"] = float((share > 0).mean())
    out["rings_with_half_claims_in_budget"] = float((share >= 0.5).mean())
    return out


def saved_test_scores(cfg: GeneratorConfig, method: str) -> np.ndarray | None:
    path = RESULTS_DIR / "synthetic" / "predictions" / f"{run_tag(cfg, 0)}__{SAVED_GROUP[method]}.npz"
    if not path.exists():
        return None
    z = np.load(path)
    return z[method] if method in z else None


def _permute_rows(x: np.ndarray, cols, rng) -> np.ndarray:
    out = x.copy()
    out[:, cols] = x[rng.permutation(len(x))][:, cols]
    return out


def _ap_triplet(y, s, ft) -> dict:
    out = {"ap": _ap(y, s)}
    for kind in ["ring", "isolated"]:
        keep = (ft == "legit") | (ft == kind)
        out[f"{kind}_ap"] = _ap((ft[keep] == kind).astype(int), s[keep])
    return out


def _drop(base: dict, perturbed: list[dict]) -> dict:
    return {k: float(base[k] - np.mean([p[k] for p in perturbed])) for k in base}


def gb_reliance(model, X_test: np.ndarray, columns: list[str], y, ft, rng) -> dict:
    base = _ap_triplet(y, model.predict_proba(X_test)[:, 1], ft)
    groups = {"C": [i for i, c in enumerate(columns) if c not in A_COLUMNS and c not in G_COLUMNS],
              "A": [columns.index(c) for c in A_COLUMNS], "G": [columns.index(c) for c in G_COLUMNS]}
    groups.update({f"G:{c}": [columns.index(c)] for c in G_COLUMNS})
    return {name: _drop(base, [_ap_triplet(y, model.predict_proba(_permute_rows(X_test, cols, rng))[:, 1], ft)
                               for _ in range(N_PERMUTATIONS)])
            for name, cols in groups.items()}


def gnn_reliance(model, g_test, n_static: dict, y, ft, rng) -> dict:
    def score(g):
        with torch.no_grad():
            return _scores(model, [g], "claim")

    base = _ap_triplet(y, score(g_test), ft)
    out = {}
    for name in ["claim_attributes", "entity_attributes", "degree_encodings"]:
        runs = []
        for _ in range(N_PERMUTATIONS):
            g2 = copy.copy(g_test)
            if name == "claim_attributes":
                x = g_test["claim"].x
                g2["claim"].x = x[torch.from_numpy(rng.permutation(len(x)))]
            else:
                for t, k in n_static.items():
                    x = g_test[t].x.clone()
                    cols = slice(0, k) if name == "entity_attributes" else slice(k, x.size(1))
                    x[:, cols] = g_test[t].x[torch.from_numpy(rng.permutation(len(x))), cols]
                    g2[t].x = x
            runs.append(_ap_triplet(y, score(g2), ft))
        out[name] = _drop(base, runs)
    return out


def rewiring_reliance(world, models: dict, base_scores: dict, y, ft) -> dict:
    """AP lost when the test period's claim-doctor links are permuted (degree-preserving), per model."""
    test_period = int(world.period[world.split == "test"][0])
    base = {m: _ap_triplet(y, s, ft) for m, s in base_scores.items()}
    runs = {m: [] for m in models}
    for r in range(N_REWIRINGS):
        w = rewire(world, seed=10_000 + r, periods=[test_period])
        tab_r, cols_r, graphs_r = prepare(w)
        for m, (kind, model) in models.items():
            if kind == "tabular":
                s = model.predict_proba(feature_matrix(tab_r, cols_r, "CAG")["test"])[:, 1]
            else:
                with torch.no_grad():
                    s = _scores(model, graphs_r["test"], "claim")
            runs[m].append(_ap_triplet(y, s, ft))
    return {m: _drop(base[m], runs[m]) for m in models}


def run_world(cfg: GeneratorConfig, hps: dict, gnns: list[str]) -> None:
    tag = run_tag(cfg, 0)
    world = generate(cfg)
    tab, cols, graphs = prepare(world)
    split = world.split
    y = {k: world.y[split == k] for k in ["train", "val", "test"]}
    ft = {k: world.truth["fraud_type"].to_numpy()[split == k] for k in ["val", "test"]}
    ring_id = world.truth["ring_id"].to_numpy()[split == "test"]
    X = feature_matrix(tab, cols, "CAG")
    columns = [c for g in "CAG" for c in cols[g]]
    rng = np.random.default_rng(cfg.seed)

    t0 = time.time()
    fitted = {"GB-CAG": fit_tabular("gb", X, y, hps["GB-CAG"], 0)}
    for name in gnns:
        fitted[name] = fit_graph(GNNS[name], graphs, hps[name], 0)
    print(f"[{tag}] trained in {time.time() - t0:.0f}s", flush=True)

    reproduction = {}
    for name, res in fitted.items():
        saved = saved_test_scores(cfg, name)
        if saved is not None:
            reproduction[name] = {"max_abs_diff": float(np.max(np.abs(saved - res["test_scores"]))),
                                  "ap_saved": _ap(y["test"], saved), "ap_now": _ap(y["test"], res["test_scores"])}

    rows, curves = [], []
    regime = {"ring_shift": cfg.ring_shift, "ring_density": cfg.ring_density, "seed": cfg.seed}
    singles = {m: r["test_scores"] for m, r in fitted.items()}
    for hybrid in ["SAGE+AG", "HGT+AG"]:          # existing hybrids, from the main runs' saved predictions
        s = saved_test_scores(cfg, hybrid)
        if s is not None:
            singles[hybrid] = s
    for name, s in singles.items():
        rows.append({"method": name, **regime, "escalation": None, "escalated_test": None,
                     **score_metrics(y["test"], s, ft["test"], ring_id)})
    g_val, g_test = fitted["GB-CAG"]["val_scores"], fitted["GB-CAG"]["test_scores"]
    for name, mode in itertools.product(gnns, MODES):
        h_val, h_test = fitted[name]["val_scores"], fitted[name]["test_scores"]
        e, val_ap = select_escalation(y["val"], g_val, h_val, mode)
        for e_c in ESCALATION:
            s_c, esc_c = route(g_test, h_test, g_val, h_val, e_c, mode)
            curves.append({"method": f"GB-CAG>{name}", "mode": mode, **regime, "escalation": e_c,
                           "val_ap": _ap(y["val"], route(g_val, h_val, g_val, h_val, e_c, mode)[0]),
                           "escalated_test": float(esc_c.mean()),
                           **{k: v for k, v in score_metrics(y["test"], s_c, ft["test"], ring_id).items()
                              if k != "ring_share_in_budget"}})
        s, esc = route(g_test, h_test, g_val, h_val, e, mode)
        rows.append({"method": f"GB-CAG>{name} ({mode})", **regime, "escalation": e, "val_ap": val_ap,
                     "escalated_test": float(esc.mean()), **score_metrics(y["test"], s, ft["test"], ring_id)})

    reliance = {"GB-CAG": gb_reliance(fitted["GB-CAG"]["model"], X["test"], columns, y["test"], ft["test"], rng)}
    n_static = n_static_columns(world)
    for name in gnns:
        reliance[name] = gnn_reliance(fitted[name]["model"], graphs["test"][0], n_static, y["test"], ft["test"], rng)
    models = {"GB-CAG": ("tabular", fitted["GB-CAG"]["model"])}
    models.update({name: ("graph", fitted[name]["model"]) for name in gnns})
    reliance["rewired_test_period"] = rewiring_reliance(
        world, models, {m: fitted[m]["test_scores"] for m in models}, y["test"], ft["test"])

    preds = {"claim_id_val": world.claims["claim_id"].to_numpy()[split == "val"],
             "claim_id_test": world.claims["claim_id"].to_numpy()[split == "test"],
             "y_val": y["val"], "y_test": y["test"], "fraud_type_val": ft["val"], "fraud_type_test": ft["test"],
             "ring_id_test": ring_id}
    for name, res in fitted.items():
        preds[f"{name}__val"] = res["val_scores"].astype(np.float32)
        preds[f"{name}__test"] = res["test_scores"].astype(np.float32)
    (OUT / "predictions").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "predictions" / f"{tag}.npz", **preds)
    write_json(OUT / "runs" / f"{tag}.json", {
        "provenance": provenance(cfg.as_dict(), model_seed=0, escalation_grid=ESCALATION, budget=BUDGET,
                                 hyperparameters={m: hps[m] for m in fitted},
                                 claims_hash=frame_hash(world.claims), truth_hash=frame_hash(world.truth)),
        "reproduction": reproduction, "rows": rows, "curves": curves, "reliance": reliance,
        "train_seconds": {m: r["train_seconds"] for m, r in fitted.items()}})
    best = {r["method"]: round(r["ap"], 3) for r in rows}
    print(f"[{tag}] AP {best} | reproduction {{{', '.join(f'{m}: {v['max_abs_diff']:.1e}' for m, v in reproduction.items())}}}",
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shifts", type=float, nargs="+", default=[0.0, 1.0, 2.0])
    ap.add_argument("--densities", type=float, nargs="+", default=[0.1, 0.5, 0.9])
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(1, 7)))
    ap.add_argument("--gnns", nargs="+", default=list(GNNS))
    ap.add_argument("--hps", default=str(RESULTS_DIR / "tuning" / "best_hyperparameters.json"))
    args = ap.parse_args()
    hps = load_hps(Path(args.hps))
    for shift, dens, seed in itertools.product(args.shifts, args.densities, args.seeds):
        cfg = GeneratorConfig(ring_shift=shift, ring_density=dens, seed=seed)
        if (OUT / "runs" / f"{run_tag(cfg, 0)}.json").exists():
            continue
        run_world(cfg, hps, args.gnns)


if __name__ == "__main__":
    main()
