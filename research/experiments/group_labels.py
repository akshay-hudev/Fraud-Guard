"""Sixth experimental round (paper/hiss_review/analysis_plan_round6.md): group-fraud labels on the public data.

Ma et al. (IJCAI 2023) published expert group-fraud flags for the Kaggle dataset used in kaggle_exp.py, in
https://github.com/finint/RDPGL (commit 6e0151d4). Stages:

* ``describe``: confirm the provider mapping and describe flagged providers (volume, overlap, assortativity).
* ``rescore``: the stored test predictions of every method, re-scored on the group flag. Nothing is retrained.
* ``retrain``: GB-P, GB-PG and three GNNs trained on the group flag, other fraud providers removed.
* ``table``: the comparisons and the Online Resource table.

Group AP scores flagged providers against non-fraud providers, the analogue of ring AP; individual AP scores the
other fraud providers against non-fraud providers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from ..ledger import KAGGLE_DIR, REPO_ROOT, RESULTS_DIR, provenance, write_json
from ..metrics import corrected_resampled_comparison, holm
from ..models.train import fit_graph, fit_tabular
from ..realdata.kaggle import build_graph, load, load_raw
from . import kaggle_exp
from .kaggle_exp import GRAPH, TABULAR, _masked, features, splits

RDPGL_DIR = REPO_ROOT / "data" / "rdpgl"
RDPGL_COMMIT = "6e0151d40ea5e9def878959d9863bb6f2a8cf8f4"
FIRST_ID = 1000                     # RDPGL numbers the sorted Kaggle providers from here
REPEATS, FOLDS = range(5), range(5)
STRUCTURE = [("GB-PG", "GB-P"), ("GB-PGL", "GB-P")]
GRAPH_MODELS = ["SAGE", "RGCN", "HAN", "SimpleHGN", "HGT", "BWGNN", "CARE-GNN", "SAGE+G", "HGT+G"]
METHODS = ["LR-P", "RF-P", "GB-P", "GB-PG", "GB-PGL", "MLP-P"] + GRAPH_MODELS
RETRAIN_TABULAR, RETRAIN_GNN = ["GB-P", "GB-PG"], ["SAGE", "RGCN", "HGT"]
TEST_TRAIN = 1 / 3                  # one test fold against three training folds
FLAG_FEATURES = ["bene_max_jaccard", "bene_shared_share", "bene_log_n_coproviders"]


def rank_ids(kaggle_ids) -> dict[str, int]:
    """RDPGL's integer identifier of each Kaggle provider: its rank among the sorted identifiers, from FIRST_ID."""
    return {p: FIRST_ID + i for i, p in enumerate(sorted(set(kaggle_ids)))}


def fingerprint(frame: pd.DataFrame, provider: str, inpatient: str) -> pd.DataFrame:
    """Per-provider claim count, inpatient count and amounts. RDPGL stores missing values as -1, Kaggle as NaN."""
    amounts = frame[["InscClaimAmtReimbursed", "DeductibleAmtPaid"]].fillna(0).clip(lower=0)
    return (pd.concat([frame[[provider, inpatient]], amounts], axis=1)
            .groupby(provider).agg(n=(inpatient, "size"), inpatient=(inpatient, "sum"),
                                   amount=("InscClaimAmtReimbursed", "sum"), deductible=("DeductibleAmtPaid", "sum")))


def group_flags(kaggle_dir: Path = KAGGLE_DIR, rdpgl_dir: Path = RDPGL_DIR) -> pd.DataFrame:
    """fraud and group flags per Kaggle provider; raises unless the mapping holds for every provider."""
    rd = pd.read_csv(rdpgl_dir / "claims.csv")
    lab, _, claims = load_raw(kaggle_dir)
    ids = rank_ids(lab["Provider"])
    mine = fingerprint(claims, "Provider", "is_inpatient").rename(index=ids).sort_index()
    theirs = fingerprint(rd, "Provider", "inpatent").sort_index()
    if not mine.index.equals(theirs.index) or not np.allclose(mine.to_numpy(float), theirs.to_numpy(float)):
        raise ValueError("RDPGL providers do not map onto the Kaggle providers by rank")
    flags = rd.groupby("Provider").agg(fraud=("PotentialFraud", "max"), fraud_min=("PotentialFraud", "min"),
                                       group=("PotentialGroupFraud", "max"), group_min=("PotentialGroupFraud", "min"))
    if not ((flags.fraud == flags.fraud_min).all() and (flags.group == flags.group_min).all()):
        raise ValueError("a flag varies within a provider")
    out = pd.DataFrame({"provider_id": sorted(ids)})
    rid = out["provider_id"].map(ids)
    out["fraud"], out["group"] = flags.fraud.reindex(rid).to_numpy(), flags.group.reindex(rid).to_numpy()
    known = lab.set_index("Provider")["PotentialFraud"].map({"Yes": 1, "No": 0}).reindex(out["provider_id"]).to_numpy()
    if not (known == out["fraud"].to_numpy()).all() or (out["group"] > out["fraud"]).any():
        raise ValueError("RDPGL labels disagree with the Kaggle labels")
    return out


def group_ap(y_fraud: np.ndarray, y_group: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """(group AP, individual AP) for one test fold."""
    other = (y_fraud == 1) & (y_group == 0)
    return (float(average_precision_score(y_group[~other], scores[~other])),
            float(average_precision_score(other[y_group == 0], scores[y_group == 0])))


def stored_rotation(results: Path, repeat: int, fold: int) -> dict:
    """Every method's stored test predictions for one rotation; each method must have one set of scores."""
    pattern = f"fold{fold}_*.npz" if repeat == 0 else f"r{repeat}_fold{fold}_*.npz"
    folder = results / ("kaggle" if repeat == 0 else "kaggle_repeated") / "predictions"
    out: dict = {}
    for f in sorted(folder.glob(pattern)):
        z = np.load(f, allow_pickle=True)
        for key in z.files:
            if key in out and not np.array_equal(out[key], z[key]):
                raise ValueError(f"{key} differs between prediction files of repeat {repeat} fold {fold}")
            out[key] = z[key]
    return out


def recorded_ap(results: Path) -> dict:
    rows = []
    for sub in ["kaggle", "kaggle_repeated"]:
        for f in sorted((results / sub / "runs").glob("*.json")):
            rows += json.load(open(f))["rows"]
    return {(int(r.get("repeat", 0)), int(r["fold"]), r["method"]): r["ap"] for r in rows}


def rescore(results: Path, flags: pd.DataFrame) -> None:
    """Group and individual AP of every method on all 25 rotations, from the stored predictions alone."""
    group = flags.set_index("provider_id")["group"]
    recorded, rows, checked = recorded_ap(results), [], 0
    for repeat in REPEATS:
        for fold in FOLDS:
            z = stored_rotation(results, repeat, fold)
            y, g = z["y"].astype(int), group.reindex(z["provider_id"]).to_numpy().astype(int)
            other = (y == 1) & (g == 0)
            for m in METHODS:
                # The stored scores must still give the AP recorded for this run: this guards the alignment.
                if abs(average_precision_score(y, z[m]) - recorded[(repeat, fold, m)]) > 1e-6:
                    raise ValueError(f"stored predictions of {m} do not reproduce the recorded AP")
                checked += 1
                ga, ia = group_ap(y, g, z[m])
                rows.append({"repeat": repeat, "fold": fold, "rotation": 5 * repeat + fold, "method": m, "group_ap": ga,
                             "individual_ap": ia, "n_group": int(g.sum()), "n_other": int(other.sum()),
                             "n_legit": int((y == 0).sum())})
    out = results / "group_labels"
    write_json(out / "rescored.json", {"provenance": provenance({"rdpgl_commit": RDPGL_COMMIT}, checked_against_records=checked),
                                       "rows": rows})
    print(f"re-scored {len(rows)} method rotations; {checked} reproduced their recorded AP")


def retrain_rotation(data, flags: pd.DataFrame, hps: dict, results: Path, repeat: int, fold: int, methods: list[str]) -> None:
    """Models trained on the group flag, with the other fraud providers removed from every split."""
    run = results / "group_labels" / "runs" / f"r{repeat}_fold{fold}_{'_'.join(methods)}.json"
    if run.exists():
        return
    y, g = flags["fraud"].to_numpy(), flags["group"].to_numpy()
    keep = lambda idx: idx[~((y[idx] == 1) & (g[idx] == 0))]
    train, val, test = (keep(i) for i in splits(repeat)[fold])
    rows = []
    for m in methods:
        if m in TABULAR:
            fam, groups = TABULAR[m]
            X = features(data, groups, train)
            res = fit_tabular(fam, {"train": X[train], "val": X[val], "test": X[test]},
                              {"train": g[train], "val": g[val], "test": g[test]}, hps[m], 0)
        else:
            kind, groups = GRAPH[m]
            graph = build_graph(data, features(data, groups, train), g)
            graphs = {k: _masked(graph, rows_, len(g)) for k, rows_ in [("train", train), ("val", val), ("test", test)]}
            res = fit_graph(kind, graphs, hps[m], 0, max_epochs=100, patience=10, target="provider")
        rows.append({"method": m, "repeat": repeat, "fold": fold, "rotation": 5 * repeat + fold,
                     "group_ap": float(average_precision_score(g[test], res["test_scores"])), "n_group": int(g[test].sum()),
                     "n_legit": int((g[test] == 0).sum()), "best_epoch": res.get("best_epoch")})
        print(f"r{repeat} fold {fold} {m} group AP {rows[-1]['group_ap']:.3f}", flush=True)
    write_json(run, {"provenance": provenance({"repeat": repeat, "fold": fold, "methods": methods, "rdpgl_commit": RDPGL_COMMIT},
                                              hyperparameters={m: hps[m] for m in methods}), "rows": rows})


def describe(data, flags: pd.DataFrame, results: Path) -> None:
    """Volume, overlap and assortativity of flagged providers; uses all labels, and no model sees it."""
    g = flags["group"].to_numpy()
    cls = np.where(g == 1, "group", np.where(flags["fraud"] == 1, "other fraud", "non-fraud"))
    claims = data.claims.groupby("Provider").size().reindex(data.providers["provider_id"]).to_numpy()
    out: dict = {"n": {c: int((cls == c).sum()) for c in ["group", "other fraud", "non-fraud"]},
                 "group_rate": float(g.mean()), "median_claims": {}, "features": {}}
    for c in out["n"]:
        out["median_claims"][c] = float(np.median(claims[cls == c]))
        out["features"][c] = {f: float(np.median(data.G[f].to_numpy()[cls == c])) for f in FLAG_FEATURES}
    for name, B in [("beneficiary", data.B_pb), ("physician", data.B_pf)]:
        C = (B @ B.T).tocsr()
        C.setdiag(0)
        C.eliminate_zeros()
        C.data[:] = 1.0
        deg = np.asarray(C.sum(1)).ravel()
        share = np.divide(C @ g.astype(float), deg, out=np.full(len(g), np.nan), where=deg > 0)
        out[name] = {"flag_share_among_coproviders_of_group": float(np.nanmean(share[g == 1])),
                     "flag_share_among_coproviders_of_others": float(np.nanmean(share[g == 0])),
                     "flag_share_among_coproviders_of_other_fraud": float(np.nanmean(share[cls == "other fraud"]))}
    write_json(results / "group_labels" / "description.json",
               {"provenance": provenance({"rdpgl_commit": RDPGL_COMMIT}), "description": out})
    print(json.dumps(out, indent=1))


def compare(results: Path) -> dict:
    """The families of the plan, written with the means to comparisons.csv."""
    df = pd.DataFrame(json.load(open(results / "group_labels" / "rescored.json"))["rows"])
    ga = df.pivot_table(index="rotation", columns="method", values="group_ap")
    ia = df.pivot_table(index="rotation", columns="method", values="individual_ap")
    rows = [{"family": "G1", "method": a, "reference": b, "metric": "group_ap",
             **corrected_resampled_comparison(ga[a].to_numpy(), ga[b].to_numpy(), TEST_TRAIN)} for a, b in STRUCTURE]
    rows += [{"family": "G2", "method": m, "reference": "GB-PG", "metric": "group_ap",
              **corrected_resampled_comparison(ga[m].to_numpy(), ga["GB-PG"].to_numpy(), TEST_TRAIN)} for m in GRAPH_MODELS]
    rows.append({"family": "G3", "method": "GB-PG - GB-P", "reference": "individual_ap", "metric": "gain difference",
                 **corrected_resampled_comparison((ga["GB-PG"] - ga["GB-P"]).to_numpy(), (ia["GB-PG"] - ia["GB-P"]).to_numpy(),
                                                  TEST_TRAIN)})
    c = pd.DataFrame(rows)
    c["t_p_holm"] = c.groupby("family")["t_p"].transform(lambda p: holm(p.tolist()))
    c.to_csv(results / "group_labels" / "comparisons.csv", index=False)
    first = df[df.method == "GB-P"]
    return {"comparisons": c, "group_ap": ga, "individual_ap": ia,
            "group_chance": float((first.n_group / (first.n_group + first.n_legit)).mean()),
            "individual_chance": float((first.n_other / (first.n_other + first.n_legit)).mean())}


def compare_retrained(results: Path) -> pd.DataFrame:
    rows = []
    for f in sorted((results / "group_labels" / "runs").glob("*.json")):
        rows += json.load(open(f))["rows"]
    df = pd.DataFrame(rows)
    piv = df.pivot_table(index="rotation", columns="method", values="group_ap")
    out = [{"method": "GB-PG", "reference": "GB-P", "n_rotations": 25,
            **corrected_resampled_comparison(piv["GB-PG"].to_numpy(), piv["GB-P"].to_numpy(), TEST_TRAIN)}]
    five = piv.loc[list(FOLDS)]
    gnn = [{"method": m, "reference": "GB-PG", "n_rotations": 5,
            **corrected_resampled_comparison(five[m].to_numpy(), five["GB-PG"].to_numpy(), TEST_TRAIN)} for m in RETRAIN_GNN]
    p = holm([r["t_p"] for r in gnn])
    for r, q in zip(gnn, p):
        r["t_p_holm"] = q
    c = pd.DataFrame(out + gnn)
    c["mean"] = [piv[m].mean() for m in c["method"]]
    c.to_csv(results / "group_labels" / "retrained_comparisons.csv", index=False)
    return c


def table(results: Path) -> None:
    r = compare(results)
    c, ga, ia = r["comparisons"], r["group_ap"], r["individual_ap"]
    label = {"SimpleHGN": "Simple-HGN"}
    lines = [r"\begin{table}[h]",
             r"\caption{Public data re-scored on the published group-fraud flags (25 rotations, models trained on the provider "
             r"fraud label, nothing retrained). Group AP: 100 flagged providers against non-fraud providers, other fraud "
             rf"providers excluded (chance {r['group_chance']:.3f}). Individual AP: the other fraud providers against "
             rf"non-fraud providers (chance {r['individual_chance']:.3f}). Mean $\pm$ s.d. over rotations. $\Delta$: difference "
             r"in group AP from GB-PG (or, for GB-PG and GB-PGL, from GB-P), with the corrected repeated cross-validation "
             r"95\% CI and Holm-adjusted $p$ within each family.}\label{tab:grouplabels}",
             r"\scriptsize", r"\begin{tabular}{@{}lcccc@{}}", r"\toprule",
             r"Method & Group AP & Individual AP & $\Delta$ [95\% CI] & $p$ \\", r"\midrule"]
    for m in METHODS:
        row = c[(c.method == m) & c.family.isin(["G1", "G2"])]
        delta = ["--", "--"]
        if len(row):
            v = row.iloc[0]
            delta = [f"${v.mean_diff:+.3f}$ $[{v.ci_low:.3f}, {v.ci_high:.3f}]$", "$<$0.001" if v.t_p_holm < 0.001 else f"{v.t_p_holm:.3f}"]
        lines.append(" & ".join([label.get(m, m), f"{ga[m].mean():.3f} $\\pm$ {ga[m].std():.3f}",
                                 f"{ia[m].mean():.3f} $\\pm$ {ia[m].std():.3f}"] + delta) + r" \\")
    lines += [r"\botrule", r"\end{tabular}", r"\end{table}"]
    (results / "tables" / "group_labels.tex").write_text("\n".join(lines))
    write_json(results / "tables" / "group_labels_chance.json", {k: r[k] for k in ["group_chance", "individual_chance"]})
    print(c[["family", "method", "reference", "mean_diff", "ci_low", "ci_high", "t_p", "t_p_holm"]].round(4).to_string())
    print(pd.DataFrame({"group_ap": ga.mean(), "individual_ap": ia.mean()}).round(3).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["describe", "rescore", "retrain", "table"], required=True)
    ap.add_argument("--results", type=Path, default=RESULTS_DIR)
    ap.add_argument("--kaggle-dir", type=Path, default=KAGGLE_DIR)
    ap.add_argument("--rdpgl-dir", type=Path, default=RDPGL_DIR)
    ap.add_argument("--repeats", type=int, nargs="+", default=list(REPEATS))
    ap.add_argument("--folds", type=int, nargs="+", default=list(FOLDS))
    ap.add_argument("--methods", nargs="+", default=RETRAIN_TABULAR)
    args = ap.parse_args()
    if args.stage == "table":
        table(args.results)
        return
    flags = group_flags(args.kaggle_dir, args.rdpgl_dir)
    print(f"mapping confirmed for {len(flags)} providers: {int(flags.fraud.sum())} fraud, {int(flags.group.sum())} group")
    if args.stage == "rescore":
        rescore(args.results, flags)
        return
    data = load(args.kaggle_dir)
    kaggle_exp._DATA = data
    if not (flags["provider_id"].to_numpy() == data.providers["provider_id"].to_numpy()).all():
        raise ValueError("flags are not aligned with the provider nodes")
    if args.stage == "describe":
        describe(data, flags, args.results)
        return
    hps = json.load(open(args.results / "kaggle" / "tuning" / "best_hyperparameters.json"))["best"]
    for repeat in args.repeats:
        for fold in args.folds:
            retrain_rotation(data, flags, hps, args.results, repeat, fold, args.methods)


if __name__ == "__main__":
    main()
