"""Aggregate run files into the tables and statistics reported in the manuscript.

Inputs : results/{synthetic,synthetic_persistent,ablation,kaggle}/runs/*.json
Outputs: results/tables/*.csv, results/tables/*.tex, results/summary.json

Every number in the manuscript's result tables is produced here.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..ledger import RESULTS_DIR, provenance, write_json
from ..metrics import clustered_comparison, corrected_resampled_comparison, holm, paired_comparison, t_ci

TABLES = RESULTS_DIR / "tables"
SYN_ORDER = ["LR-CA", "RF-CA", "GB-C", "GB-CA", "GB-CAG", "MLP-C", "SAGE", "RGCN", "HAN", "SimpleHGN", "HGT",
             "SAGE+AG", "HGT+AG", "Overlap-score"]
GNN_PURE = ["SAGE", "RGCN", "HAN", "SimpleHGN", "HGT"]
KAG_ORDER = ["LR-P", "RF-P", "GB-P", "GB-PG", "GB-PGL", "MLP-P", "SAGE", "RGCN", "HAN", "SimpleHGN", "HGT",
             "SAGE+G", "HGT+G"]


# Pre-specified design. Runs outside it (from an earlier, larger launch that was cut for compute)
# are kept on disk but excluded so that every cell has the same replicates.
DESIGN_SHIFTS = {0.0, 1.0, 2.0}
DESIGN_SEEDS = set(range(1, 7))
# Real-data rotations: one fold tests, three folds train, so the rotations' training sets overlap.
KAGGLE_TEST_TRAIN_RATIO = 1 / 3


def pooled_over_regimes(piv: pd.DataFrame, method: str, reference: str) -> dict:
    """Difference method - reference pooled over regimes, with the generator seed as the unit.

    ``piv`` is indexed by (ring_shift, ring_density, seed). The regimes of one seed share their
    legitimate claims, isolated fraud and ring members, so the seed means are tested (df = seeds - 1);
    the naive test that treats every regime-seed world as independent is kept for the audit trail.
    """
    a, b = piv[method].to_numpy(), piv[reference].to_numpy()
    naive = paired_comparison(a, b)
    res = clustered_comparison(a, b, piv.index.get_level_values("seed").to_numpy())
    return {**res, "naive_n": naive["n"], "naive_ci_low": naive["ci_low"], "naive_ci_high": naive["ci_high"],
            "naive_t_p": naive["t_p"]}


def load_rows(sub: str) -> pd.DataFrame:
    rows = []
    for f in sorted((RESULTS_DIR / sub / "runs").glob("*.json")):
        rows += json.load(open(f))["rows"]
    df = pd.DataFrame(rows)
    if sub.startswith("synthetic") and not df.empty:
        df = df[df["ring_shift"].isin(DESIGN_SHIFTS) & df["seed"].isin(DESIGN_SEEDS)].reset_index(drop=True)
    return df


def fmt(mean: float, sd: float, digits: int = 3) -> str:
    return "--" if not np.isfinite(mean) else f"{mean:.{digits}f} $\\pm$ {sd:.{digits}f}"


def regime_table(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    g = df.groupby(["ring_shift", "ring_density", "method"])[metric]
    t = g.agg(["mean", "std", "count"]).reset_index()
    return t


def comparisons(df: pd.DataFrame, metric: str, reference: str, methods: list[str]) -> pd.DataFrame:
    """Paired (by generator seed) comparison of each method against ``reference`` in every regime cell."""
    out = []
    for (shift, dens), cell in df.groupby(["ring_shift", "ring_density"]):
        piv = cell.pivot_table(index="seed", columns="method", values=metric)
        for m in methods:
            if m not in piv or reference not in piv:
                continue
            res = paired_comparison(piv[m].to_numpy(), piv[reference].to_numpy())
            out.append({"ring_shift": shift, "ring_density": dens, "method": m, "reference": reference, **res})
    out = pd.DataFrame(out)
    if len(out):
        # Family = one method-vs-reference comparison across the nine regime cells.
        out["t_p_holm"] = out.groupby("method")["t_p"].transform(lambda p: holm(p.tolist()))
        out["wilcoxon_p_holm"] = out.groupby("method")["wilcoxon_p"].transform(lambda p: holm(p.tolist()))
    return out


LABEL = {"SimpleHGN": "Simple-HGN", "Overlap-score": "Overlap score"}


def latex_regime(t: pd.DataFrame, methods: list[str], caption: str, label: str) -> str:
    """Methods as rows, the nine regimes as columns (grouped by feature shift); best per column in bold."""
    cells = sorted({(s, d) for s, d in zip(t["ring_shift"], t["ring_density"])})
    shifts = sorted({c[0] for c in cells})
    dens = sorted({c[1] for c in cells})
    best = t.loc[t.groupby(["ring_shift", "ring_density"])["mean"].idxmax()].set_index(["ring_shift", "ring_density"])["method"]
    head1 = " & ".join(rf"\multicolumn{{{len(dens)}}}{{c}}{{$\delta={s:g}$}}" for s in shifts)
    rules = "".join(rf"\cmidrule(lr){{{2 + i * len(dens)}-{1 + (i + 1) * len(dens)}}}" for i in range(len(shifts)))
    head2 = " & ".join(f"{d:g}" for _ in shifts for d in dens)
    lines = [r"\begin{table}[h]", rf"\caption{{{caption}}}\label{{{label}}}", r"\scriptsize",
             r"\setlength{\tabcolsep}{4pt}",
             r"\begin{tabular}{@{}l" + "c" * len(cells) + "@{}}", r"\toprule",
             r" & " + head1 + r" \\", rules, r"Method / $s$ & " + head2 + r" \\", r"\midrule"]
    groups = [["GB-C", "MLP-C", "LR-CA", "RF-CA", "GB-CA", "GB-CAG"], ["SAGE", "RGCN", "HAN", "SimpleHGN", "HGT"],
              ["SAGE+AG", "HGT+AG", "Overlap-score"]]
    for gi, group in enumerate(groups):
        for m in [m for m in group if m in methods]:
            row = [LABEL.get(m, m)]
            for c in cells:
                r = t[(t.ring_shift == c[0]) & (t.ring_density == c[1]) & (t.method == m)]
                v = f"{r['mean'].iloc[0]:.3f}" if len(r) else "--"
                row.append(rf"\textbf{{{v}}}" if best.get(c) == m else v)
            lines.append(" & ".join(row) + r" \\")
        if gi < len(groups) - 1:
            lines.append(r"\midrule")
    lines += [r"\botrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def _pfmt(p: float) -> str:
    return "$p<0.001$" if p < 0.001 else f"$p={p:.3f}$"


def _sgn(v: float) -> str:
    return f"${v:+.3f}$".replace("+0.000", "0.000").replace("-0.000", "0.000")


def _ci(lo: float, hi: float) -> str:
    return f"[{_sgn(lo)}, {_sgn(hi)}]"


def pooled_latex(pooled: pd.DataFrame) -> str:
    """Paired differences versus GB-CAG pooled over the nine regimes (seed as unit) for ring, isolated and overall AP."""
    ref = pooled[pooled.reference == "GB-CAG"]
    rows = ["SAGE", "RGCN", "HAN", "SimpleHGN", "HGT", "SAGE+AG", "HGT+AG", "MLP-C"]
    lines = [r"\begin{table}[h]",
             r"\caption{Paired difference in average precision versus GB-CAG, averaged over the nine regimes: "
             r"mean $\Delta$ $\pm$ half-width of the 95\% $t$-interval over the six generator seeds, and "
             r"Holm-adjusted paired $t$-test $p$ (within each metric). The nine regimes of a seed share its "
             r"legitimate claims, isolated fraud and ring members, so the seed, not the world, is the unit of "
             r"replication.}\label{tab:pooled}",
             r"\scriptsize", r"\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{@{}lcccccc@{}}", r"\toprule",
             r" & \multicolumn{2}{c}{Ring AP} & \multicolumn{2}{c}{Isolated AP} & \multicolumn{2}{c}{Overall AP} \\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
             r"Method & $\Delta \pm$ CI & $p$ & $\Delta \pm$ CI & $p$ & $\Delta \pm$ CI & $p$ \\", r"\midrule"]
    for m in rows:
        cells = [LABEL.get(m, m)]
        for metric in ["ring_ap", "isolated_ap", "ap"]:
            r = ref[(ref.method == m) & (ref.metric == metric)]
            if r.empty:
                cells += ["--", "--"]
                continue
            r = r.iloc[0]
            cells += [f"{_sgn(r.mean_diff)} $\\pm$ {(r.ci_high - r.ci_low) / 2:.3f}",
                      "$<$0.001" if r.t_p_holm < 0.001 else f"{r.t_p_holm:.3f}"]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\botrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def summarise_synthetic(summary: dict) -> None:
    df = load_rows("synthetic")
    if df.empty:
        return
    df = df.rename(columns={"seed": "seed"})
    df["seed"] = df["seed"].astype(int)
    n_seeds = df.groupby(["ring_shift", "ring_density"])["seed"].nunique()
    summary["synthetic"] = {"n_datasets": int(df[["ring_shift", "ring_density", "seed"]].drop_duplicates().shape[0]),
                            "seeds_per_cell": n_seeds.to_dict().__repr__()}
    for metric in ["ring_ap", "isolated_ap", "ap", "f1", "recall_at_5pct", "roc_auc", "ring_recall"]:
        t = regime_table(df, metric)
        t.to_csv(TABLES / f"synthetic_{metric}.csv", index=False)
    t = regime_table(df, "ring_ap")
    show = SYN_ORDER
    (TABLES / "synthetic_ring_ap.tex").write_text(latex_regime(
        t, show, "Ring-fraud detection on unseen test rings: average precision of ring versus legitimate "
                 "claims (chance level 0.08), mean over six worlds per regime. Bold: best per regime. "
                 "Standard deviations are in the supplementary results files.", "tab:regime_ringap"))
    tiso = regime_table(df, "isolated_ap")
    (TABLES / "synthetic_isolated_ap.tex").write_text(latex_regime(
        tiso, show, "Isolated-fraud detection: average precision of isolated-fraud versus legitimate claims "
                    "(chance level 0.03), mean over six worlds per regime. Bold: best per regime.",
        "tab:regime_isoap"))
    comps = []
    for ref in ["GB-CAG", "GB-CA"]:
        c = comparisons(df, "ring_ap", ref, GNN_PURE + ["SAGE+AG", "HGT+AG"] + (["GB-CAG"] if ref == "GB-CA" else []))
        comps.append(c)
    comp = pd.concat(comps, ignore_index=True)
    comp.to_csv(TABLES / "synthetic_ring_ap_comparisons.csv", index=False)
    # Overall (all regimes pooled): seed-level differences averaged over the nine regimes
    pooled = []
    for metric in ["ring_ap", "isolated_ap", "ap"]:
        piv = df.pivot_table(index=["ring_shift", "ring_density", "seed"], columns="method", values=metric)
        for ref in ["GB-CAG", "GB-CA"]:
            for m in GNN_PURE + ["SAGE+AG", "HGT+AG", "GB-CAG", "MLP-C"]:
                if m == ref or m not in piv:
                    continue
                pooled.append({"metric": metric, "method": m, "reference": ref,
                               **pooled_over_regimes(piv, m, ref)})
    pooled = pd.DataFrame(pooled)
    pooled["t_p_holm"] = pooled.groupby(["metric", "reference"])["t_p"].transform(lambda p: holm(p.tolist()))
    pooled["naive_t_p_holm"] = pooled.groupby(["metric", "reference"])["naive_t_p"].transform(lambda p: holm(p.tolist()))
    pooled.to_csv(TABLES / "synthetic_pooled.csv", index=False)
    pooled[pooled.metric == "ring_ap"].to_csv(TABLES / "synthetic_ring_ap_pooled.csv", index=False)
    (TABLES / "synthetic_pooled.tex").write_text(pooled_latex(pooled))
    summary["synthetic"]["best_method_per_cell"] = (
        t.loc[t.groupby(["ring_shift", "ring_density"])["mean"].idxmax()]
        .assign(cell=lambda x: x.ring_shift.astype(str) + "/" + x.ring_density.astype(str))
        .set_index("cell")["method"].to_dict())
    timing = df.groupby("method")[["train_seconds", "infer_seconds"]].median()
    timing.to_csv(TABLES / "synthetic_timing.csv")


# Low-volume stress test: runs with frozen hyperparameters (tuned at default volume) and with the
# tuning protocol repeated on low-volume development worlds.
LOWVOL_SETS = {"frozen": "synthetic_lowvolume", "retuned": "synthetic_lowvolume_retuned"}
LOWVOL_LABEL = {"default": "Default volume", "frozen": "Low volume, frozen", "retuned": "Low volume, re-tuned"}


def volume_signal(summary: dict, seeds=range(1, 7), densities=(0.1, 0.5, 0.9)) -> None:
    """Test-period claim volume of ring members versus everyone else, at default and low ring volume (delta = 0)."""
    from ..synth.generator import GeneratorConfig, generate
    rows = []
    for volume in [GeneratorConfig.ring_claims_per_month, 4.0]:
        for dens in densities:
            for seed in seeds:
                w = generate(GeneratorConfig(ring_shift=0.0, ring_density=dens, seed=seed, ring_claims_per_month=volume))
                test = w.split == "test"
                c, ft = w.claims.loc[test], w.truth["fraud_type"].to_numpy()[test]
                rp, rd = set(c.loc[ft == "ring", "patient_id"]), set(c.loc[ft == "ring", "doctor_id"])
                pc, dc = c.groupby("patient_id").size(), c.groupby("doctor_id").size()
                is_rp, is_rd = pc.index.isin(list(rp)), dc.index.isin(list(rd))
                rows.append({"volume": volume, "ring_density": dens, "seed": seed,
                             "ring_prevalence": float((ft == "ring").sum() / (ft != "isolated").sum()),
                             "ring_patient_claims": pc[is_rp].mean(), "other_patient_claims": pc[~is_rp].mean(),
                             "other_patient_claims_p95": pc[~is_rp].quantile(0.95),
                             "other_patient_claims_p99": pc[~is_rp].quantile(0.99),
                             "ring_doctor_claims": dc[is_rd].mean(), "other_doctor_claims": dc[~is_rd].mean()})
    v = pd.DataFrame(rows)
    out = v.groupby("volume").mean(numeric_only=True).drop(columns=["seed", "ring_density"])
    out.to_csv(TABLES / "volume_signal.csv")
    summary["volume_signal"] = out.round(3).to_dict(orient="index")


def lowvolume_latex(t: pd.DataFrame, settings: list[str]) -> str:
    dens = sorted(t.ring_density.unique())
    t = t.set_index(["setting", "ring_density", "method"])["ring_ap_mean"]
    cols = [(s, d) for s in settings for d in dens]
    best = {c: t.loc[c[0], c[1]].idxmax() for c in cols}
    head = " & ".join(rf"\multicolumn{{{len(dens)}}}{{c}}{{{LOWVOL_LABEL[s]}}}" for s in settings)
    rules = "".join(rf"\cmidrule(lr){{{2 + i * len(dens)}-{1 + (i + 1) * len(dens)}}}" for i in range(len(settings)))
    lines = [r"\begin{table}[h]",
             r"\caption{Ring AP on unseen test rings at $\delta=0$ when ring volume is reduced from 16 to 4 claims per "
             r"month (mean over six worlds per cell). At low volume a ring patient files about as many claims as the 95th "
             r"percentile of other patients and a ring doctor 1.3 times as many as other doctors (default: 5.7 and 2.5 times "
             r"the averages), and the chance level of ring AP falls from 0.080 to 0.021. Frozen: "
             r"hyperparameters tuned at default volume; re-tuned: the tuning protocol repeated on low-volume development "
             r"worlds (hybrids inherit their base configuration). Bold: best per column.}\label{tab:lowvolume}",
             r"\scriptsize", r"\setlength{\tabcolsep}{3.5pt}",
             r"\begin{tabular}{@{}l" + "c" * len(cols) + "@{}}", r"\toprule", r" & " + head + r" \\", rules,
             r"Method / $s$ & " + " & ".join(f"{d:g}" for _ in settings for d in dens) + r" \\", r"\midrule"]
    groups = [["GB-C", "MLP-C", "LR-CA", "RF-CA", "GB-CA", "GB-CAG"], ["SAGE", "RGCN", "HAN", "SimpleHGN", "HGT"],
              ["SAGE+AG", "HGT+AG", "Overlap-score"]]
    for gi, group in enumerate(groups):
        for m in group:
            row = [LABEL.get(m, m)]
            for c in cols:
                v = t.get((c[0], c[1], m), np.nan)
                s = "--" if not np.isfinite(v) else f"{v:.3f}"
                row.append(rf"\textbf{{{s}}}" if best[c] == m else s)
            lines.append(" & ".join(row) + r" \\")
        if gi < len(groups) - 1:
            lines.append(r"\midrule")
    lines += [r"\botrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def summarise_lowvolume(summary: dict) -> None:
    """Stress test: ring volume 4 instead of 16 claims per month (delta = 0), against the default-volume runs."""
    base = load_rows("synthetic")
    sets = {name: load_rows(sub) for name, sub in LOWVOL_SETS.items()}
    sets = {name: df[df.ring_shift == 0.0] for name, df in sets.items() if not df.empty}
    if base.empty or not sets:
        return
    frames = [base[base.ring_shift == 0.0].assign(setting="default")] + [df.assign(setting=k) for k, df in sets.items()]
    both = pd.concat(frames, ignore_index=True)
    both["ring_chance"] = both["n_ring"] / (both["n_test"] - both["n_isolated"])
    t = both.groupby(["setting", "ring_density", "method"])[["ring_ap", "isolated_ap", "ap", "ring_chance"]].agg(["mean", "std", "count"])
    t.columns = [f"{a}_{b}" for a, b in t.columns]
    t = t.reset_index()
    t.to_csv(TABLES / "lowvolume_ring_ap.csv", index=False)
    (TABLES / "lowvolume.tex").write_text(lowvolume_latex(t, ["default"] + list(sets)))
    rows = []
    for setting, g in sets.items():
        for metric in ["ring_ap", "isolated_ap", "ap"]:
            piv = g.pivot_table(index=["ring_shift", "ring_density", "seed"], columns="method", values=metric)
            for ref in ["GB-CAG", "GB-CA"]:
                for m in [m for m in SYN_ORDER if m in piv and m != ref]:
                    for dens in sorted(g.ring_density.unique()):
                        cell = piv.xs(dens, level="ring_density")
                        rows.append({"setting": setting, "metric": metric, "reference": ref, "method": m,
                                     "ring_density": dens, **paired_comparison(cell[m].to_numpy(), cell[ref].to_numpy())})
                    rows.append({"setting": setting, "metric": metric, "reference": ref, "method": m,
                                 "ring_density": "pooled", **pooled_over_regimes(piv, m, ref)})
    c = pd.DataFrame(rows)
    cell = c.ring_density != "pooled"
    # Families: one method-vs-reference comparison across the three densities; pooled rows across methods.
    keys = ["setting", "metric", "reference"]
    c.loc[cell, "t_p_holm"] = c[cell].groupby(keys + ["method"])["t_p"].transform(lambda p: holm(p.tolist()))
    c.loc[~cell, "t_p_holm"] = c[~cell].groupby(keys)["t_p"].transform(lambda p: holm(p.tolist()))
    c.to_csv(TABLES / "lowvolume_comparisons.csv", index=False)
    pooled = c[(c.metric == "ring_ap") & (c.reference == "GB-CAG") & ~cell]
    summary["lowvolume"] = {
        "worlds_per_density": {k: df.groupby("ring_density")["seed"].nunique().to_dict().__repr__() for k, df in sets.items()},
        "ring_chance": float(t[t.setting != "default"]["ring_chance_mean"].mean()),
        "ring_ap_pooled_vs_gbcag": {s: g.set_index("method")[["mean_diff", "ci_low", "ci_high", "t_p_holm"]].round(4)
                                    .to_dict(orient="index") for s, g in pooled.groupby("setting")}}


def summarise_persistent(summary: dict) -> None:
    per = load_rows("synthetic_persistent")
    emg = load_rows("synthetic")
    if per.empty or emg.empty:
        return
    keys = ["ring_shift", "ring_density", "seed"]
    per = per.assign(protocol="persistent")
    emg = emg.merge(per[keys].drop_duplicates(), on=keys).assign(protocol="emergent")
    both = pd.concat([emg, per], ignore_index=True)
    t = both.groupby(["protocol", "ring_shift", "ring_density", "method"])["ring_ap"].agg(["mean", "std", "count"]).reset_index()
    t.to_csv(TABLES / "protocol_ring_ap.csv", index=False)
    # memorisation gap = persistent - emergent, paired by seed
    out = []
    for (s, d, m), g in both.groupby(["ring_shift", "ring_density", "method"]):
        piv = g.pivot_table(index="seed", columns="protocol", values="ring_ap")
        if {"persistent", "emergent"} <= set(piv.columns):
            out.append({"ring_shift": s, "ring_density": d, "method": m,
                        **paired_comparison(piv["persistent"].to_numpy(), piv["emergent"].to_numpy())})
    gap = pd.DataFrame(out)
    gap.to_csv(TABLES / "protocol_gap.csv", index=False)
    lines = [r"\begin{table}[h]",
             r"\caption{Ring AP when test rings are new (emergent) versus recurring members of training rings "
             r"(persistent), $\delta=0$, six worlds each; gain = persistent $-$ emergent (paired over worlds, "
             r"95\% $t$-interval). The overlap score is unsupervised and cannot memorise.}\label{tab:protocol}",
             r"\scriptsize", r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{@{}lccp{2.2cm}ccp{2.2cm}@{}}", r"\toprule",
             r" & \multicolumn{3}{c}{$s=0.1$} & \multicolumn{3}{c}{$s=0.9$} \\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
             r"Method & new & recurring & gain & new & recurring & gain \\", r"\midrule"]
    for m in ["MLP-C", "GB-CA", "GB-CAG", "SAGE", "HGT", "Overlap-score"]:
        cells = [LABEL.get(m, m)]
        for d in [0.1, 0.9]:
            r = t[(t.ring_shift == 0.0) & (t.ring_density == d) & (t.method == m)].set_index("protocol")["mean"]
            gg = gap[(gap.ring_density == d) & (gap.method == m)]
            if len(r) < 2 or gg.empty:
                cells += ["--"] * 3
                continue
            gg = gg.iloc[0]
            cells += [f"{r['emergent']:.3f}", f"{r['persistent']:.3f}",
                      f"{_sgn(gg.mean_diff)} {_ci(gg.ci_low, gg.ci_high)}"]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\botrule", r"\end{tabular}", r"\end{table}"]
    (TABLES / "protocol.tex").write_text("\n".join(lines))


ABLATION_NAMES = {
    "full": "Full model", "rewired": "Degree-preserving rewiring", "no_visits": "No patient--doctor relation",
    "no_degree": "No degree encodings", "layers_1": "1 message-passing layer", "layers_2": "2 layers",
    "no_reverse": "No reverse relations (legacy graph)", "ring_id_leak": "+ ring identifier (legacy leak)",
    "layers_4": "4 layers", "heads_1": "1 attention head", "heads_2": "2 heads", "heads_8": "8 heads",
    "no_files": "No patient--claim relation", "no_billed_by": "No claim--doctor relation",
    "no_hospital": "No hospital nodes", "no_claim_attributes": "Claim attributes zeroed",
    "no_entity_attributes": "Entity attributes zeroed", "degree_only": "Degree encodings only"}
# The HGT single-factor table (rows grouped by factor); every other ablation is in the mechanism table.
HGT_ARCH_GROUPS = [("Depth (full: 3)", ["layers_1", "layers_2", "layers_4"]),
                   ("Heads (full: 4)", ["heads_1", "heads_2", "heads_8"]),
                   ("Relations", ["no_files", "no_billed_by", "no_visits", "no_hospital"]),
                   ("Node inputs", ["no_degree", "no_claim_attributes", "no_entity_attributes", "degree_only"])]


def summarise_ablation(summary: dict) -> None:
    df = pd.concat([load_rows("ablation"), load_rows("ablation_arch")], ignore_index=True)
    if df.empty:
        return
    t = df.groupby(["ring_density", "method", "variant"])[["ring_ap", "isolated_ap"]].agg(["mean", "std", "count"])
    t.columns = [f"{a}_{b}" for a, b in t.columns]
    t = t.reset_index()
    t.to_csv(TABLES / "ablation_ring_ap.csv", index=False)
    out = []
    for (d, m), g in df.groupby(["ring_density", "method"]):
        piv = g.pivot_table(index="seed", columns="variant", values="ring_ap")
        piv_iso = g.pivot_table(index="seed", columns="variant", values="isolated_ap")
        for v in piv.columns:
            if v != "full":
                iso = paired_comparison(piv_iso[v].to_numpy(), piv_iso["full"].to_numpy())
                out.append({"ring_density": d, "method": m, "variant": v,
                            **paired_comparison(piv[v].to_numpy(), piv["full"].to_numpy()),
                            "iso_mean_diff": iso["mean_diff"], "iso_t_p": iso["t_p"]})
    o = pd.DataFrame(out)
    # One family over every ablation comparison (mechanism and HGT single-factor tables together).
    o["t_p_holm"] = holm(o["t_p"].tolist())
    o["iso_t_p_holm"] = holm(o["iso_t_p"].tolist())
    o.to_csv(TABLES / "ablation_comparisons.csv", index=False)

    def cell(m, v, metric="ring_ap", sd=True):
        r = t[(t.method == m) & (t.variant == v)].iloc[0]
        return f"{r[metric + '_mean']:.3f}" + (f" $\\pm$ {r[metric + '_std']:.3f}" if sd else "")

    def delta(m, v, iso=False):
        c = o[(o.method == m) & (o.variant == v)].iloc[0]
        return (f"{_sgn(c.iso_mean_diff)} ({_pfmt(c.iso_t_p_holm)})" if iso
                else f"{_sgn(c.mean_diff)} ({_pfmt(c.t_p_holm)})")

    lines = [r"\begin{table}[h]",
             r"\caption{Mechanism ablations at $\delta=0$, $s=0.9$ (six worlds): ring AP (mean $\pm$ s.d.) and paired "
             r"change from the full model, with Holm-adjusted $p$ over every ablation comparison in this table and "
             r"Table~\ref{tab:hgt_arch}. The tuned GNNs use three layers; HGT's single-factor variants are in "
             r"Table~\ref{tab:hgt_arch}.}\label{tab:ablation}",
             r"\scriptsize", r"\begin{tabular}{@{}llcc@{}}", r"\toprule",
             r"Method & Variant & Ring AP & $\Delta$ vs.\ full \\", r"\midrule"]
    mech = {"GB-CA": ["full", "rewired", "ring_id_leak"], "GB-CAG": ["full", "rewired"],
            "SAGE": ["full", "rewired", "no_visits", "no_degree", "layers_1", "layers_2"],
            "HGT": ["full", "rewired", "no_reverse"]}
    for m, variants in mech.items():
        for v in variants:
            if t[(t.method == m) & (t.variant == v)].empty:
                continue
            lines.append(f"{m if v == 'full' else ''} & {ABLATION_NAMES[v]} & {cell(m, v)} & "
                         f"{'--' if v == 'full' else delta(m, v)} \\\\")
        lines.append(r"\midrule" if m != "HGT" else "")
    lines += [r"\botrule", r"\end{tabular}", r"\end{table}"]
    (TABLES / "ablation.tex").write_text("\n".join(l for l in lines if l))

    arch = [r"\begin{table}[h]",
            r"\caption{HGT single-factor ablations at $\delta=0$, $s=0.9$ (six worlds; one factor changed at a time "
            r"from the tuned configuration: 3 layers, 4 heads, hidden size 32). Ring AP (mean $\pm$ s.d.), isolated AP "
            r"(mean) and paired change from the full model; Holm-adjusted $p$ over every ablation comparison in this table "
            r"and Table~\ref{tab:ablation}.}\label{tab:hgt_arch}",
            r"\scriptsize", r"\setlength{\tabcolsep}{2.5pt}", r"\begin{tabular}{@{}lcccc@{}}", r"\toprule",
            r"Variant & Ring AP & $\Delta$ ring AP & Isol.\ AP & $\Delta$ isol.\ AP \\", r"\midrule",
            f"Full model & {cell('HGT', 'full')} & -- & {cell('HGT', 'full', 'isolated_ap', sd=False)} & -- \\\\"]
    for title, variants in HGT_ARCH_GROUPS:
        present = [v for v in variants if not t[(t.method == "HGT") & (t.variant == v)].empty]
        if not present:
            continue
        arch += [r"\midrule", rf"\multicolumn{{5}}{{@{{}}l}}{{\textit{{{title}}}}} \\"]
        for v in present:
            name = "1 layer" if v == "layers_1" else ABLATION_NAMES[v]
            arch.append(f"{name} & {cell('HGT', v)} & {delta('HGT', v)} & "
                        f"{cell('HGT', v, 'isolated_ap', sd=False)} & {delta('HGT', v, iso=True)} \\\\")
    arch += [r"\botrule", r"\end{tabular}", r"\end{table}"]
    (TABLES / "hgt_arch.tex").write_text("\n".join(arch))
    summary["ablation"] = {"n_comparisons_in_holm_family": int(len(o)),
                           "hgt_arch_worlds": int(df[df.variant.isin(sum((v for _, v in HGT_ARCH_GROUPS), []))]
                                                  .groupby("variant")["seed"].nunique().min())}


def summarise_kaggle(summary: dict) -> None:
    df = load_rows("kaggle")
    if df.empty:
        return
    t = df.groupby("method")[["ap", "roc_auc", "f1", "precision", "recall", "recall_at_10pct", "train_seconds"]].agg(["mean", "std"])
    t.to_csv(TABLES / "kaggle_metrics.csv")
    rows = []
    piv = df.pivot_table(index="fold", columns="method", values="ap")
    for ref in ["GB-P", "GB-PGL"]:
        for m in KAG_ORDER:
            if m != ref and m in piv and ref in piv:
                rows.append({"method": m, "reference": ref, **corrected_resampled_comparison(
                    piv[m].to_numpy(), piv[ref].to_numpy(), KAGGLE_TEST_TRAIN_RATIO)})
    c = pd.DataFrame(rows)
    c["t_p_holm"] = holm(c["t_p"].tolist())
    c["naive_t_p_holm"] = holm(c["naive_t_p"].tolist())
    c.to_csv(TABLES / "kaggle_comparisons.csv", index=False)
    lines = [r"\begin{table}[h]",
             r"\caption{Provider-level fraud detection on the public provider-labelled data "
             r"(5-fold rotation; mean $\pm$ s.d. over folds). F1 at the validation-selected threshold; "
             r"Recall@10\%: share of all fraudulent providers that rank among the top 10\% of providers by score."
             r"}\label{tab:kaggle}",
             r"\footnotesize", r"\begin{tabular}{@{}lcccc@{}}", r"\toprule",
             r"Method & AP & ROC-AUC & F1 & Recall@10\% \\", r"\midrule"]
    best_ap = df.groupby("method")["ap"].mean().idxmax()
    for m in KAG_ORDER:
        g = df[df.method == m]
        if g.empty:
            continue
        cells = [fmt(g[k].mean(), g[k].std(ddof=1)) for k in ["ap", "roc_auc", "f1", "recall_at_10pct"]]
        name = rf"\textbf{{{LABEL.get(m, m)}}}" if m == best_ap else LABEL.get(m, m)
        lines.append(" & ".join([name] + cells) + r" \\")
    lines += [r"\botrule", r"\end{tabular}", r"\end{table}"]
    (TABLES / "kaggle.tex").write_text("\n".join(lines))
    summary["kaggle"] = {"n_folds": int(df.fold.nunique()),
                         "mean_ap": df.groupby("method")["ap"].mean().round(4).to_dict()}


def kaggle_bootstrap(summary: dict, n_boot: int = 1000) -> None:
    """Provider-level bootstrap of the AP difference from GB-P, holding the trained models fixed.

    Providers are resampled within each rotation's test fold and label, and AP is averaged over the
    five rotations as in Table tab:kaggle. The interval reflects which providers happen to be tested,
    not retraining variability (which the corrected test over rotations addresses).
    """
    files = sorted((RESULTS_DIR / "kaggle" / "predictions").glob("fold*_*.npz"))
    if not files:
        return
    from sklearn.metrics import average_precision_score
    scores, y, fold = {}, {}, {}
    for f in files:
        k = int(f.name[len("fold")])
        z = np.load(f, allow_pickle=True)
        y[k] = z["y"]
        for m in z.files:
            if m not in ("provider_id", "y"):
                scores.setdefault(m, {})[k] = z[m]
    folds = sorted(y)
    rng = np.random.default_rng(0)
    draws = [{k: np.concatenate([rng.choice(np.flatnonzero(y[k] == lab), (y[k] == lab).sum()) for lab in (0, 1)])
              for k in folds} for _ in range(n_boot)]

    def mean_ap(m, idx=None):
        return np.mean([average_precision_score(y[k] if idx is None else y[k][idx[k]],
                                                scores[m][k] if idx is None else scores[m][k][idx[k]]) for k in folds])

    ref = {i: mean_ap("GB-P", d) for i, d in enumerate(draws)}
    rows = []
    for m in [m for m in KAG_ORDER if m in scores and m != "GB-P"]:
        diffs = np.array([mean_ap(m, d) - ref[i] for i, d in enumerate(draws)])
        rows.append({"method": m, "reference": "GB-P", "mean_diff": mean_ap(m) - mean_ap("GB-P"),
                     "boot_ci_low": float(np.quantile(diffs, 0.025)), "boot_ci_high": float(np.quantile(diffs, 0.975)),
                     "share_boot_below_zero": float((diffs < 0).mean()), "n_boot": n_boot})
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "kaggle_bootstrap.csv", index=False)
    summary["kaggle_bootstrap"] = out.set_index("method")[["mean_diff", "boot_ci_low", "boot_ci_high"]].round(4).to_dict(orient="index")


CASCADE_ORDER = ["GB-CAG", "HGT", "SAGE", "HGT+AG", "SAGE+AG", "GB-CAG>HGT (rerank)", "GB-CAG>HGT (fuse)",
                 "GB-CAG>SAGE (rerank)", "GB-CAG>SAGE (fuse)"]
CASCADE_LABEL = {"GB-CAG>HGT (rerank)": r"GB-CAG$\to$HGT, re-rank", "GB-CAG>HGT (fuse)": r"GB-CAG$\to$HGT, fused",
                 "GB-CAG>SAGE (rerank)": r"GB-CAG$\to$SAGE, re-rank", "GB-CAG>SAGE (fuse)": r"GB-CAG$\to$SAGE, fused"}


def load_cascade() -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    rows, curves, runs = [], [], []
    for f in sorted((RESULTS_DIR / "cascade" / "runs").glob("*.json")):
        r = json.load(open(f))
        rows += r["rows"]
        curves += r["curves"]
        runs.append(r)
    return pd.DataFrame(rows), pd.DataFrame(curves), runs


def summarise_cascade(summary: dict) -> None:
    df, curves, runs = load_cascade()
    if df.empty:
        return
    keys = ["ring_shift", "ring_density", "seed"]
    metrics = ["ap", "ring_ap", "isolated_ap", "precision_at_5pct", "ring_recall_at_5pct", "isolated_recall_at_5pct",
               "rings_with_any_claim_in_budget", "rings_with_half_claims_in_budget"]
    agg = df.groupby("method")[metrics + ["escalated_test"]].mean().reindex(CASCADE_ORDER)
    agg.to_csv(TABLES / "cascade_means.csv")
    comps = []
    for metric in ["ap", "ring_ap", "isolated_ap", "precision_at_5pct"]:
        piv = df.pivot_table(index=keys, columns="method", values=metric)
        for m in CASCADE_ORDER[1:]:
            if m in piv:
                comps.append({"metric": metric, "method": m, **pooled_over_regimes(piv, m, "GB-CAG")})
    comps = pd.DataFrame(comps)
    comps["t_p_holm"] = comps.groupby("metric")["t_p"].transform(lambda p: holm(p.tolist()))
    comps["naive_t_p_holm"] = comps.groupby("metric")["naive_t_p"].transform(lambda p: holm(p.tolist()))
    comps.to_csv(TABLES / "cascade_comparisons.csv", index=False)
    by_regime = df.groupby(["ring_shift", "ring_density", "method"])[["ap", "ring_ap", "isolated_ap"]].mean().reset_index()
    by_regime.to_csv(TABLES / "cascade_by_regime.csv", index=False)
    curve = curves.groupby(["method", "mode", "escalation"])[["ap", "ring_ap", "isolated_ap", "escalated_test", "val_ap"]]
    curve.mean().reset_index().to_csv(TABLES / "cascade_curves.csv", index=False)
    selected = (df[df.method.str.contains(">")].groupby("method")["escalation"]
                .value_counts().unstack(fill_value=0).to_dict(orient="index"))

    lines = [r"\begin{table}[h]",
             r"\caption{Two-stage routing versus single models and feature-level hybrids, averaged over the 54 worlds "
             r"(same test claims for every row). Routing escalates claims whose GB-CAG score exceeds a threshold set "
             r"on validation data; the escalation fraction is chosen by validation AP. P@5\%: precision among the top "
             r"5\% of test claims. Esc.: mean share of test claims escalated. $\Delta$AP: paired difference in AP from "
             r"GB-CAG ($\pm$ half-width of the 95\% $t$-interval over the six generator seeds), Holm-adjusted $p$ "
             r"over the rows.}\label{tab:cascade}",
             r"\scriptsize", r"\setlength{\tabcolsep}{2.5pt}", r"\begin{tabular}{@{}lcccccc@{}}", r"\toprule",
             r"Method & AP & Ring AP & Isol.\ AP & P@5\% & Esc. & $\Delta$AP ($p$) \\", r"\midrule"]
    for m in CASCADE_ORDER:
        if m not in agg.index or not np.isfinite(agg.loc[m, "ap"]):
            continue
        r = agg.loc[m]
        esc = "--" if not np.isfinite(r["escalated_test"]) else f"{r['escalated_test']:.2f}"
        c = comps[(comps.metric == "ap") & (comps.method == m)]
        d = "--" if c.empty else (f"{_sgn(c.iloc[0].mean_diff)} $\\pm$ {(c.iloc[0].ci_high - c.iloc[0].ci_low) / 2:.3f} "
                                  f"({'$<$0.001' if c.iloc[0].t_p_holm < 0.001 else f'{c.iloc[0].t_p_holm:.3f}'})")
        lines.append(f"{CASCADE_LABEL.get(m, m)} & {r['ap']:.3f} & {r['ring_ap']:.3f} & {r['isolated_ap']:.3f} & "
                     f"{r['precision_at_5pct']:.3f} & {esc} & {d} \\\\")
        if m in ("SAGE", "SAGE+AG"):
            lines.append(r"\midrule")
    lines += [r"\botrule", r"\end{tabular}", r"\end{table}"]
    (TABLES / "cascade.tex").write_text("\n".join(lines))

    repro = {}
    for r in runs:
        for m, v in r["reproduction"].items():
            repro.setdefault(m, []).append(v["max_abs_diff"])
    reliance = []
    for r in runs:
        cfg = r["provenance"]["config"]
        base = {"ring_shift": cfg["ring_shift"], "ring_density": cfg["ring_density"], "seed": cfg["seed"]}
        for model, groups in r["reliance"].items():
            if model == "rewired_test_period":
                for m, v in groups.items():
                    reliance.append({**base, "model": m, "intervention": "rewired_test_period", **v})
            else:
                for g, v in groups.items():
                    reliance.append({**base, "model": model, "intervention": g, **v})
    rel = pd.DataFrame(reliance)
    rel_t = rel.groupby(["ring_shift", "ring_density", "model", "intervention"])[["ring_ap", "isolated_ap", "ap"]].mean()
    rel_t.reset_index().to_csv(TABLES / "reliance.csv", index=False)
    summary["cascade"] = {
        "n_worlds": int(df[keys].drop_duplicates().shape[0]),
        "reproduction_max_abs_diff": {m: float(np.max(v)) for m, v in repro.items()},
        "reproduction_worlds": {m: len(v) for m, v in repro.items()},
        "selected_escalation_counts": selected,
        "means": agg.round(4).to_dict(orient="index")}


def summarise_legacy(summary: dict) -> None:
    path = RESULTS_DIR / "legacy" / "legacy_reanalysis.json"
    if not path.exists():
        return
    r = json.load(open(path))
    leg, bench = r["legacy"], r["benchmark_s0.5"]
    syn = load_rows("synthetic")
    gbc = syn[syn.method == "GB-C"].groupby("ring_shift")[["ring_ap", "ring_recall"]].mean()
    comp = leg["split_composition"]
    n_test = sum(comp["test"].values())
    gb = leg["gb_raw_claim_features"]
    lo, hi = gb["ring_recall_ci"]
    b0, b2 = bench["delta_0"], bench["delta_2"]
    rows = [
        ("Claims in total (in the test split / period)", f"{leg['n_claims']:,} ({n_test})",
         f"{b0['n_claims']:,.0f} ({b0['n_test_claims']:,.0f}), mean per world"),
        ("Ring claims in test", f"{comp['test']['ring']} (3--6 per ring)", f"{b0['test_ring_claims']:.0f} (4 rings $\\times$ 96)"),
        ("Rings", "5, active all 24 months", "4 per period, each active 6 months"),
        ("Test-ring doctors with labelled ring claims in training",
         f"{100 * leg['test_ring_members_seen_in_training_rings']['doctor']:.0f}\\%", "0\\% (by design)"),
        ("Ring vs.\\ legitimate log-amount shift (s.d.)", f"{leg['separability']['log_amount_shift_sd']:.2f}",
         f"{b0['log_amount_shift_sd']:.2f} ($\\delta=0$) to {b2['log_amount_shift_sd']:.2f} ($\\delta=2$)"),
        ("ROC-AUC of claim amount alone, ring vs.\\ legitimate", f"{leg['separability']['auc_claim_amount']:.2f}",
         f"{b0['auc_claim_amount']:.2f} to {b2['auc_claim_amount']:.2f}"),
        ("Ring AP, GB on claim attributes only", f"{gb['ring_ap']:.3f}",
         f"{gbc.loc[0.0, 'ring_ap']:.3f} to {gbc.loc[2.0, 'ring_ap']:.3f}"),
        ("Ring recall, GB on claim attributes only", f"{gb['ring_recall_counts'][0]}/{gb['ring_recall_counts'][1]} "
         f"[{lo:.2f}, {hi:.2f}]", f"{gbc.loc[0.0, 'ring_recall']:.2f} to {gbc.loc[2.0, 'ring_recall']:.2f}"),
        ("Independent datasets", "1", "54"),
    ]
    lines = [r"\begin{table}[h]",
             r"\caption{A small fixed benchmark (2,100 claims, five rings; an earlier, simpler generator of ours, "
             r"included in the repository), "
             r"re-analysed with its own split (date-sorted, 60/20/20), compared with our benchmark. Benchmark ranges span $\delta=0$ to $\delta=2$ "
             r"($s=0.5$, six worlds each; ring AP and recall of GB-C pooled over $s$). Shift and ROC-AUC use raw claim "
             r"amounts, not adjusted for specialty, for both. Interval: exact 95\% binomial.}"
             r"\label{tab:legacy}",
             r"\scriptsize", r"\begin{tabular}{@{}p{5.0cm}p{3.1cm}p{3.6cm}@{}}", r"\toprule",
             r" & Fixed dataset & This benchmark \\", r"\midrule"]
    lines += [f"{a} & {b} & {c} \\\\" for a, b, c in rows]
    lines += [r"\botrule", r"\end{tabular}", r"\end{table}"]
    (TABLES / "legacy.tex").write_text("\n".join(lines))
    summary["legacy"] = {"sha256_match": leg["sha256"] == leg["sha256_expected"], "test_ring_claims": comp["test"]["ring"],
                         "gb_raw_ring_ap": gb["ring_ap"], "gb_raw_ring_recall": gb["ring_recall"]}


def summarise_kaggle_nested(summary: dict) -> None:
    nested = load_rows("kaggle_nested")
    shared = load_rows("kaggle")
    if nested.empty or shared.empty:
        return
    rows = []
    for m in nested.method.unique():
        a = nested[nested.method == m].set_index("fold")["ap"]
        b = shared[shared.method == m].set_index("fold")["ap"].reindex(a.index)
        rows.append({"method": m, "nested_ap_mean": a.mean(), "nested_ap_sd": a.std(ddof=1), "shared_ap_mean": b.mean(),
                     **corrected_resampled_comparison(a.to_numpy(), b.to_numpy(), KAGGLE_TEST_TRAIN_RATIO)})
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "kaggle_nested.csv", index=False)
    gnn = shared[shared.method.isin(["SAGE", "RGCN", "HAN", "SimpleHGN", "HGT", "SAGE+G", "HGT+G"])]
    summary["kaggle_nested"] = {"ap": out.set_index("method")[["nested_ap_mean", "shared_ap_mean"]].round(4).to_dict(orient="index"),
                                "best_nested": str(out.loc[out.nested_ap_mean.idxmax(), "method"]),
                                "best_gnn_shared_tuning": float(gnn.groupby("method")["ap"].mean().max())}


def summarise_latency(summary: dict) -> None:
    path = RESULTS_DIR / "latency" / "latency.json"
    if not path.exists():
        return
    r = json.load(open(path))
    w = r["worlds"]
    pc = lambda key, stat: float(np.median([x["per_claim"][key][stat] for x in w]))
    batch = lambda key: float(np.median([x["batch"][key] for x in w]))
    routing = pd.DataFrame([dict(e, world=i) for i, x in enumerate(w) for e in x["batch_routing"]])
    summary["latency"] = {
        "worlds": len(w), "threads": r["provenance"].get("threads"), "peak_rss_mb": r["peak_rss_mb"],
        "gb_p50_ms": pc("gb", "p50_ms"), "gb_p99_ms": pc("gb", "p99_ms"),
        "hgt_p50_ms": pc("hgt_total", "p50_ms"), "hgt_p99_ms": pc("hgt_total", "p99_ms"),
        "hgt_extract_p50_ms": pc("hgt_extract", "p50_ms"), "hgt_forward_p50_ms": pc("hgt_forward", "p50_ms"),
        "gb_throughput_per_s": float(np.median([x["per_claim"]["gb_throughput_per_s"] for x in w])),
        "hgt_throughput_per_s": float(np.median([x["per_claim"]["hgt_throughput_per_s"] for x in w])),
        "subgraph_share_median": float(np.median([x["per_claim"]["subgraph_share_of_graph_median"] for x in w])),
        "subgraph_nodes_median": float(np.median([x["per_claim"]["subgraph_nodes_median"] for x in w])),
        "max_abs_diff_vs_full": float(np.max([x["per_claim"]["max_abs_diff_vs_full_graph"] for x in w])),
        "features_per_period_s": batch("features_per_period_s"), "graph_per_period_s": batch("graph_per_period_s"),
        "gb_score_period_s": batch("gb_score_period_s"), "hgt_score_period_s": batch("hgt_score_period_s"),
        "test_graph_nodes": float(np.median([x["test_graph_nodes"] for x in w])),
        "routing": routing.groupby("escalation")[["escalated_test", "union_subgraph_share", "seconds"]].median()
        .round(4).to_dict(orient="index")}


def hyperparameter_table() -> None:
    """Appendix table of selected configurations (synthetic and Kaggle), straight from the tuning files."""
    def tex(v):
        return "--" if v is None else (f"{v:g}" if isinstance(v, float) else str(v).replace("_", r"\_"))
    lines = [r"\begin{table}[h]",
             r"\caption{Selected hyperparameters (equal budget per method; hybrids inherit their base "
             r"configuration; methods sharing a search space were offered the same candidate configurations). Synthetic: tuned on development worlds; public data: tuned on a separate split. The configurations re-tuned for the low-volume stress test are in the repository (\texttt{results/tuning\_lowvolume}).}\label{tab:hyperparams}",
             r"\scriptsize", r"\begin{tabular}{@{}p{2.2cm}p{5.0cm}p{5.0cm}@{}}", r"\toprule", r"Method & Synthetic benchmark & Public data \\", r"\midrule"]
    syn = json.load(open(RESULTS_DIR / "tuning" / "best_hyperparameters.json"))["best"]
    kpath = RESULTS_DIR / "kaggle" / "tuning" / "best_hyperparameters.json"
    kag = json.load(open(kpath))["best"] if kpath.exists() else {}
    pairs = [("LR", "LR-CA", "LR-P"), ("RF", "RF-CA", "RF-P"), ("GB (C)", "GB-C", None), ("GB (C+A / P)", "GB-CA", "GB-P"),
             ("GB (C+A+G / P+G)", "GB-CAG", "GB-PG"), ("GB (P+G+L)", None, "GB-PGL"), ("MLP", "MLP-C", "MLP-P"),
             ("SAGE", "SAGE", "SAGE"), ("RGCN", "RGCN", "RGCN"), ("HAN", "HAN", "HAN"), ("Simple-HGN", "SimpleHGN", "SimpleHGN"),
             ("HGT", "HGT", "HGT")]
    fmt_hp = lambda d: "; ".join(f"{k.replace('_', ' ')}={tex(v)}" for k, v in sorted(d.items())) if d else "--"
    for name, s_key, k_key in pairs:
        lines.append(f"{name} & {fmt_hp(syn.get(s_key)) if s_key else '--'} & {fmt_hp(kag.get(k_key)) if k_key else '--'} \\\\")
    lines += [r"\botrule", r"\end{tabular}", r"\end{table}"]
    (TABLES / "hyperparameters.tex").write_text("\n".join(lines))


def kaggle_descriptives(summary: dict) -> None:
    """Descriptive label homophily on the provider graph (uses all labels; not used by any model)."""
    from ..ledger import KAGGLE_DIR
    if not (KAGGLE_DIR / "Train-1542865627584.csv").exists():
        return
    from ..realdata.kaggle import load
    d = load()
    y = d.providers["y"].to_numpy().astype(float)
    out = {"n_providers": int(len(y)), "fraud_rate": float(y.mean())}
    for name, B in [("beneficiary", d.B_pb), ("physician", d.B_pf)]:
        C = (B @ B.T).tocsr()
        C.setdiag(0)
        C.eliminate_zeros()
        C.data[:] = 1.0
        deg = np.asarray(C.sum(1)).ravel()
        share = np.divide(C @ y, deg, out=np.full(len(y), np.nan), where=deg > 0)
        out[name] = {
            "share_with_coproviders": float((deg > 0).mean()),
            "median_coproviders": float(np.median(deg)),
            "fraud_share_among_coproviders_of_fraud": float(np.nanmean(share[y == 1])),
            "fraud_share_among_coproviders_of_nonfraud": float(np.nanmean(share[y == 0])),
        }
        # fraud providers' share of claims: are they simply larger?
    claims_per_provider = d.claims.groupby("Provider").size().reindex(d.providers["provider_id"]).to_numpy()
    out["median_claims_fraud"] = float(np.median(claims_per_provider[y == 1]))
    out["median_claims_nonfraud"] = float(np.median(claims_per_provider[y == 0]))
    write_json(TABLES / "kaggle_descriptives.json", out)
    summary["kaggle_descriptives"] = out


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    hyperparameter_table()
    summary = {}
    summarise_synthetic(summary)
    volume_signal(summary)
    summarise_lowvolume(summary)
    summarise_persistent(summary)
    summarise_ablation(summary)
    summarise_kaggle(summary)
    kaggle_bootstrap(summary)
    kaggle_descriptives(summary)
    summarise_cascade(summary)
    summarise_legacy(summary)
    summarise_kaggle_nested(summary)
    summarise_latency(summary)
    write_json(RESULTS_DIR / "summary.json", {"provenance": provenance({"analysis": "analyze.py"}), **summary})
    print(json.dumps(summary, indent=1, default=str))


if __name__ == "__main__":
    main()
