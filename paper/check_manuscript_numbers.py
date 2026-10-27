"""Check that numbers quoted in the manuscript text match the result artifacts.

Every check recomputes a quoted value from results/ (run records, generated tables, summary.json) and, where given,
confirms that the quoted text still appears in fraud_detection_paper.tex or its Online Resource
(supplementary_material.tex). Run from the repository root after analyze.py:

    .venv-research/bin/python paper/check_manuscript_numbers.py

Exit status 1 if any check fails. Written for the HISS revision of 2026-10-25 (see paper/hiss_review/).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
R = Path("results")
TEX = Path("fraud_detection_paper.tex").read_text() + "\n" + Path("supplementary_material.tex").read_text()
checks = []


def check(label, quoted, value, tol=0.0005, in_text=None):
    ok = abs(float(quoted) - float(value)) <= tol
    present = True if in_text is None else (in_text in TEX)
    checks.append((ok and present, label, quoted, round(float(value), 4), "" if present else f"[text '{in_text}' not found]"))


def rows(sub, seeds=range(1, 13)):
    """Run rows; synthetic sets are restricted to the main shifts and, by default, the twelve design seeds."""
    out = []
    for f in sorted((R / sub / "runs").glob("*.json")):
        out += json.load(open(f))["rows"]
    df = pd.DataFrame(out)
    if sub.startswith("synthetic"):
        df = df[df.ring_shift.isin([0.0, 1.0, 2.0]) & df.seed.isin(seeds)]
    return df


syn = rows("synthetic")
one = syn[syn.method == "GB-CAG"]
# Section 6 intro: test-period composition and chance levels
check("test claims per period (mean)", 4943, one.n_test.mean(), 0.5, "4{,}943 claims")
check("ring claims per test period", 384, one.n_ring.mean(), 0.5, "384 belong")
check("isolated claims per test period", 133, one.n_isolated.mean(), 0.5, "133 are isolated")
check("ring AP chance", 0.080, (one.n_ring / (one.n_test - one.n_isolated)).mean(), 0.0006, "0.080 and 0.029")
check("isolated AP chance", 0.029, (one.n_isolated / (one.n_test - one.n_ring)).mean(), 0.0006)
# 6.1 claim-only models at delta = 0
m = syn[syn.ring_shift == 0.0].groupby(["method", "ring_density"]).ring_ap.mean()
check("GB-C min ring AP d0", 0.082, m["GB-C"].min(), 0.0006, "GB-C 0.082--0.084")
check("GB-C max ring AP d0", 0.084, m["GB-C"].max(), 0.0006)
check("MLP-C min d0", 0.083, m["MLP-C"].min(), 0.0006, "MLP-C 0.083--0.089")
check("MLP-C max d0", 0.089, m["MLP-C"].max(), 0.0006)
g = syn[syn.method == "GB-C"].groupby("ring_shift").ring_ap.mean()
check("GB-C ring AP d1 about 0.33", 0.33, g[1.0], 0.01, "about 0.33")
check("GB-C ring AP d2 about 0.81", 0.81, g[2.0], 0.01)
check("GB-CA min d0", 0.886, m["GB-CA"].min(), 0.0006, "0.886--0.919")
check("GB-CA max d0", 0.919, m["GB-CA"].max(), 0.0006)
check("overlap s0.1", 0.283, m["Overlap-score"][0.1], 0.0006, "(0.283)")
check("overlap s0.5", 0.906, m["Overlap-score"][0.5], 0.0006, "0.906 and 0.982")
check("overlap s0.9", 0.982, m["Overlap-score"][0.9], 0.0006)
C = pd.read_csv(R / "tables" / "synthetic_ring_ap_comparisons.csv")
gg = C[(C.reference == "GB-CA") & (C.method == "GB-CAG") & (C.ring_shift == 0.0)].set_index("ring_density")
check("GB-CAG - GB-CA d0 s0.1 +0.005", 0.005, gg.loc[0.1, "mean_diff"], 0.0006, "$+0.005$ ($s=0.1$, n.s.)")
check("GB-CAG - GB-CA d0 s0.1 n.s.", 1, int(gg.loc[0.1, "t_p_holm"] > 0.05), 0)
check("GB-CAG - GB-CA d0 s0.5 +0.074", 0.074, gg.loc[0.5, "mean_diff"], 0.0006, "$+0.074$ ($s=0.5$)")
check("GB-CAG - GB-CA d0 s0.9 +0.104", 0.104, gg.loc[0.9, "mean_diff"], 0.0006, "$+0.104$ ($s=0.9$; both Holm")
check("GB-CAG - GB-CA d0 s>=0.5 Holm < 0.001", 0.001, max(0.001, gg.loc[0.5, "t_p_holm"], gg.loc[0.9, "t_p_holm"]), 0.0)
sg = C[(C.reference == "GB-CA") & (C.method == "SAGE") & (C.ring_shift == 0.0) & (C.ring_density == 0.9)].iloc[0]
check("SAGE - GB-CA d0 s0.9 +0.085", 0.085, sg.mean_diff, 0.0006, "$+0.085$ at $\\delta=0$, $s=0.9$ (Holm $p<0.001$)")
check("SAGE - GB-CA d0 s0.9 Holm < 0.001", 0.001, max(0.001, sg.t_p_holm), 0.0)
pure5 = ["SAGE", "RGCN", "HAN", "SimpleHGN", "HGT"]
cg = C[(C.reference == "GB-CAG") & C.method.isin(pure5)]
up = cg[(cg.mean_diff > 0) & (cg.t_p_holm < 0.05)]
check("one significant pure-GNN advantage (RGCN, d2, s0.1)", 1,
      int(len(up) == 1 and up.iloc[0].method == "RGCN" and up.iloc[0].ring_shift == 2.0 and up.iloc[0].ring_density == 0.1), 0,
      "RGCN at $\\delta=2$, $s=0.1$ ($+0.005$, Holm $p=0.007$)")
check("RGCN d2 s0.1 +0.005", 0.005, up.iloc[0].mean_diff, 0.0006)
check("RGCN d2 s0.1 Holm 0.007", 0.007, up.iloc[0].t_p_holm, 0.0006)
check("pure GNNs significantly worse in 8 pairs", 8, len(cg[(cg.mean_diff < 0) & (cg.t_p_holm < 0.05)]), 0, "worse in eight\nmethod--regime pairs")
d2 = syn[(syn.ring_shift == 2.0) & syn.method.isin(["LR-CA", "RF-CA", "GB-CA", "GB-CAG", "SAGE", "RGCN", "HAN",
                                                    "SimpleHGN", "HGT", "SAGE+AG", "HGT+AG"])]
check("delta=2: all C+A methods exceed 0.98", 0.98, min(0.98, d2.groupby(["method", "ring_density"]).ring_ap.mean().min()), 0.0)
# Cost section
t = pd.read_csv(R / "tables" / "synthetic_timing.csv", index_col=0).train_seconds
for name, q in [("GB-CAG", 3), ("SAGE", 39), ("RGCN", 56), ("HAN", 86), ("HGT", 126), ("SimpleHGN", 144)]:
    check(f"median train s {name}", q, t[name], 0.6)
lat = json.load(open(R / "summary.json"))["latency"]
for key, q, tol in [("gb_p50_ms", 2.1, 0.05), ("gb_p99_ms", 3.1, 0.05), ("hgt_p50_ms", 10.7, 0.05), ("hgt_p99_ms", 14.7, 0.05),
                    ("hgt_extract_p50_ms", 2.8, 0.05), ("hgt_forward_p50_ms", 7.7, 0.05), ("features_per_period_s", 0.05, 0.005),
                    ("graph_per_period_s", 0.01, 0.005), ("gb_score_period_s", 0.05, 0.005), ("hgt_score_period_s", 0.12, 0.005),
                    ("subgraph_nodes_median", 126, 0.6), ("subgraph_share_median", 0.01, 0.005)]:
    check(f"latency {key}", q, lat[key], tol)
check("latency max diff vs full graph <= 1.2e-7", 1.2e-7, lat["max_abs_diff_vs_full"], 0.05e-7)
check("test graph nodes about 11,200", 11200, lat["test_graph_nodes"], 100)
r10 = lat["routing"]["0.1"]
check("routing 10% touches 55% of graph", 0.55, r10["union_subgraph_share"], 0.005)
check("routing 10% union scoring 0.08 s", 0.08, r10["seconds"], 0.005)
# Mechanism ablations text
ab = pd.read_csv(R / "tables" / "ablation_comparisons.csv")
abt = pd.read_csv(R / "tables" / "ablation_ring_ap.csv")
rew = abt[abt.variant == "rewired"].set_index("method").ring_ap_mean
check("rewired graph-aware min 0.757", 0.757, rew[["GB-CAG", "SAGE", "HGT"]].min(), 0.0006, "(0.757--0.770)")
check("rewired graph-aware max 0.770", 0.770, rew[["GB-CAG", "SAGE", "HGT"]].max(), 0.0006)
check("rewired GB-CA 0.748", 0.748, rew["GB-CA"], 0.0006, "rewiring (0.748)")
c = ab.set_index(["method", "variant"])
check("SAGE 1 layer -0.135", -0.135, c.loc[("SAGE", "layers_1"), "mean_diff"], 0.0006, "0.135")
check("HGT 1 layer -0.161", -0.161, c.loc[("HGT", "layers_1"), "mean_diff"], 0.0006, "0.161 (HGT)")
check("SAGE no visits -0.016", -0.016, c.loc[("SAGE", "no_visits"), "mean_diff"], 0.0006)
check("HGT no visits -0.047", -0.047, c.loc[("HGT", "no_visits"), "mean_diff"], 0.0006)
check("HGT no degree -0.379", -0.379, c.loc[("HGT", "no_degree"), "mean_diff"], 0.0006, "$-0.379$, Holm")
check("HGT no degree Holm p 0.003", 0.003, c.loc[("HGT", "no_degree"), "t_p_holm"], 0.0005)
check("SAGE no degree -0.029", -0.029, c.loc[("SAGE", "no_degree"), "mean_diff"], 0.0006)
check("HGT no reverse -0.177", -0.177, c.loc[("HGT", "no_reverse"), "mean_diff"], 0.0006, "loses 0.177 AP")
check("HGT full 0.949", 0.949, abt[(abt.method == "HGT") & (abt.variant == "full")].ring_ap_mean.iloc[0], 0.0006, "from 0.949 to 0.569")
check("HGT no degree 0.569", 0.569, abt[(abt.method == "HGT") & (abt.variant == "no_degree")].ring_ap_mean.iloc[0], 0.0006)
check("leak 1.000", 1.0, abt[(abt.method == "GB-CA") & (abt.variant == "ring_id_leak")].ring_ap_mean.iloc[0], 0.0006)
check("HGT layers_4 +0.002", 0.002, c.loc[("HGT", "layers_4"), "mean_diff"], 0.0006, "four layers $+0.002$")
heads = [c.loc[("HGT", v), "mean_diff"] for v in ["heads_1", "heads_2", "heads_8"]]
check("heads min -0.037", -0.037, min(heads), 0.0006, "$-0.037$ to $+0.013$")
check("heads max +0.013", 0.013, max(heads), 0.0006)
check("no_files -0.219", -0.219, c.loc[("HGT", "no_files"), "mean_diff"], 0.0006, "($-0.219$, Holm $p=0.005$)")
check("no_billed_by ring -0.113", -0.113, c.loc[("HGT", "no_billed_by"), "mean_diff"], 0.0006)
check("no_billed_by iso -0.147", -0.147, c.loc[("HGT", "no_billed_by"), "iso_mean_diff"], 0.0006)
check("claim attrs zeroed ring -0.012", -0.012, c.loc[("HGT", "no_claim_attributes"), "mean_diff"], 0.0006)
check("claim attrs zeroed iso AP 0.036", 0.036,
      abt[(abt.method == "HGT") & (abt.variant == "no_claim_attributes")].isolated_ap_mean.iloc[0], 0.0006)
dg = abt[(abt.method == "HGT") & (abt.variant == "degree_only")].iloc[0]
check("degree only 0.982", 0.982, dg.ring_ap_mean, 0.0006, "$0.982\\pm0.018$")
check("degree only sd 0.018", 0.018, dg.ring_ap_std, 0.0006)
check("HGT sd up to 0.11", 0.11, abt[abt.method == "HGT"].ring_ap_std.max(), 0.005, "up to 0.11")
# Reliance (delta = 0)
rel = pd.read_csv(R / "tables" / "reliance.csv")
rel = rel[rel.ring_shift == 0.0].set_index(["ring_density", "model", "intervention"]).ring_ap
check("GB-CAG A s0.1 -0.80", 0.80, rel[(0.1, "GB-CAG", "A")], 0.006, "A: $-0.80$")
check("GB-CAG G s0.1 -0.13", 0.13, rel[(0.1, "GB-CAG", "G")], 0.006)
check("GB-CAG G s0.5 -0.81", 0.81, rel[(0.5, "GB-CAG", "G")], 0.006)
check("GB-CAG G s0.9 -0.88", 0.88, rel[(0.9, "GB-CAG", "G")], 0.006)
check("GB-CAG A below 0.06 at s>=0.5", 0.06, max(0.06, rel[(0.5, "GB-CAG", "A")], rel[(0.9, "GB-CAG", "A")]), 0.0)
check("closure_norm alone s0.9 -0.47", 0.47, rel[(0.9, "GB-CAG", "G:claim_closure_norm")], 0.006)
deg = [rel[(d, m, "degree_encodings")] for d in [0.1, 0.5, 0.9] for m in ["SAGE", "HGT"]]
check("GNN degree reliance min 0.76", 0.76, min(deg), 0.006, "$-0.76$ to $-0.89$")
check("GNN degree reliance max 0.89", 0.89, max(deg), 0.006)
attr = [rel[(d, m, g)] for d in [0.1, 0.5, 0.9] for m in ["SAGE", "HGT"] for g in ["claim_attributes", "entity_attributes"]]
check("GNN attribute reliance below 0.05", 0.05, max(0.05, max(attr)), 0.0)
# Persistent protocol
pg = pd.read_csv(R / "tables" / "protocol_gap.csv").set_index(["ring_density", "method"])
pr = pd.read_csv(R / "tables" / "protocol_ring_ap.csv")
mlp = pr[(pr.protocol == "persistent") & (pr.method == "MLP-C")].set_index("ring_density")["mean"]
check("MLP-C persistent s0.1 0.692", 0.692, mlp[0.1], 0.0006, "0.692 ($s=0.1$)")
check("MLP-C persistent s0.9 0.591", 0.591, mlp[0.9], 0.0006)
check("HGT gain s0.1 0.056", 0.056, pg.loc[(0.1, "HGT"), "mean_diff"], 0.0006, "0.056 at $s=0.1$")
check("HGT gain CI low 0.029", 0.029, pg.loc[(0.1, "HGT"), "ci_low"], 0.0006)
check("HGT gain CI high 0.084", 0.084, pg.loc[(0.1, "HGT"), "ci_high"], 0.0006)
gca = [pg.loc[(d, "GB-CA"), "mean_diff"] for d in [0.1, 0.9]]
check("GB-CA gain min 0.035", 0.035, min(gca), 0.0006, "0.035--0.063")
check("GB-CA gain max 0.063", 0.063, max(gca), 0.0006)
check("MLP-C gain up to 0.60", 0.60, pg.loc[(0.1, "MLP-C"), "mean_diff"], 0.005)
# Routing text
cm = pd.read_csv(R / "tables" / "cascade_means.csv", index_col=0)
cur = pd.read_csv(R / "tables" / "cascade_curves.csv")
hc = cur[(cur.method == "GB-CAG>HGT") & (cur["mode"] == "rerank")].set_index("escalation")
check("rerank HGT ring AP at 5%", 0.980, hc.loc[0.05, "ring_ap"], 0.0006, "from 0.980 at")
check("rerank HGT ring AP at 100%", 0.968, hc.loc[1.0, "ring_ap"], 0.0006)
check("rerank HGT iso AP at 5%", 0.626, hc.loc[0.05, "isolated_ap"], 0.0006, "from 0.626 to 0.508")
check("rerank HGT iso AP at 100%", 0.508, hc.loc[1.0, "isolated_ap"], 0.0006)
esc = json.load(open(R / "summary.json"))["cascade"]["selected_escalation_counts"]
check("HGT rerank 5% chosen in 30 worlds", 30, esc["GB-CAG>HGT (rerank)"]["0.05"], 0, "30\nof 54")
check("HGT fuse all in 22 worlds", 22, esc["GB-CAG>HGT (fuse)"]["1.0"], 0)
check("SAGE fuse all in 30 worlds", 30, esc["GB-CAG>SAGE (fuse)"]["1.0"], 0)
check("fused HGT escalated 0.55", 0.55, cm.loc["GB-CAG>HGT (fuse)", "escalated_test"], 0.005)
check("fused SAGE escalated 0.70", 0.70, cm.loc["GB-CAG>SAGE (fuse)", "escalated_test"], 0.005)
check("all rings with a claim in budget", 1.0, cm["rings_with_any_claim_in_budget"].min(), 0.0)
check("P@5% min 0.989", 0.989, cm["precision_at_5pct"].min(), 0.0006)
check("P@5% max 0.998", 0.998, cm["precision_at_5pct"].max(), 0.0006)
cc = pd.read_csv(R / "tables" / "cascade_comparisons.csv").set_index(["metric", "method"])
check("fused SAGE ring +0.006", 0.006, cc.loc[("ring_ap", "GB-CAG>SAGE (fuse)"), "mean_diff"], 0.0006)
check("fused SAGE iso +0.023", 0.023, cc.loc[("isolated_ap", "GB-CAG>SAGE (fuse)"), "mean_diff"], 0.0006)
# Public data
kd = json.load(open(R / "tables" / "kaggle_descriptives.json"))
check("providers sharing beneficiaries 99.5%", 0.995, kd["beneficiary"]["share_with_coproviders"], 0.0006)
check("median co-providers 30", 30, kd["beneficiary"]["median_coproviders"], 0.5)
check("fraud share among co-providers of fraud 23.9%", 0.239, kd["beneficiary"]["fraud_share_among_coproviders_of_fraud"], 0.0006)
check("... of non-fraud 25.3%", 0.253, kd["beneficiary"]["fraud_share_among_coproviders_of_nonfraud"], 0.0006)
check("physicians: 19.1%", 0.191, kd["physician"]["fraud_share_among_coproviders_of_fraud"], 0.0006)
check("physicians: 14.2%", 0.142, kd["physician"]["fraud_share_among_coproviders_of_nonfraud"], 0.0006)
check("physicians connect 73%", 0.73, kd["physician"]["share_with_coproviders"], 0.005)
check("physician median 2", 2, kd["physician"]["median_coproviders"], 0.5)
check("median claims fraud 156", 156, kd["median_claims_fraud"], 0.5)
check("median claims non-fraud 27", 27, kd["median_claims_nonfraud"], 0.5)
check("fraud rate 9.4%", 0.094, kd["fraud_rate"], 0.0006)
kag = rows("kaggle")
km = kag.groupby("method")
check("ROC-AUC min 0.927", 0.927, km.roc_auc.mean().min(), 0.0006, "between 0.927 and 0.942")
check("ROC-AUC max 0.942", 0.942, km.roc_auc.mean().max(), 0.0006)
check("recall@10% min 0.59", 0.59, km.recall_at_10pct.mean().min(), 0.005, "between 0.59 and 0.64")
check("recall@10% max 0.64", 0.64, km.recall_at_10pct.mean().max(), 0.005)
check("test providers per rotation 1,082", 1082, kag.n_test.mean(), 0.5, "1{,}082 test providers")
check("fraudulent per rotation about 101", 101, kag.n_pos.mean(), 0.5)
gnn = kag[kag.method.isin(["SAGE", "RGCN", "HAN", "SimpleHGN", "HGT", "SAGE+G", "HGT+G"])].groupby("method").train_seconds.median() / 60
check("GNN median train min 2", 2, gnn.min(), 0.15, "2--15 minutes")
check("GNN median train max 15", 15, gnn.max(), 0.6)
allg = kag[kag.method.isin(["SAGE", "RGCN", "HAN", "SimpleHGN", "HGT", "SAGE+G", "HGT+G"])].train_seconds / 60
check("GNN train range min 1.4", 1.4, allg.min(), 0.05, "range 1.4--24")
check("GNN train range max 24", 24, allg.max(), 0.5)
check("GB-P median train 7 s", 7, kag[kag.method == "GB-P"].train_seconds.median(), 0.6, "against 7~s")



# ---- Numbers written in the HISS revision (pooled seed-level, routing, public-data tests, volume, low volume) ----
P = pd.read_csv(R / "tables" / "synthetic_pooled.csv").set_index(["metric", "reference", "method"])
def pv(metric, ref, m, col="mean_diff"):
    return P.loc[(metric, ref, m), col]
check("SAGE ring diff +0.001", 0.001, pv("ring_ap", "GB-CAG", "SAGE"), 0.0006, "$\\Delta=+0.001$")
check("SAGE ring CI low -0.003", -0.003, pv("ring_ap", "GB-CAG", "SAGE", "ci_low"), 0.0006, "$[-0.003, 0.006]$")
check("SAGE ring CI high 0.006", 0.006, pv("ring_ap", "GB-CAG", "SAGE", "ci_high"), 0.0006)
for m, q in [("RGCN", -0.003), ("SimpleHGN", -0.006), ("HGT", -0.008)]:
    check(f"{m} ring diff", q, pv("ring_ap", "GB-CAG", m), 0.0006)
check("RGCN/SimpleHGN/HGT text", 1, 1, 0, "($-0.003$, $-0.006$ and $-0.008$; Holm $p\\ge0.18$)")
check("RGCN/SimpleHGN/HGT Holm >= 0.18", 0.18, min(0.18, min(pv("ring_ap", "GB-CAG", m, "t_p_holm") for m in ["RGCN", "SimpleHGN", "HGT"])), 0.0)
check("HAN ring diff -0.023", -0.023, pv("ring_ap", "GB-CAG", "HAN"), 0.0006, "($-0.023$, Holm $p=0.002$")
check("HAN Holm p 0.002", 0.002, pv("ring_ap", "GB-CAG", "HAN", "t_p_holm"), 0.0006)
check("max upper CI of pure GNNs <= 0.006", 0.006, max(pv("ring_ap", "GB-CAG", m, "ci_high") for m in ["SAGE", "RGCN", "HAN", "SimpleHGN", "HGT"]), 0.0005, "at most $+0.006$")
check("SAGE+AG +0.010", 0.010, pv("ring_ap", "GB-CAG", "SAGE+AG"), 0.0006, "$+0.010$\n$[0.005, 0.015]$ for SAGE+AG (Holm $p=0.006$)")
check("SAGE+AG CI low 0.005", 0.005, pv("ring_ap", "GB-CAG", "SAGE+AG", "ci_low"), 0.0006)
check("SAGE+AG CI high 0.015", 0.015, pv("ring_ap", "GB-CAG", "SAGE+AG", "ci_high"), 0.0006)
check("SAGE+AG Holm 0.006", 0.006, pv("ring_ap", "GB-CAG", "SAGE+AG", "t_p_holm"), 0.0006)
check("HGT+AG +0.009", 0.009, pv("ring_ap", "GB-CAG", "HGT+AG"), 0.0006, "$+0.009$ $[0.004, 0.015]$\nfor HGT+AG ($p=0.016$)")
check("HGT+AG CI low 0.004", 0.004, pv("ring_ap", "GB-CAG", "HGT+AG", "ci_low"), 0.0006)
check("HGT+AG CI high 0.015", 0.015, pv("ring_ap", "GB-CAG", "HGT+AG", "ci_high"), 0.0006)
check("HGT+AG Holm 0.016", 0.016, pv("ring_ap", "GB-CAG", "HGT+AG", "t_p_holm"), 0.0006)
for m, q in [("GB-CAG", 0.033), ("SAGE", 0.034), ("RGCN", 0.030), ("SimpleHGN", 0.026), ("HGT", 0.024), ("SAGE+AG", 0.043), ("HGT+AG", 0.042), ("HAN", 0.010)]:
    check(f"vs GB-CA ring {m}", q, pv("ring_ap", "GB-CA", m), 0.0006)
check("vs GB-CA text", 1, 1, 0, "(GB-CAG $+0.033$, SAGE\n$+0.034$, RGCN $+0.030$, Simple-HGN $+0.026$, HGT $+0.024$, HAN $+0.010$,\nhybrids $+0.043$ and $+0.042$; Holm $p\\le0.020$")
check("vs GB-CA Holm <= 0.020", 0.020, max(0.020, max(pv("ring_ap", "GB-CA", m, "t_p_holm") for m in ["GB-CAG", "SAGE", "RGCN", "HAN", "SimpleHGN", "HGT", "SAGE+AG", "HGT+AG"])), 0.0)
iso = [pv("isolated_ap", "GB-CAG", m) for m in ["SAGE", "RGCN", "HAN", "SimpleHGN", "HGT"]]
check("iso min deficit 0.093 (Simple-HGN)", -0.093, pv("isolated_ap", "GB-CAG", "SimpleHGN"), 0.0006, "by 0.093 (Simple-HGN) to 0.138\n(HAN)")
check("iso: Simple-HGN smallest deficit", 1, int(max(iso) == pv("isolated_ap", "GB-CAG", "SimpleHGN")), 0)
check("iso max deficit 0.138 (HAN)", -0.138, pv("isolated_ap", "GB-CAG", "HAN"), 0.0006)
check("iso: HAN largest deficit", 1, int(min(iso) == pv("isolated_ap", "GB-CAG", "HAN")), 0)
check("iso Holm < 0.001", 0.001, max(0.001, max(pv("isolated_ap", "GB-CAG", m, "t_p_holm") for m in ["SAGE", "RGCN", "HAN", "SimpleHGN", "HGT"])), 0.0)
check("MLP-C iso -0.025", -0.025, pv("isolated_ap", "GB-CAG", "MLP-C"), 0.0006, "within 0.025 of GB-CAG ($p=0.006$)")
check("MLP-C iso p 0.006", 0.006, pv("isolated_ap", "GB-CAG", "MLP-C", "t_p_holm"), 0.0006)
check("hybrids iso -0.087", -0.087, pv("isolated_ap", "GB-CAG", "SAGE+AG"), 0.0006, "($-0.087$ and\n$-0.113$)")
check("hybrids iso -0.113", -0.113, pv("isolated_ap", "GB-CAG", "HGT+AG"), 0.0006)
ov = [pv("ap", "GB-CAG", m) for m in ["SAGE", "RGCN", "HAN", "SimpleHGN", "HGT"]]
check("overall min 0.008", -0.008, max(ov), 0.0006, "by 0.008--0.030 (all Holm $p\\le0.004$)")
check("overall max 0.030", -0.030, min(ov), 0.0006)
check("overall pure GNN Holm <= 0.004", 0.004, max(0.004, max(pv("ap", "GB-CAG", m, "t_p_holm") for m in ["SAGE", "RGCN", "HAN", "SimpleHGN", "HGT"])), 0.0)
check("overall hybrids p >= 0.37", 0.37, min(0.37, min(pv("ap", "GB-CAG", m, "t_p_holm") for m in ["SAGE+AG", "HGT+AG"])), 0.0, "($p\\ge0.37$)")
hs = C[(C.ring_shift == 0.0) & (C.ring_density == 0.1) & (C.reference == "GB-CAG")].set_index("method")
check("SAGE+AG d0 s0.1 +0.038", 0.038, hs.loc["SAGE+AG", "mean_diff"], 0.0006, "$+0.038$ and $+0.035$, Holm $p=0.007$\nand $0.018$")
check("HGT+AG d0 s0.1 +0.035", 0.035, hs.loc["HGT+AG", "mean_diff"], 0.0006)
check("SAGE+AG d0 s0.1 Holm 0.007", 0.007, hs.loc["SAGE+AG", "t_p_holm"], 0.0006)
check("HGT+AG d0 s0.1 Holm 0.018", 0.018, hs.loc["HGT+AG", "t_p_holm"], 0.0006)
CC = pd.read_csv(R / "tables" / "cascade_comparisons.csv").set_index(["metric", "method"])
check("rerank HGT CI low -0.006", -0.006, CC.loc[("ap", "GB-CAG>HGT (rerank)"), "ci_low"], 0.0006, "$[-0.006, 0.003]$")
check("rerank HGT CI high 0.003", 0.003, CC.loc[("ap", "GB-CAG>HGT (rerank)"), "ci_high"], 0.0006)
check("fused HGT Holm 0.60", 0.60, CC.loc[("ap", "GB-CAG>HGT (fuse)"), "t_p_holm"], 0.005, "Holm $p=0.60$")
check("fused SAGE +0.006", 0.006, CC.loc[("ap", "GB-CAG>SAGE (fuse)"), "mean_diff"], 0.0006)
check("fused SAGE CI low 0.003", 0.003, CC.loc[("ap", "GB-CAG>SAGE (fuse)"), "ci_low"], 0.0006, "$[0.003, 0.010]$")
check("fused SAGE CI high 0.010", 0.010, CC.loc[("ap", "GB-CAG>SAGE (fuse)"), "ci_high"], 0.0006)
check("fused SAGE Holm 0.042", 0.042, CC.loc[("ap", "GB-CAG>SAGE (fuse)"), "t_p_holm"], 0.0006, "Holm $p=0.042$")
K = pd.read_csv(R / "tables" / "kaggle_comparisons.csv").set_index(["method", "reference"])
check("GB-PG corrected CI low", -0.042, K.loc[("GB-PG", "GB-P"), "ci_low"], 0.0006, "$[-0.042, 0.023]$")
check("GB-PG corrected CI high", 0.023, K.loc[("GB-PG", "GB-P"), "ci_high"], 0.0006)
check("GB-PGL corrected CI low", -0.030, K.loc[("GB-PGL", "GB-P"), "ci_low"], 0.0006, "$[-0.030, 0.016]$")
check("GB-PGL corrected CI high", 0.016, K.loc[("GB-PGL", "GB-P"), "ci_high"], 0.0006)
check("smallest corrected Holm 0.13", 0.13, K.t_p_holm.min(), 0.005, "$p=0.13$")
check("naive Holm SAGE+G 0.021", 0.021, K.loc[("SAGE+G", "GB-P"), "naive_t_p_holm"], 0.0006, "Holm $p=0.021$")
B = pd.read_csv(R / "tables" / "kaggle_bootstrap.csv").set_index("method")
check("boot RGCN low", -0.108, B.loc["RGCN", "boot_ci_low"], 0.0006, "$[-0.108, -0.055]$")
check("boot RGCN high", -0.055, B.loc["RGCN", "boot_ci_high"], 0.0006)
check("boot HGT low", -0.048, B.loc["HGT", "boot_ci_low"], 0.0006, "$[-0.048, -0.006]$")
check("boot HGT high", -0.006, B.loc["HGT", "boot_ci_high"], 0.0006)
check("all GNN/hybrid/GB-PG boot highs below 0", 1, int(B.loc[["SAGE", "RGCN", "HAN", "SimpleHGN", "HGT", "SAGE+G", "HGT+G", "GB-PG"], "boot_ci_high"].max() < 0), 0)
N = pd.read_csv(R / "tables" / "kaggle_nested.csv").set_index("method")
TAB = ["LR-P", "RF-P", "GB-P", "GB-PG", "GB-PGL", "MLP-P"]
GNNH = ["SAGE", "RGCN", "HAN", "SimpleHGN", "HGT", "SAGE+G", "HGT+G"]
check("nested: tabular/MLP max |change| 0.010", 0.010, N.loc[TAB, "mean_diff"].abs().max(), 0.0006, "at most 0.010 for\nthe tabular models and the MLP")
check("nested: GNN change min -0.008 (HGT)", -0.008, N.loc["HGT", "mean_diff"], 0.0006, "$-0.008$ to $+0.036$ for the GNNs")
check("nested: HGT is the minimum", 1, int(N.loc[GNNH, "mean_diff"].idxmin() == "HGT"), 0)
check("nested: GNN change max +0.036 (RGCN)", 0.036, N.loc["RGCN", "mean_diff"], 0.0006)
check("nested: RGCN is the maximum", 1, int(N.loc[GNNH, "mean_diff"].idxmax() == "RGCN"), 0)
check("nested: no change significant", 0.05, min(0.05, N.t_p.dropna().min()), 0.0, "none significantly")
check("nested: SAGE same choice in every rotation", 0, N.loc["SAGE", "mean_diff"], 0, "SAGE chose its shared configuration in every\nrotation")
NV = pd.read_csv(R / "tables" / "kaggle_nested_vs_gbp.csv").set_index("method")
check("nested GB-P 0.706 is the best nested method", 1, int(N.nested_ap_mean.idxmax() == "GB-P"), 0, "GB-P (0.706) remained the best")
check("nested GB-P AP 0.706", 0.706, N.loc["GB-P", "nested_ap_mean"], 0.0006)
check("nested: GNN/hybrid gap min 0.023", -0.023, NV.loc[GNNH, "mean_diff"].max(), 0.0006, "0.023--0.053 below it")
check("nested: GNN/hybrid gap max 0.053", -0.053, NV.loc[GNNH, "mean_diff"].min(), 0.0006)
check("nested: smallest Holm p 0.051", 0.051, NV.t_p_holm.min(), 0.0006, "smallest Holm $p=0.051$")
KR = pd.read_csv(R / "tables" / "kaggle_repeated_comparisons.csv").set_index("method")
KM = pd.read_csv(R / "tables" / "kaggle_repeated_metrics.csv", header=[0, 1], index_col=0)
check("repeated: 25 rotations per method", 25, KR.n.min(), 0)
check("repeated: GB-P highest mean AP 0.693", 0.693, KM[("ap", "mean")]["GB-P"], 0.0006, "highest mean AP (0.693)")
check("repeated: GB-P is the best", 1, int(KM[("ap", "mean")].idxmax() == "GB-P"), 0)
check("repeated: GB-PG -0.007", -0.007, KR.loc["GB-PG", "mean_diff"], 0.0006, "$-0.007$ $[-0.019, 0.006]$ (GB-PG)")
check("repeated: GB-PG CI low", -0.019, KR.loc["GB-PG", "ci_low"], 0.0006)
check("repeated: GB-PG CI high", 0.006, KR.loc["GB-PG", "ci_high"], 0.0006)
check("repeated: GB-PGL -0.006", -0.006, KR.loc["GB-PGL", "mean_diff"], 0.0006, "$-0.006$\n$[-0.019, 0.007]$ (GB-PGL)")
check("repeated: GB-PGL CI low", -0.019, KR.loc["GB-PGL", "ci_low"], 0.0006)
check("repeated: GB-PGL CI high", 0.007, KR.loc["GB-PGL", "ci_high"], 0.0006, "at most $+0.006$ to\n$+0.007$ (upper 95\\% limits)")
# Third round (E-7): RGCN, Simple-HGN and the hybrids added; Holm over all eleven comparisons with GB-P.
GNN5K = ["SAGE", "RGCN", "HAN", "SimpleHGN", "HGT"]
check("repeated: twelve comparisons", 12, len(KR), 0, "all twelve differences significant")
check("repeated: GNN gap min 0.025", -0.025, KR.loc[GNN5K, "mean_diff"].max(), 0.0006, "the five GNNs were 0.025--0.092 and the hybrids")
check("repeated: GNN gap max 0.092", -0.092, KR.loc[GNN5K, "mean_diff"].min(), 0.0006, "the GNNs are 0.025--0.092 lower")
check("repeated: hybrid gap min 0.039", -0.039, KR.loc[["SAGE+G", "HGT+G"], "mean_diff"].max(), 0.0006, "0.039--0.063 below it")
check("repeated: hybrid gap max 0.063", -0.063, KR.loc[["SAGE+G", "HGT+G"], "mean_diff"].min(), 0.0006, "hybrids\n0.039--0.063 lower")
check("repeated: only RGCN significant", 1, int(list(KR.index[KR.t_p_holm < 0.05]) == ["RGCN"]), 0, "Only RGCN's deficit was significant")
check("repeated: RGCN Holm p 0.008", 0.008, KR.loc["RGCN", "t_p_holm"], 0.0006, "(Holm $p=0.008$; next smallest $0.052$,")
check("repeated: RGCN -0.092", -0.092, KR.loc["RGCN", "mean_diff"], 0.0006, "($-0.092$, Holm $p=0.008$)")
check("repeated: next smallest Holm p 0.052 (SAGE+G)", 0.052, KR.drop("RGCN").t_p_holm.min(), 0.0006, "adjusted $p$ is 0.052 (SAGE+G)")
check("repeated: SAGE+G has it", 1, int(KR.drop("RGCN").t_p_holm.idxmin() == "SAGE+G"), 0)
from research.metrics import holm
check("repeated: naive test + Holm would call all twelve significant", 1, int(max(holm(KR.naive_t_p.tolist())) < 0.05), 0)
check("repeated: all naive p < 0.01", 0.01, max(0.01, KR.naive_t_p.max()), 0.0, "(all unadjusted\n$p<0.01$)")
V = pd.read_csv(R / "tables" / "volume_signal.csv").set_index("volume")
check("default ring patient 9.3", 9.3, V.loc[16.0, "ring_patient_claims"], 0.05, "9.3 claims")
check("other patients 1.6", 1.6, V.loc[16.0, "other_patient_claims"], 0.05)
check("p99 about 5", 5, V.loc[16.0, "other_patient_claims_p99"], 0.2)
check("ring doctor 38", 38, V.loc[16.0, "ring_doctor_claims"], 0.5)
check("other doctor 16", 16, V.loc[16.0, "other_doctor_claims"], 0.5)
check("low ring patient 3.7", 3.7, V.loc[4.0, "ring_patient_claims"], 0.05, "3.7 claims")
check("low p95 3.8 (about the 95th percentile)", 3.8, V.loc[4.0, "other_patient_claims_p95"], 0.1)
check("low ring doctor 20", 20, V.loc[4.0, "ring_doctor_claims"], 0.5)
check("low prevalence 2.1%", 0.021, V.loc[4.0, "ring_prevalence"], 0.0006, "2.1\\%")
check("ratio 5.7", 5.7, V.loc[16.0, "ring_patient_claims"] / V.loc[16.0, "other_patient_claims"], 0.05, "5.7 times")
L = pd.read_csv(R / "tables" / "lowvolume_ring_ap.csv")
fz = L[L.setting == "frozen"].set_index(["ring_density", "method"]).ring_ap_mean
check("low GB-CA min 0.049", 0.049, min(fz[(d, "GB-CA")] for d in [0.1, 0.5, 0.9]), 0.0006, "0.049--0.055")
check("low GB-CA max 0.055", 0.055, max(fz[(d, "GB-CA")] for d in [0.1, 0.5, 0.9]), 0.0006)
check("low GB-CAG s0.5 0.409", 0.409, fz[(0.5, "GB-CAG")], 0.0006, "0.409 and 0.570")
check("low GB-CAG s0.9 0.570", 0.570, fz[(0.9, "GB-CAG")], 0.0006)
check("low GB-CAG s0.1 0.066", 0.066, fz[(0.1, "GB-CAG")], 0.0006, "0.066")
pure = [fz[(d, m)] for d in [0.5, 0.9] for m in ["SAGE", "RGCN", "HAN", "SimpleHGN", "HGT"]]
check("low pure GNN min s>=0.5 0.058", 0.058, min(pure), 0.0006, "0.058--0.151")
check("low pure GNN max s>=0.5 0.151", 0.151, max(pure), 0.0006)
check("overlap low s0.5 0.194", 0.194, fz[(0.5, "Overlap-score")], 0.0006, "(0.194 and 0.281)")
check("overlap low s0.9 0.281", 0.281, fz[(0.9, "Overlap-score")], 0.0006)
check("overlap beats every pure GNN at s>=0.5", 1, int(all(fz[(d, "Overlap-score")] > max(fz[(d, m)] for m in ["SAGE", "RGCN", "HAN", "SimpleHGN", "HGT"]) for d in [0.5, 0.9])), 0)
LC = pd.read_csv(R / "tables" / "lowvolume_comparisons.csv")
lf = LC[(LC.setting == "frozen") & (LC.metric == "ring_ap")]
pool = lf[(lf.reference == "GB-CAG") & (lf.ring_density == "pooled")].set_index("method")
check("low GNN vs GB-CAG min gap 0.23", -0.23, max(pool.loc[m, "mean_diff"] for m in ["SAGE", "RGCN", "HAN", "SimpleHGN", "HGT"]), 0.005, "0.23--0.29 below")
check("low GNN vs GB-CAG max gap 0.29", -0.29, min(pool.loc[m, "mean_diff"] for m in ["SAGE", "RGCN", "HAN", "SimpleHGN", "HGT"]), 0.005)
check("low GNN vs GB-CAG Holm <= 0.002", 0.002, max(0.002, max(float(pool.loc[m, "t_p_holm"]) for m in ["SAGE", "RGCN", "HAN", "SimpleHGN", "HGT"])), 0.0)
check("low SAGE+AG +0.001", 0.001, pool.loc["SAGE+AG", "mean_diff"], 0.0006, "$+0.001$, 95\\% CI $[-0.059, 0.061]$")
check("low SAGE+AG CI low", -0.059, pool.loc["SAGE+AG", "ci_low"], 0.0006)
check("low SAGE+AG CI high", 0.061, pool.loc["SAGE+AG", "ci_high"], 0.0006)
check("low HGT+AG -0.041", -0.041, pool.loc["HGT+AG", "mean_diff"], 0.0006, "$-0.041$")
check("low HGT+AG CI low", -0.098, pool.loc["HGT+AG", "ci_low"], 0.0006, "$[-0.098, 0.015]$")
check("low HGT+AG CI high", 0.015, pool.loc["HGT+AG", "ci_high"], 0.0006)
pa = lf[(lf.reference == "GB-CA") & (lf.ring_density == "pooled")].set_index("method")
gains = [pa.loc[m, "mean_diff"] for m in ["SAGE", "RGCN", "HAN", "SimpleHGN", "HGT"]]
check("low GNN gain over GB-CA min 0.01", 0.01, min(gains), 0.005, "only 0.01--0.07 above GB-CA")
check("low GNN gain over GB-CA max 0.07", 0.07, max(gains), 0.005)
cells = lf[(lf.reference == "GB-CA") & (lf.method == "GB-CAG") & (lf.ring_density != "pooled")].set_index("ring_density")
check("low GB-CAG - GB-CA s0.5 0.36", 0.36, cells.loc["0.5", "mean_diff"], 0.005, "0.36 and 0.51 above GB-CA")
check("low GB-CAG - GB-CA s0.9 0.51", 0.51, cells.loc["0.9", "mean_diff"], 0.005)
check("low GB-CAG - GB-CA Holm <= 0.001", 0.001, max(0.001, max(float(cells.loc[d, "t_p_holm"]) for d in ["0.5", "0.9"])), 0.0)
better = lf[(lf.reference == "GB-CAG") & (lf.ring_density != "pooled") & (lf.mean_diff > 0) & (lf.t_p_holm < 0.05)]
check("no method significantly above GB-CAG at any density (frozen)", 0, len(better), 0)
li = LC[(LC.setting == "frozen") & (LC.metric == "isolated_ap") & (LC.reference == "GB-CAG") & (LC.ring_density == "pooled")].set_index("method")
igap = [li.loc[m, "mean_diff"] for m in ["SAGE", "RGCN", "HAN", "SimpleHGN", "HGT"]]
check("low iso gap min 0.03", -0.03, max(igap), 0.006, "by\n0.03--0.08")
check("low iso gap max 0.08", -0.08, min(igap), 0.006)
check("low iso not significant", 0.05, min(0.05, min(float(li.loc[m, "t_p_holm"]) for m in ["SAGE", "RGCN", "HAN", "SimpleHGN", "HGT"])), 0.0)

# Budgeted recall (Section 6 introduction)
rec = pd.read_csv(R / "tables" / "synthetic_recall_at_5pct.csv")
graphish = ["LR-CA", "RF-CA", "GB-CA", "GB-CAG", "SAGE", "RGCN", "HAN", "SimpleHGN", "HGT", "SAGE+AG", "HGT+AG"]
check("budget recall ceiling 0.479", 0.479, (0.05 * one.n_test / one.n_pos).clip(upper=1).mean(), 0.0006, "cannot exceed 0.479")
check("fraud prevalence 10.5%", 0.105, (one.n_pos / one.n_test).mean(), 0.0006, "10.5\\% of test")
check("budget recall min 0.43", 0.43, rec[rec.method.isin(graphish)]["mean"].min(), 0.005, "0.43--0.48 in every regime")
check("budget recall max 0.48", 0.48, rec[rec.method.isin(graphish)]["mean"].max(), 0.005)

# Low volume, re-tuned hyperparameters (Section 6.6, second paragraph; abstract; discussion)
rt = L[L.setting == "retuned"].set_index(["ring_density", "method"]).ring_ap_mean
check("retuned GB-CAG s0.5 0.445", 0.445, rt[(0.5, "GB-CAG")], 0.0006, "0.445 and 0.600")
check("retuned GB-CAG s0.9 0.600", 0.600, rt[(0.9, "GB-CAG")], 0.0006)
pr = [rt[(d, m)] for d in [0.5, 0.9] for m in ["SAGE", "RGCN", "HAN", "SimpleHGN", "HGT"]]
check("retuned pure GNN min s>=0.5 0.058", 0.058, min(pr), 0.0006, "between 0.058 and 0.162")
check("retuned pure GNN max s>=0.5 0.162", 0.162, max(pr), 0.0006)
for m in ["HAN", "RGCN", "SimpleHGN"]:
    check(f"{m} unchanged configuration reproduces frozen results", 0, max(abs(rt[(d, m)] - fz[(d, m)]) for d in [0.1, 0.5, 0.9]), 1e-9)
lr = LC[(LC.setting == "retuned") & (LC.metric == "ring_ap")]
prt = lr[(lr.reference == "GB-CAG") & (lr.ring_density == "pooled")].set_index("method")
gaps = [prt.loc[m, "mean_diff"] for m in ["SAGE", "RGCN", "HAN", "SimpleHGN", "HGT"]]
check("retuned gap min 0.26", -0.26, max(gaps), 0.005, "0.26--0.31 below")
check("retuned gap max 0.31", -0.31, min(gaps), 0.005)
check("retuned gap Holm <= 0.001", 0.001, max(0.001, max(float(prt.loc[m, "t_p_holm"]) for m in ["SAGE", "RGCN", "HAN", "SimpleHGN", "HGT"])), 0.0)
check("retuned SAGE+AG +0.009", 0.009, prt.loc["SAGE+AG", "mean_diff"], 0.0006, "$+0.009$")
check("retuned SAGE+AG CI low", -0.048, prt.loc["SAGE+AG", "ci_low"], 0.0006, "$[-0.048, 0.065]$")
check("retuned SAGE+AG CI high", 0.065, prt.loc["SAGE+AG", "ci_high"], 0.0006)
rc = lr[(lr.reference == "GB-CA") & (lr.method == "GB-CAG") & (lr.ring_density != "pooled")].set_index("ring_density")
check("retuned GB-CAG - GB-CA s0.5 0.39", 0.39, rc.loc["0.5", "mean_diff"], 0.005, "0.39\nand 0.55 above GB-CA")
check("retuned GB-CAG - GB-CA s0.9 0.55", 0.55, rc.loc["0.9", "mean_diff"], 0.005)
sig = lr[(lr.reference == "GB-CAG") & (lr.ring_density != "pooled") & (lr.mean_diff > 0) & (lr.t_p_holm < 0.05)]
check("retuned: exactly one significant advantage over GB-CAG (SAGE+AG, s=0.1)", 1,
      int(len(sig) == 1 and sig.iloc[0].method == "SAGE+AG" and sig.iloc[0].ring_density == "0.1"), 0)
check("retuned SAGE+AG s0.1 Holm 0.035", 0.035, sig.iloc[0].t_p_holm, 0.0006, "Holm $p=0.035$")
check("retuned SAGE+AG s0.1 AP 0.129", 0.129, rt[(0.1, "SAGE+AG")], 0.0006, "0.129 against 0.078")
check("retuned GB-CAG s0.1 AP 0.078", 0.078, rt[(0.1, "GB-CAG")], 0.0006)
check("discussion: overlap beats every pure GNN after re-tuning too", 1,
      int(all(fz[(d, "Overlap-score")] > max(rt[(d, m)] for m in ["SAGE", "RGCN", "HAN", "SimpleHGN", "HGT"]) for d in [0.5, 0.9])), 0)

# Ring-volume dose-response (Section 6.6, last paragraph; Online Resource S3; seeds 1-6, main hyperparameters)
D = pd.read_csv(R / "tables" / "dose_response_ring_ap.csv").set_index(["volume", "ring_density", "method"]).ring_ap_mean
dens = [0.1, 0.5, 0.9]
def dgain(v, d=None):
    ds = dens if d is None else [d]
    return np.mean([D[(v, x, "GB-CAG")] - D[(v, x, "GB-CA")] for x in ds])
check("dose: structure gain 0.06 at 16", 0.06, dgain(16.0), 0.005, "graph features add 0.06 AP averaged")
check("dose: structure gain 0.42 at 8", 0.42, dgain(8.0), 0.005, "features add 0.42 (0.51 and 0.61")
check("dose: structure gain s0.5 at 8 0.51", 0.51, dgain(8.0, 0.5), 0.005)
check("dose: structure gain s0.9 at 8 0.61", 0.61, dgain(8.0, 0.9), 0.005)
check("dose: structure gain 0.30 at 4", 0.30, dgain(4.0), 0.005, "4 they add 0.30")
check("dose: no method far above chance at 2 (all < 0.05)", 0.05, max(0.05, max(D[(2.0, d, m)] for d in dens for m in D[(2.0, 0.1)].index)), 0.0)
check("dose: aggregates at 8 min 0.31", 0.31, min(D[(8.0, d, "GB-CA")] for d in dens), 0.005, "aggregates fall to 0.31--0.42")
check("dose: aggregates at 8 max 0.42", 0.42, max(D[(8.0, d, "GB-CA")] for d in dens), 0.005)
def pure_gaps(v):
    return [np.mean([D[(v, d, "GB-CAG")] - D[(v, d, m)] for d in dens]) for m in ["SAGE", "RGCN", "HAN", "SimpleHGN", "HGT"]]
for v, lo, hi in [(16.0, 0.01, 0.06), (8.0, 0.10, 0.29), (4.0, 0.23, 0.29)]:
    check(f"dose: pure GNN gap min at {v:g}", lo, min(pure_gaps(v)), 0.005)
    check(f"dose: pure GNN gap max at {v:g}", hi, max(pure_gaps(v)), 0.005)
check("dose: pure GNN gap text", 1, 1, 0, "by 0.01--0.06 AP at 16, 0.10--0.29 at 8 and 0.23--0.29 at 4 claims\nper month")
hyb = [abs(np.mean([D[(v, d, m)] - D[(v, d, "GB-CAG")] for d in dens])) for v in [16.0, 8.0, 4.0, 2.0] for m in ["SAGE+AG", "HGT+AG"]]
check("dose: hybrids within 0.05 of GB-CAG at every volume", 0.05, max(0.05, max(hyb)), 0.0, "hybrids stay within 0.05 of it at every volume")
for v, q in [(8.0, 5.4), (4.0, 3.7), (2.0, 3.0)]:
    check(f"dose: ring patient claims at {v:g}", q, V.loc[v, "ring_patient_claims"], 0.05)
for v, q in [(8.0, 26), (4.0, 20), (2.0, 18)]:
    check(f"dose: ring doctor claims at {v:g}", q, V.loc[v, "ring_doctor_claims"], 0.5)
check("dose: p99 of other patients 5.2", 5.2, V.loc[16.0, "other_patient_claims_p99"], 0.05, "5.2), and a ring doctor 38, 26, 20 and 18")
check("dose: volume 8 about the p99 of other patients", 5.2, V.loc[8.0, "ring_patient_claims"], 0.25, "about as many claims as the\n99th percentile")
check("dose: prevalence at 2 is 1.1%", 0.011, V.loc[2.0, "ring_prevalence"], 0.0006, "1.1\\% of the non-isolated claims")
check("dose: chance at 8 0.042", 0.042, V.loc[8.0, "ring_prevalence"], 0.0006, "0.042 at 8")

# Map table (Discussion) rows that are not checked above
check("map: volume 8 sparse gain +0.13", 0.13, dgain(8.0, 0.1), 0.005, "& $+0.13$ &")
v8 = [D[(8.0, 0.1, m)] for m in ["GB-CAG", "SAGE+AG", "HGT+AG", "SAGE", "RGCN", "SimpleHGN", "HGT"]]
check("map: volume 8 sparse methods min 0.54", 0.54, min(v8), 0.005, "other than HAN alike (0.54--0.66)")
check("map: volume 8 sparse methods max 0.66", 0.66, max(v8), 0.005)
check("map: HAN below the others at volume 8 sparse", 1, int(D[(8.0, 0.1, "HAN")] < min(v8) - 0.05), 0)
dense_gain = [D[(8.0, d, "GB-CAG")] - D[(8.0, d, "GB-CA")] for d in [0.5, 0.9]] + \
             [fz[(d, "GB-CAG")] - fz[(d, "GB-CA")] for d in [0.5, 0.9]] + [rt[(d, "GB-CAG")] - rt[(d, "GB-CA")] for d in [0.5, 0.9]]
check("map/discussion: dense structure gain at volume 4-8 min 0.36", 0.36, min(dense_gain), 0.005, "$+0.36$ to $+0.61$")
check("map/discussion: dense structure gain at volume 4-8 max 0.61", 0.61, max(dense_gain), 0.005, "0.36--0.61 at reduced volume")
cells48 = [(D[(8.0, d, "GB-CAG")], D[(8.0, d, m)]) for d in [0.5, 0.9] for m in ["SAGE", "RGCN", "HAN", "SimpleHGN", "HGT"]] + \
          [(s[(d, "GB-CAG")], s[(d, m)]) for s in (fz, rt) for d in [0.5, 0.9] for m in ["SAGE", "RGCN", "HAN", "SimpleHGN", "HGT"]]
gap48 = [a - b for a, b in cells48]
check("abstract/map: pure GNNs trail by 0.17 (min, volume 4-8, s>=0.5)", 0.17, min(gap48), 0.005, "pure GNNs 0.17--0.54 lower")
check("abstract/map: pure GNNs trail by 0.54 (max)", 0.54, max(gap48), 0.005, "pure GNNs trailed them by 0.17--0.54")
low_sparse = [fz[(0.1, "GB-CAG")] - fz[(0.1, "GB-CA")], rt[(0.1, "GB-CAG")] - rt[(0.1, "GB-CA")]]
check("map: volume 4 sparse gain at most +0.03", 0.03, max(low_sparse), 0.005, "At most $+0.03$")
check("map: default sparse +0.005 / dense 0.07-0.10", 0.07, gg.loc[0.5, "mean_diff"], 0.005, "graph information added\n0.07--0.10 AP")
check("map: dense default max 0.10", 0.10, gg.loc[0.9, "mean_diff"], 0.005)
check("map: public GB-PG -0.007 over 25 rotations", -0.007, KR.loc["GB-PG", "mean_diff"], 0.0006, "None (GB-PG $-0.007$, 25 rotations)")
check("public GB-PG -0.009 (five rotations)", -0.009, K.loc[("GB-PG", "GB-P"), "mean_diff"], 0.0006, "(GB-PG, $-0.009$,")

# Abstract (synthetic part)
m0 = syn[syn.ring_shift == 0.0].groupby(["method", "ring_density"]).ring_ap.mean()
check("abstract: aggregates 0.89 (GB-CA min, d0)", 0.89, m0["GB-CA"].min(), 0.005, "(average\nprecision 0.89--0.92)")
check("abstract: aggregates 0.92 (GB-CA max, d0)", 0.92, m0["GB-CA"].max(), 0.005)
check("abstract: graph features add up to 0.10", 0.10, gg.loc[0.9, "mean_diff"], 0.005, "graph features added up to 0.10 for dense rings")
agg48 = [D[(8.0, d, "GB-CA")] for d in dens] + [fz[(d, "GB-CA")] for d in dens] + [rt[(d, "GB-CA")] for d in dens]
check("abstract: aggregates at volume 4-8 min 0.05", 0.05, min(agg48), 0.005, "aggregates fell to 0.05--0.42")
check("abstract: aggregates at volume 4-8 max 0.42", 0.42, max(agg48), 0.005)
gf48 = [D[(8.0, d, "GB-CAG")] for d in [0.5, 0.9]] + [s[(d, "GB-CAG")] for s in (fz, rt) for d in [0.5, 0.9]]
check("abstract: graph features at volume 4-8 min 0.41", 0.41, min(gf48), 0.005, "features reached 0.41--0.92")
check("abstract: graph features at volume 4-8 max 0.92", 0.92, max(gf48), 0.005)
check("abstract: upper limits <= 0.006", 0.006, max(pv("ring_ap", "GB-CAG", m, "ci_high") for m in ["SAGE", "RGCN", "HAN", "SimpleHGN", "HGT"]), 0.0005, "$\\le$0.006)")
check("abstract: hybrids gained 0.01", 0.01, np.mean([pv("ring_ap", "GB-CAG", m) for m in ["SAGE+AG", "HGT+AG"]]), 0.001, "hybrids of both gained 0.01")
check("abstract: isolated 0.09-0.14 worse (min)", 0.09, -max(iso), 0.005, "GNNs were 0.09--0.14 worse")
check("abstract: isolated 0.09-0.14 worse (max)", 0.14, -min(iso), 0.005)
check("abstract: recurring members inflate by up to 0.60", 0.60, pg.loc[(0.1, "MLP-C"), "mean_diff"], 0.005, "inflated performance by up to 0.60")

# Pre-declared confirmatory test (Section 6.6; Online Resource S4)
H = json.load(open(R / "tables" / "hybrid_confirmation.json"))
check("confirm: fresh seeds 7-18", 1, int(H["seeds"] == list(range(7, 19))), 0, "twelve fresh seeds (7--18)")
check("confirm: +0.024", 0.024, H["mean_diff"], 0.0006, "$+0.024$ $[0.005, 0.043]$ ($p=0.016$; nine of twelve")
check("confirm: CI low 0.005", 0.005, H["ci_low"], 0.0006)
check("confirm: CI high 0.043", 0.043, H["ci_high"], 0.0006)
check("confirm: p 0.016", 0.016, H["t_p"], 0.0006)
check("confirm: 9 of 12 seeds positive", 9, H["n_seeds_positive"], 0)
check("confirm: SAGE+AG 0.129", 0.129, H["sage_ag_mean"], 0.0006, "(0.129 and 0.104 against 0.021)")
check("confirm: GB-CAG 0.104", 0.104, H["gbcag_mean"], 0.0006)
check("confirm: chance 0.021", 0.021, H["ring_chance"], 0.0006)
check("confirm: decision rule met", 1, int(H["confirmed"]), 0)
check("confirm: about half the exploratory estimate", 0.5, H["mean_diff"] / (rt[(0.1, "SAGE+AG")] - rt[(0.1, "GB-CAG")]), 0.05, "about half the exploratory estimate")
check("confirm: Wilcoxon p 0.009", 0.009, H["wilcoxon_p"], 0.0006, "Wilcoxon")

# ---- Third round (paper/hiss_review/analysis_plan_round3.md): E-2, E-3, E-4 ----
# E-2: delta = 1 and 2 at ring volume 4 (Online Resource S5; Section 6.6; map table)
from research.experiments.analyze import SYN_ORDER as SYN
LS = pd.read_csv(R / "tables" / "lowvolume_shift_comparisons.csv")
LS["ring_density"] = LS["ring_density"].astype(str)
def ls(vol, shift, dens, method, ref="GB-CAG", metric="ring_ap", col="mean_diff"):
    r = LS[(LS.volume == vol) & (LS.ring_shift == shift) & (LS.ring_density == str(dens)) & (LS.method == method)
           & (LS.reference == ref) & (LS.metric == metric)]
    assert len(r) == 1, (vol, shift, dens, method, ref, metric)
    return r.iloc[0][col]
LT = pd.read_csv(R / "tables" / "lowvolume_shift_ring_ap.csv").set_index(["ring_shift", "ring_density", "method"])["ring_ap_mean"]
check("E-2: worlds 36", 36, len(rows("synthetic_lowvolume_shift", range(1, 7)).groupby(["ring_shift", "ring_density", "seed"])), 0)
check("E-2: structure gain d1 s0.5 +0.288", 0.288, ls(4.0, 1.0, 0.5, "GB-CAG", "GB-CA"), 0.0006, "$+0.288$ and $+0.301$")
check("E-2: structure gain d1 s0.9 +0.301", 0.301, ls(4.0, 1.0, 0.9, "GB-CAG", "GB-CA"), 0.0006)
check("E-2: structure gain d1 dense Holm < 0.01", 1, int(max(ls(4.0, 1.0, d, "GB-CAG", "GB-CA", col="t_p_holm") for d in [0.5, 0.9]) < 0.01), 0, "(Holm $p<0.01$), against")
check("E-2: default-volume d1 gains +0.039", 0.039, ls(16.0, 1.0, 0.5, "GB-CAG", "GB-CA"), 0.0006, "$+0.039$ and $+0.048$ at default volume")
check("E-2: default-volume d1 gains +0.048", 0.048, ls(16.0, 1.0, 0.9, "GB-CAG", "GB-CA"), 0.0006)
check("E-2: structure gain d1 s0.1 +0.017 n.s.", 0.017, ls(4.0, 1.0, 0.1, "GB-CAG", "GB-CA"), 0.0006, "$+0.017$ at $s=0.1$ (not significant)")
check("E-2: d1 s0.1 gain not significant", 1, int(ls(4.0, 1.0, 0.1, "GB-CAG", "GB-CA", col="t_p_holm") > 0.05), 0)
check("E-2: d2 pooled structure gain +0.036", 0.036, ls(4.0, 2.0, "pooled", "GB-CAG", "GB-CA"), 0.0006, "($+0.036$ pooled\nover densities, 95\\% CI $[-0.002, 0.074]$, Holm $p=0.27$)")
check("E-2: d2 pooled gain CI low", -0.002, ls(4.0, 2.0, "pooled", "GB-CAG", "GB-CA", col="ci_low"), 0.0006)
check("E-2: d2 pooled gain CI high", 0.074, ls(4.0, 2.0, "pooled", "GB-CAG", "GB-CA", col="ci_high"), 0.0006)
check("E-2: d2 pooled gain Holm 0.27", 0.27, ls(4.0, 2.0, "pooled", "GB-CAG", "GB-CA", col="t_p_holm"), 0.005)
for m, q, pq in [("HAN", 0.178, 0.020), ("SAGE", 0.139, 0.030), ("SAGE+AG", 0.123, 0.042), ("LR-CA", 0.109, 0.022)]:
    check(f"E-2: {m} d1 s0.1 vs GB-CAG", q, ls(4.0, 1.0, 0.1, m), 0.0006, f"$+{q:.3f}$")
    check(f"E-2: {m} d1 s0.1 Holm p", pq, ls(4.0, 1.0, 0.1, m, col="t_p_holm"), 0.0006, f"$p={pq:.3f}$")
d2s = ["SAGE", "HAN", "RGCN", "SAGE+AG", "HGT+AG", "LR-CA"]
check("E-2: d2 s0.1 six methods min +0.053", 0.053, min(ls(4.0, 2.0, 0.1, m) for m in d2s), 0.0006, "exceeded it by $+0.053$ to $+0.087$ (Holm $p\\le0.037$)")
check("E-2: d2 s0.1 six methods max +0.087", 0.087, max(ls(4.0, 2.0, 0.1, m) for m in d2s), 0.0006)
check("E-2: d2 s0.1 six methods Holm <= 0.037", 0.037, max(ls(4.0, 2.0, 0.1, m, col="t_p_holm") for m in d2s), 0.0006)
sig_d2 = {m for m in SYN if m != "GB-CAG" and ls(4.0, 2.0, 0.1, m) > 0 and ls(4.0, 2.0, 0.1, m, col="t_p_holm") < 0.05}
check("E-2: exactly these six beat GB-CAG at d2 s0.1", 1, int(sig_d2 == set(d2s)), 0)
sig_d1 = {m for m in SYN if m != "GB-CAG" and ls(4.0, 1.0, 0.1, m) > 0 and ls(4.0, 1.0, 0.1, m, col="t_p_holm") < 0.05}
check("E-2: exactly HAN, SAGE, SAGE+AG, LR-CA beat GB-CAG at d1 s0.1", 1, int(sig_d1 == {"HAN", "SAGE", "SAGE+AG", "LR-CA"}), 0)
check("E-2: GB-CA 0.269 d1 s0.1", 0.269, LT[(1.0, 0.1, "GB-CA")], 0.0006, "(GB-CA 0.269 against 0.395 for LR-CA")
check("E-2: LR-CA 0.395 d1 s0.1", 0.395, LT[(1.0, 0.1, "LR-CA")], 0.0006)
check("E-2: SAGE 0.425 d1 s0.1", 0.425, LT[(1.0, 0.1, "SAGE")], 0.0006, "(SAGE 0.425 and HAN 0.464 at $\\delta=1$")
check("E-2: HAN 0.464 d1 s0.1", 0.464, LT[(1.0, 0.1, "HAN")], 0.0006)
check("E-2: best pure GNNs 0.03 above LR-CA", 0.03, LT[(1.0, 0.1, "SAGE")] - LT[(1.0, 0.1, "LR-CA")], 0.005, "0.03--0.07 above LR-CA")
check("E-2: best pure GNNs 0.07 above LR-CA", 0.07, LT[(1.0, 0.1, "HAN")] - LT[(1.0, 0.1, "LR-CA")], 0.005)
dense_gaps = [ls(4.0, 1.0, d, m) for d in [0.5, 0.9] for m in pure5]
check("E-2: d1 dense pure GNN gap min 0.11", -0.11, max(dense_gaps), 0.005, "the pure GNNs were 0.11--0.29 below GB-CAG")
check("E-2: d1 dense pure GNN gap max 0.29", -0.29, min(dense_gaps), 0.005)
sig_dense = {m for d in [0.5, 0.9] for m in pure5 if ls(4.0, 1.0, d, m, col="t_p_holm") < 0.05}
check("E-2: d1 dense significant only for HGT", 1, int(sig_dense == {"HGT"}), 0, "only for HGT, Holm $p<0.01$")
check("E-2: HGT d1 dense Holm < 0.01", 1, int(max(ls(4.0, 1.0, d, "HGT", col="t_p_holm") for d in [0.5, 0.9]) < 0.01), 0)
check("E-2: SAGE+AG d1 dense min +0.04", 0.04, ls(4.0, 1.0, 0.9, "SAGE+AG"), 0.005, "SAGE+AG was 0.04--0.07 above it (Holm $p\\ge0.079$)")
check("E-2: SAGE+AG d1 dense max +0.07", 0.07, ls(4.0, 1.0, 0.5, "SAGE+AG"), 0.005)
check("E-2: SAGE+AG d1 dense Holm >= 0.079", 0.079, min(ls(4.0, 1.0, d, "SAGE+AG", col="t_p_holm") for d in [0.5, 0.9]), 0.0006)
pooled_sig = [(s, m) for s in [1.0, 2.0] for m in SYN if m != "GB-CAG" and ls(4.0, s, "pooled", m) > 0 and ls(4.0, s, "pooled", m, col="t_p_holm") < 0.05]
check("E-2: no pooled advantage over GB-CAG", 0, len(pooled_sig), 0, "no method had significantly higher ring AP than GB-CAG at either shift")
check("E-2: SAGE+AG pooled d1 +0.079", 0.079, ls(4.0, 1.0, "pooled", "SAGE+AG"), 0.0006, "$+0.079$\n$[0.013, 0.146]$ at $\\delta=1$, Holm $p=0.20$")
check("E-2: SAGE+AG pooled d1 CI", 0.013, ls(4.0, 1.0, "pooled", "SAGE+AG", col="ci_low"), 0.0006)
check("E-2: SAGE+AG pooled d1 CI high", 0.146, ls(4.0, 1.0, "pooled", "SAGE+AG", col="ci_high"), 0.0006)
check("E-2: SAGE+AG pooled d1 Holm 0.20", 0.20, ls(4.0, 1.0, "pooled", "SAGE+AG", col="t_p_holm"), 0.005)
check("E-2: SAGE+AG pooled d2 +0.037", 0.037, ls(4.0, 2.0, "pooled", "SAGE+AG"), 0.0006, "$+0.037$ $[0.016, 0.057]$ at $\\delta=2$, $p=0.055$")
check("E-2: SAGE+AG pooled d2 CI low", 0.016, ls(4.0, 2.0, "pooled", "SAGE+AG", col="ci_low"), 0.0006)
check("E-2: SAGE+AG pooled d2 CI high", 0.057, ls(4.0, 2.0, "pooled", "SAGE+AG", col="ci_high"), 0.0006)
check("E-2: SAGE+AG pooled d2 Holm 0.055", 0.055, ls(4.0, 2.0, "pooled", "SAGE+AG", col="t_p_holm"), 0.0006)
check("E-2: SAGE+AG is the largest pooled difference d1", 1, int(max(SYN, key=lambda m: ls(4.0, 1.0, "pooled", m) if m != "GB-CAG" else -9) == "SAGE+AG"), 0)
check("E-2: HGT pooled d1 -0.202", -0.202, ls(4.0, 1.0, "pooled", "HGT"), 0.0006, "($-0.202$, $p=0.007$)")
check("E-2: HGT pooled d1 Holm 0.007", 0.007, ls(4.0, 1.0, "pooled", "HGT", col="t_p_holm"), 0.0006)
iso_ls = [ls(4.0, s, "pooled", m, metric="isolated_ap") for s in [1.0, 2.0] for m in pure5 + ["SAGE+AG", "HGT+AG"]]
check("E-2: isolated gap min 0.02", -0.02, max(iso_ls), 0.005, "by 0.02--0.11 pooled over densities")
check("E-2: isolated gap max 0.11", -0.11, min(iso_ls), 0.005)
sig_iso = {(s, m) for s in [1.0, 2.0] for m in pure5 + ["SAGE+AG", "HGT+AG"] if ls(4.0, s, "pooled", m, metric="isolated_ap", col="t_p_holm") < 0.05}
check("E-2: isolated significant set", 1, int(sig_iso == {(1.0, "SAGE"), (1.0, "HAN"), (1.0, "HGT+AG"), (2.0, "HGT+AG")}), 0,
      "significantly for SAGE, HAN and HGT+AG at $\\delta=1$ and for\nHGT+AG at $\\delta=2$")
# Article summary of E-2 (Section 6.6)
check("E-2 article: structure gain 0.29-0.30", 0.29, ls(4.0, 1.0, 0.5, "GB-CAG", "GB-CA"), 0.005, "graph features added 0.29--0.30 AP\n(Holm $p<0.01$)")
gnn_sparse = [ls(4.0, 1.0, 0.1, m) for m in ["SAGE", "HAN"]] + [ls(4.0, 2.0, 0.1, m) for m in ["SAGE", "HAN", "RGCN"]]
check("E-2 article: pure GNN advantage min 0.07", 0.07, min(gnn_sparse), 0.005, "outperformed GB-CAG by 0.07--0.18 (Holm $p\\le0.030$ within")
check("E-2 article: pure GNN advantage max 0.18", 0.18, max(gnn_sparse), 0.005)
check("E-2 article: pure GNN Holm <= 0.030", 1, int(max([ls(4.0, 1.0, 0.1, m, col="t_p_holm") for m in ["SAGE", "HAN"]] + [ls(4.0, 2.0, 0.1, m, col="t_p_holm") for m in ["SAGE", "HAN", "RGCN"]]) <= 0.030), 0)
sig_pure = {(s, m) for s in [1.0, 2.0] for m in pure5 if ls(4.0, s, 0.1, m) > 0 and ls(4.0, s, 0.1, m, col="t_p_holm") < 0.05}
check("E-2 article: SAGE, HAN at both shifts and RGCN at d2", 1, int(sig_pure == {(1.0, "SAGE"), (1.0, "HAN"), (2.0, "SAGE"), (2.0, "HAN"), (2.0, "RGCN")}), 0,
      "SAGE and HAN at both shifts and\nRGCN at $\\delta=2$")
check("E-2 article: LR-CA +0.11", 0.11, ls(4.0, 1.0, 0.1, "LR-CA"), 0.005, "($+0.11$ and $+0.06$)")
check("E-2 article: LR-CA +0.06", 0.06, ls(4.0, 2.0, 0.1, "LR-CA"), 0.005)
# Map rows
check("map: d2 volume 4 gain +0.04 n.s.", 0.04, ls(4.0, 2.0, "pooled", "GB-CAG", "GB-CA"), 0.005, "or 4 ($+0.04$, n.s.)")
check("map: d1 dense gain +0.29 to +0.30", 0.30, ls(4.0, 1.0, 0.9, "GB-CAG", "GB-CA"), 0.005, "& $+0.29$ to $+0.30$ &")
check("map: d>=1 sparse gain at most +0.02", 0.02, max(ls(4.0, s, 0.1, "GB-CAG", "GB-CA") for s in [1.0, 2.0]), 0.005, "At most $+0.02$")
sparse_adv = [ls(4.0, s, 0.1, m) for s, m in sig_pure] + [ls(4.0, s, 0.1, "LR-CA") for s in [1.0, 2.0]]
check("map: sparse advantage min +0.06", 0.06, min(sparse_adv), 0.005, "logistic regression, $+0.06$ to $+0.18$ over GB-CAG")
check("map: sparse advantage max +0.18", 0.18, max(sparse_adv), 0.005)

# E-3: training seeds (Online Resource S6; Section 6.4)
MS = pd.read_csv(R / "tables" / "modelseeds_comparisons.csv")
MS["ring_density"] = MS["ring_density"].astype(str)
def ms(name, dens, method, metric="ring_ap", col="mean_diff"):
    r = MS[(MS.training_seeds == name) & (MS.ring_density == str(dens)) & (MS.method == method) & (MS.metric == metric)]
    assert len(r) == 1
    return r.iloc[0][col]
check("E-3: SAGE pooled mean of 3 -0.011", -0.011, ms("mean of 3", "pooled", "SAGE"), 0.0006, "(SAGE $-0.011$, HGT $-0.040$ pooled over densities)")
check("E-3: HGT pooled mean of 3 -0.040", -0.040, ms("mean of 3", "pooled", "HGT"), 0.0006)
check("E-3: SAGE+AG seed 0 +0.013", 0.013, ms("seed 0", "pooled", "SAGE+AG"), 0.0006, "(SAGE+AG $+0.013$ with seed 0, $+0.009$ with the mean of three")
check("E-3: SAGE+AG mean of 3 +0.009", 0.009, ms("mean of 3", "pooled", "SAGE+AG"), 0.0006, "fell from $+0.013$ to $+0.009$")
check("E-3: HGT+AG seed 0 +0.012", 0.012, ms("seed 0", "pooled", "HGT+AG"), 0.0006, "HGT+AG $+0.012$\nand $+0.007$")
check("E-3: HGT+AG mean of 3 +0.007", 0.007, ms("mean of 3", "pooled", "HGT+AG"), 0.0006)
check("E-3: no pooled difference significant", 0, int((MS[(MS.metric == "ring_ap") & (MS.ring_density == "pooled")].t_p_holm < 0.05).sum()), 0,
      "none of these pooled differences is significant after Holm adjustment")
check("E-3: SAGE+AG s0.1 seed 0 +0.043", 0.043, ms("seed 0", 0.1, "SAGE+AG"), 0.0006, "by $+0.043$ with seed 0 (Holm $p=0.032$)")
check("E-3: SAGE+AG s0.1 seed 0 Holm 0.032", 0.032, ms("seed 0", 0.1, "SAGE+AG", col="t_p_holm"), 0.0006)
check("E-3: SAGE+AG s0.1 mean of 3 +0.030", 0.030, ms("mean of 3", 0.1, "SAGE+AG"), 0.0006, "$+0.030$ with the mean of three ($p=0.18$)")
check("E-3: SAGE+AG s0.1 mean of 3 Holm 0.18", 0.18, ms("mean of 3", 0.1, "SAGE+AG", col="t_p_holm"), 0.005)
check("E-3 article: +0.043 to +0.030 for sparse rings", 0.030, ms("mean of 3", 0.1, "SAGE+AG"), 0.0006, "from $+0.043$ to $+0.030$ for sparse rings")
MM = pd.read_csv(R / "tables" / "modelseeds_means.csv").set_index(["ring_density", "method", "model_seed"])["ring_ap"]
for sd, q in [(0, 0.973), (1, 0.965), (2, 0.956)]:
    check(f"E-3: SAGE+AG s0.1 seed {sd} mean {q}", q, MM[(0.1, "SAGE+AG", sd)], 0.0006)
check("E-3: text SAGE+AG seed means", 1, 1, 0, "0.973 with seed 0 against 0.965 and 0.956 with\nseeds 1 and 2")
check("E-3: GB-CAG s0.1 min 0.930", 0.930, min(MM[(0.1, "GB-CAG", s)] for s in range(3)), 0.0006, "GB-CAG's ranged from 0.930 to 0.938")
check("E-3: GB-CAG s0.1 max 0.938", 0.938, max(MM[(0.1, "GB-CAG", s)] for s in range(3)), 0.0006)
iso3 = [ms("mean of 3", "pooled", m, "isolated_ap") for m in ["SAGE", "HGT", "SAGE+AG", "HGT+AG"]]
check("E-3: isolated deficits min 0.086", -0.086, max(iso3), 0.0006, "($-0.086$ to $-0.219$ with the mean of three seeds, Holm $p\\le0.003$)")
check("E-3: isolated deficits max 0.219", -0.219, min(iso3), 0.0006)
check("E-3: isolated Holm <= 0.003", 1, int(max(ms("mean of 3", "pooled", m, "isolated_ap", "t_p_holm") for m in ["SAGE", "HGT", "SAGE+AG", "HGT+AG"]) <= 0.003), 0)
MV = pd.read_csv(R / "tables" / "modelseeds_variance.csv").set_index(["ring_density", "method"])
check("E-3: GB-CAG training s.d. min 0.001", 0.001, MV.loc[(slice(None), "GB-CAG"), "ring_ap_sd_training"].min(), 0.0005, "within a world 0.001--0.005")
check("E-3: GB-CAG training s.d. max 0.005", 0.005, MV.loc[(slice(None), "GB-CAG"), "ring_ap_sd_training"].max(), 0.0005)
check("E-3: HGT training s.d. min 0.022", 0.022, MV.loc[(slice(None), "HGT"), "ring_ap_sd_training"].min(), 0.0005, "(0.022--0.034 against 0.028--0.029)")
check("E-3: HGT training s.d. max 0.034", 0.034, MV.loc[(slice(None), "HGT"), "ring_ap_sd_training"].max(), 0.0005)
check("E-3: HGT worlds s.d. min 0.028", 0.028, MV.loc[(slice(None), "HGT"), "ring_ap_sd_worlds"].min(), 0.0005)
check("E-3: HGT worlds s.d. max 0.029", 0.029, MV.loc[(slice(None), "HGT"), "ring_ap_sd_worlds"].max(), 0.0005)

# E-4: per-regime tuning (Online Resource S7; Section 6.4)
RT = pd.read_csv(R / "tables" / "regimetuned_comparisons.csv")
def rtv(dens, method, comp, col="mean_diff", metric="ring_ap"):
    r = RT[(RT.ring_density == dens) & (RT.method == method) & (RT.comparison == comp) & (RT.metric == metric)]
    assert len(r) == 1
    return r.iloc[0][col]
main_best = json.load(open(R / "tuning" / "best_hyperparameters.json"))["best"]
tuned11 = ["LR-CA", "RF-CA", "GB-C", "GB-CA", "GB-CAG", "MLP-C", "SAGE", "RGCN", "HAN", "SimpleHGN", "HGT"]
for d, q in [("0.1", 8), ("0.9", 7)]:
    rb = json.load(open(R / f"tuning_regime_d0.0_s{d}" / "best_hyperparameters.json"))["best"]
    check(f"E-4: configurations changed at s={d}", q, sum(rb[m] != main_best[m] for m in tuned11), 0,
          "different configuration for 8 of the 11 tuned methods at $s=0.1$ and 7 at $s=0.9$")
ret = RT[(RT.metric == "ring_ap") & (RT.comparison == "re-tuned - main")]
check("E-4: no re-tuning change significant", 0, int((ret.t_p_holm < 0.05).sum()), 0, "changed no method's ring AP significantly")
check("E-4: largest change SAGE s0.9 -0.042", -0.042, ret.loc[ret.mean_diff.abs().idxmax(), "mean_diff"], 0.0006, "$-0.042$ (SAGE at $s=0.9$, Holm $p=0.51$)")
check("E-4: largest change is SAGE at s0.9", 1, int(tuple(ret.loc[ret.mean_diff.abs().idxmax(), ["method", "ring_density"]]) == ("SAGE", 0.9)), 0)
check("E-4: SAGE s0.9 Holm 0.51", 0.51, rtv(0.9, "SAGE", "re-tuned - main", "t_p_holm"), 0.005)
kept = ret[ret.mean_diff == 0]
check("E-4: kept configurations reproduce exactly", 1, int(all(rtv(d, m, "re-tuned - main") == 0 for d in [0.1, 0.9] for m in tuned11
      if json.load(open(R / f"tuning_regime_d0.0_s{d}" / "best_hyperparameters.json"))["best"][m] == main_best[m])), 0,
      "kept their configuration reproduced their results exactly")
vsr = RT[(RT.metric == "ring_ap") & (RT.comparison == "vs GB-CAG (re-tuned)") & RT.method.isin(pure5 + ["SAGE+AG", "HGT+AG"])]
check("E-4: no GNN or hybrid significantly above GB-CAG", 0, int(((vsr.mean_diff > 0) & (vsr.t_p_holm < 0.05)).sum()), 0,
      "no GNN or\nhybrid exceeded GB-CAG significantly in either regime")
check("E-4: SAGE+AG s0.1 +0.043", 0.043, rtv(0.1, "SAGE+AG", "vs GB-CAG (re-tuned)"), 0.0006, "(largest: SAGE+AG, $+0.043$ at $s=0.1$, Holm $p=0.092$)")
check("E-4: SAGE+AG s0.1 Holm 0.092", 0.092, rtv(0.1, "SAGE+AG", "vs GB-CAG (re-tuned)", "t_p_holm"), 0.0006)
check("E-4: SAGE+AG is the largest", 1, int(vsr.loc[vsr.mean_diff.idxmax(), "method"] == "SAGE+AG"), 0)
g09 = [rtv(0.9, m, "vs GB-CAG (re-tuned)") for m in pure5]
check("E-4: s0.9 pure GNN gap min 0.03", -0.03, max(g09), 0.005, "the pure GNNs were 0.03--0.07 below GB-CAG")
check("E-4: s0.9 pure GNN gap max 0.07", -0.07, min(g09), 0.005)
check("E-4: s0.9 only Simple-HGN significant", 1, int([m for m in pure5 if rtv(0.9, m, "vs GB-CAG (re-tuned)", "t_p_holm") < 0.05] == ["SimpleHGN"]), 0,
      "(significantly for Simple-HGN, Holm $p=0.026$)")
check("E-4: Simple-HGN s0.9 Holm 0.026", 0.026, rtv(0.9, "SimpleHGN", "vs GB-CAG (re-tuned)", "t_p_holm"), 0.0006)
check("E-4: hybrids match GB-CAG at s0.9", 0.0, max(abs(rtv(0.9, m, "vs GB-CAG (re-tuned)")) for m in ["SAGE+AG", "HGT+AG"]), 0.0006)

bad = [c for c in checks if not c[0]]
for ok, label, q, v, note in checks:
    if not ok:
        print("FAIL", f"{label}: quoted {q}, recomputed {v} {note}")
print(f"{len(checks) - len(bad)}/{len(checks)} checks passed (all sections)")
sys.exit(1 if bad else 0)
