"""Check that numbers quoted in the manuscript text match the result artifacts.

Every check recomputes a quoted value from results/ (run records, generated tables, summary.json) and, where given,
confirms that the quoted text still appears in fraud_detection_paper.tex. Run from the repository root after
analyze.py:

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
TEX = Path("fraud_detection_paper.tex").read_text()
checks = []


def check(label, quoted, value, tol=0.0005, in_text=None):
    ok = abs(float(quoted) - float(value)) <= tol
    present = True if in_text is None else (in_text in TEX)
    checks.append((ok and present, label, quoted, round(float(value), 4), "" if present else f"[text '{in_text}' not found]"))


def rows(sub):
    out = []
    for f in sorted((R / sub / "runs").glob("*.json")):
        out += json.load(open(f))["rows"]
    df = pd.DataFrame(out)
    if sub.startswith("synthetic"):
        df = df[df.ring_shift.isin([0.0, 1.0, 2.0]) & df.seed.isin(range(1, 7))]
    return df


syn = rows("synthetic")
one = syn[syn.method == "GB-CAG"]
# Section 6 intro: test-period composition and chance levels
check("test claims per period (mean)", 4924, one.n_test.mean(), 0.5, "4{,}924 claims")
check("ring claims per test period", 384, one.n_ring.mean(), 0.5, "384 belong")
check("isolated claims per test period", 133, one.n_isolated.mean(), 0.5, "133 are isolated")
check("ring AP chance", 0.080, (one.n_ring / (one.n_test - one.n_isolated)).mean(), 0.0006, "0.080 and 0.029")
check("isolated AP chance", 0.029, (one.n_isolated / (one.n_test - one.n_ring)).mean(), 0.0006)
# 6.1 claim-only models at delta = 0
m = syn[syn.ring_shift == 0.0].groupby(["method", "ring_density"]).ring_ap.mean()
check("GB-C min ring AP d0", 0.084, m["GB-C"].min(), 0.0006, "GB-C 0.084--0.087")
check("GB-C max ring AP d0", 0.087, m["GB-C"].max(), 0.0006)
check("MLP-C min d0", 0.087, m["MLP-C"].min(), 0.0006, "MLP-C 0.087--0.095")
check("MLP-C max d0", 0.095, m["MLP-C"].max(), 0.0006)
g = syn[syn.method == "GB-C"].groupby("ring_shift").ring_ap.mean()
check("GB-C ring AP d1 about 0.33", 0.33, g[1.0], 0.01, "about 0.33")
check("GB-C ring AP d2 about 0.81", 0.81, g[2.0], 0.01)
check("GB-CA min d0", 0.889, m["GB-CA"].min(), 0.0006, "0.889--0.925")
check("GB-CA max d0", 0.925, m["GB-CA"].max(), 0.0006)
check("overlap s0.1", 0.254, m["Overlap-score"][0.1], 0.0006, "(0.254)")
check("overlap s0.5", 0.937, m["Overlap-score"][0.5], 0.0006, "0.937 and 0.985")
check("overlap s0.9", 0.985, m["Overlap-score"][0.9], 0.0006)
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
check("SAGE ring diff -0.0001", -0.0001, pv("ring_ap", "GB-CAG", "SAGE"), 0.00006, "$\\Delta=-0.0001$")
check("SAGE ring CI low -0.008", -0.008, pv("ring_ap", "GB-CAG", "SAGE", "ci_low"), 0.0006, "$[-0.008, 0.008]$")
check("SAGE ring CI high 0.008", 0.008, pv("ring_ap", "GB-CAG", "SAGE", "ci_high"), 0.0006)
for m, q in [("RGCN", -0.005), ("SimpleHGN", -0.008), ("HGT", -0.011), ("HAN", -0.024)]:
    check(f"{m} ring diff", q, pv("ring_ap", "GB-CAG", m), 0.0006)
check("HAN Holm p 0.063", 0.063, pv("ring_ap", "GB-CAG", "HAN", "t_p_holm"), 0.0006, "HAN $p=0.063$")
check("max upper CI of pure GNNs 0.008", 0.008, max(pv("ring_ap", "GB-CAG", m, "ci_high") for m in ["SAGE", "RGCN", "HAN", "SimpleHGN", "HGT"]), 0.0006)
check("SAGE+AG CI [0.002", 0.002, pv("ring_ap", "GB-CAG", "SAGE+AG", "ci_low"), 0.0006, "$[0.002, 0.014]$")
check("SAGE+AG CI 0.014]", 0.014, pv("ring_ap", "GB-CAG", "SAGE+AG", "ci_high"), 0.0006)
check("HGT+AG CI [-0.001", -0.001, pv("ring_ap", "GB-CAG", "HGT+AG", "ci_low"), 0.0006, "$[-0.001, 0.016]$")
check("HGT+AG CI 0.016]", 0.016, pv("ring_ap", "GB-CAG", "HGT+AG", "ci_high"), 0.0006)
check("SAGE+AG Holm 0.12", 0.12, pv("ring_ap", "GB-CAG", "SAGE+AG", "t_p_holm"), 0.005, "$p=0.12$ and $0.29$")
check("HGT+AG Holm 0.29", 0.29, pv("ring_ap", "GB-CAG", "HGT+AG", "t_p_holm"), 0.005)
for m, q in [("GB-CAG", 0.031), ("SAGE", 0.031), ("RGCN", 0.026), ("SimpleHGN", 0.024), ("HGT", 0.021), ("SAGE+AG", 0.039), ("HGT+AG", 0.039), ("HAN", 0.007)]:
    check(f"vs GB-CA ring {m}", q, pv("ring_ap", "GB-CA", m), 0.0006)
check("vs GB-CA Holm <= 0.021 (excl HAN)", 0.021, max(0.021, max(pv("ring_ap", "GB-CA", m, "t_p_holm") for m in ["GB-CAG", "SAGE", "RGCN", "SimpleHGN", "HGT", "SAGE+AG", "HGT+AG"])), 0.0)
check("HAN vs GB-CA p 0.085", 0.085, pv("ring_ap", "GB-CA", "HAN", "t_p_holm"), 0.0006, "$p=0.085$")
iso = [pv("isolated_ap", "GB-CAG", m) for m in ["SAGE", "RGCN", "HAN", "SimpleHGN", "HGT"]]
check("iso min deficit 0.085", -0.085, max(iso), 0.0006, "by 0.085 (SAGE) to 0.147")
check("iso max deficit 0.147", -0.147, min(iso), 0.0006)
check("iso Holm <= 0.009", 0.009, max(0.009, max(pv("isolated_ap", "GB-CAG", m, "t_p_holm") for m in ["SAGE", "RGCN", "HAN", "SimpleHGN", "HGT"])), 0.0)
check("MLP-C iso -0.019", -0.019, pv("isolated_ap", "GB-CAG", "MLP-C"), 0.0006, "within 0.019 of GB-CAG ($p=0.11$)")
check("MLP-C iso p 0.11", 0.11, pv("isolated_ap", "GB-CAG", "MLP-C", "t_p_holm"), 0.005)
ov = [pv("ap", "GB-CAG", m) for m in ["SAGE", "RGCN", "HAN", "SimpleHGN", "HGT"]]
check("overall min 0.007", -0.007, max(ov), 0.0006, "(by\n0.007--0.030)")
check("overall max 0.030", -0.030, min(ov), 0.0006)
check("overall RGCN/HAN/SHGN Holm <= 0.041", 0.041, max(pv("ap", "GB-CAG", m, "t_p_holm") for m in ["RGCN", "HAN", "SimpleHGN"]), 0.0006)
check("overall SAGE p 0.30", 0.30, pv("ap", "GB-CAG", "SAGE", "t_p_holm"), 0.005)
check("overall HGT p 0.059", 0.059, pv("ap", "GB-CAG", "HGT", "t_p_holm"), 0.0006)
check("overall hybrids p >= 0.87", 0.87, min(pv("ap", "GB-CAG", m, "t_p_holm") for m in ["SAGE+AG", "HGT+AG"]), 0.005)
C = pd.read_csv(R / "tables" / "synthetic_ring_ap_comparisons.csv")
c0 = C[(C.ring_shift == 0.0) & (C.ring_density == 0.1) & (C.method == "SAGE+AG") & (C.reference == "GB-CAG")].iloc[0]
check("SAGE+AG s0.1 +0.043", 0.043, c0.mean_diff, 0.0006, "SAGE+AG $+0.043$")
check("SAGE+AG s0.1 p 0.011", 0.011, c0.t_p, 0.0006)
check("SAGE+AG s0.1 Holm 0.095", 0.095, c0.t_p_holm, 0.0006, "Holm $p=0.095$")
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
check("nested max |change| 0.010", 0.010, N.mean_diff.abs().max(), 0.0006)
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
check("budget recall ceiling 0.476", 0.476, (0.05 * one.n_test / one.n_pos).clip(upper=1).mean(), 0.0006, "cannot exceed 0.476")
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
check("abstract: pure GNNs at most 0.16", 0.16, max(max(pr), max(pure)), 0.005, "most 0.16")
check("abstract: graph features 0.41-0.60 (low)", 0.41, min(fz[(0.5, "GB-CAG")], rt[(0.5, "GB-CAG")]), 0.005, "0.41--0.60")
check("abstract: graph features up to 0.60", 0.60, max(fz[(0.9, "GB-CAG")], rt[(0.9, "GB-CAG")]), 0.005)
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
check("discussion: structure adds 0.36-0.55 at low volume", 0.55, max(rc.loc["0.9", "mean_diff"], cells.loc["0.9", "mean_diff"]), 0.005, "0.36--0.55")

bad = [c for c in checks if not c[0]]
for ok, label, q, v, note in checks:
    if not ok:
        print("FAIL", f"{label}: quoted {q}, recomputed {v} {note}")
print(f"{len(checks) - len(bad)}/{len(checks)} checks passed (all sections)")
sys.exit(1 if bad else 0)
