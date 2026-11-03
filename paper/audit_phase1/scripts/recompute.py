"""Independent recomputation of the main manuscript tables from raw run rows (does not import analyze.py)."""
import json, re
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
R = Path("results"); TEX = Path("fraud_detection_paper.tex").read_text()
def rows(sub):
    out = []
    for f in sorted((R / sub / "runs").glob("*.json")): out += json.load(open(f))["rows"]
    return pd.DataFrame(out)
def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for k, i in enumerate(o): run = max(run, min(1, (m - k) * p[i])); adj[i] = run
    return adj
def table(label):
    blk = next(b for b in re.findall(r"\\begin\{table\}.*?\\end\{table\}", TEX, re.S) if "\\label{" + label + "}" in b)
    out = {}
    for line in blk.split("\n"):
        if "&" in line and "\\\\" in line and not line.strip().startswith(("Method", " &", "&")):
            cells = [c.strip() for c in line.replace("\\\\", "").split("&")]
            out[re.sub(r"\\textbf\{(.*?)\}", r"\1", cells[0])] = [re.sub(r"\\textbf\{(.*?)\}", r"\1", c) for c in cells[1:]]
    return out
name = {"Simple-HGN": "SimpleHGN", "Overlap score": "Overlap-score"}
syn = rows("synthetic"); syn = syn[syn.ring_shift.isin([0., 1., 2.]) & syn.seed.between(1, 12)]
cnt = syn.groupby(["ring_shift", "ring_density", "method"]).seed.nunique()
print("main grid: cells", len(cnt), "seeds per cell min/max", cnt.min(), cnt.max(), "| shifts on disk:", sorted(rows("synthetic").ring_shift.unique()),
      "seeds on disk:", sorted(rows("synthetic").seed.unique()))
for lab, metric in [("tab:regime_ringap", "ring_ap"), ("tab:regime_isoap", "isolated_ap")]:
    m = syn.groupby(["method", "ring_shift", "ring_density"])[metric].mean()
    bad = 0; n = 0
    for meth, vals in table(lab).items():
        mm = name.get(meth, meth); cols = [(s, d) for s in [0., 1., 2.] for d in [.1, .5, .9]]
        for (s, d), v in zip(cols, vals):
            n += 1; bad += abs(float(v) - m[(mm, s, d)]) > 0.0005
    print(f"{lab}: {n} cells checked, {bad} mismatches")
# pooled table: seed-level means over nine regimes, paired t, Holm within metric over the 10 rows
piv = {mt: syn.pivot_table(index=["ring_shift", "ring_density", "seed"], columns="method", values=mt) for mt in ["ring_ap", "isolated_ap", "ap"]}
meths = ["SAGE", "RGCN", "HAN", "SimpleHGN", "HGT", "BWGNN", "CARE-GNN", "SAGE+AG", "HGT+AG", "MLP-C"]
tp = table("tab:pooled"); bad = 0
for mt, col in [("ring_ap", 0), ("isolated_ap", 2), ("ap", 4)]:
    sm = piv[mt].groupby(level="seed").mean()
    d = {m: (sm[m] - sm["GB-CAG"]).to_numpy() for m in meths}
    p = holm([stats.ttest_1samp(d[m], 0).pvalue for m in meths])
    for m, pa in zip(meths, p):
        hw = stats.t.ppf(.975, len(d[m]) - 1) * d[m].std(ddof=1) / np.sqrt(len(d[m]))
        cell = tp[{"SimpleHGN": "Simple-HGN"}.get(m, m)]
        q_mean = float(re.search(r"([+-]?\d\.\d+)", cell[col].replace("$", "")).group(1)); q_hw = float(cell[col].split("pm")[-1].replace("$", "").strip())
        q_p = cell[col + 1].replace("$<$", "<")
        ok = abs(q_mean - d[m].mean()) <= 0.0005 and abs(q_hw - hw) <= 0.0005 and ((q_p == "<0.001" and pa < 0.001) or (q_p != "<0.001" and abs(float(q_p) - pa) <= 0.0005))
        bad += not ok
        if not ok: print("  pooled mismatch", mt, m, cell[col], cell[col + 1], round(d[m].mean(), 4), round(hw, 4), round(pa, 4))
print(f"tab:pooled: 30 cells checked, {bad} mismatches")
# Kaggle table
k = rows("kaggle"); tk = table("tab:kaggle"); bad = 0
for meth, vals in tk.items():
    g = k[k.method == name.get(meth, meth)]
    for v, c in zip(vals, ["ap", "roc_auc", "f1", "recall_at_10pct"]):
        mu, sd = [float(x) for x in v.replace("$\\pm$", " ").split()]
        bad += abs(mu - g[c].mean()) > 0.0005 or abs(sd - g[c].std(ddof=1)) > 0.0005
print(f"tab:kaggle: {len(tk) * 4} cells, {bad} mismatches; folds per method {k.groupby('method').fold.nunique().unique()}")
# low-volume table
lv = {"frozen": rows("synthetic_lowvolume"), "retuned": rows("synthetic_lowvolume_retuned")}
base = syn[(syn.ring_shift == 0) & syn.seed.between(1, 6)]
tl = table("tab:lowvolume"); bad = 0
for meth, vals in tl.items():
    mm = name.get(meth, meth); exp = []
    for df in [base, lv["frozen"], lv["retuned"]]:
        mg = df[df.method == mm].groupby("ring_density").ring_ap.mean(); exp += [mg[d] for d in [.1, .5, .9]]
    bad += sum(abs(float(v) - e) > 0.0005 for v, e in zip(vals, exp))
print(f"tab:lowvolume: {len(tl) * 9} cells, {bad} mismatches; worlds per cell frozen/retuned:",
      lv["frozen"].groupby(["ring_density", "method"]).seed.nunique().unique(), lv["retuned"].groupby(["ring_density", "method"]).seed.nunique().unique())
# protocol table
per = rows("synthetic_persistent"); tpro = table("tab:protocol")
print("persistent seeds:", sorted(per.seed.unique()), "densities:", sorted(per.ring_density.unique()), "methods:", sorted(per.method.unique()))
# confirmatory test
c = rows("synthetic_hybrid_confirm"); pc = c.pivot_table(index="seed", columns="method", values="ring_ap")
d = pc["SAGE+AG"] - pc["GB-CAG"]; t = stats.ttest_1samp(d, 0)
print("confirm: seeds", sorted(c.seed.unique()), "vol", c.ring_claims_per_month.unique(), "dens", c.ring_density.unique(),
      f"diff={d.mean():.4f} CI=[{d.mean() - stats.t.ppf(.975, 11) * d.std(ddof=1) / np.sqrt(12):.3f},{d.mean() + stats.t.ppf(.975, 11) * d.std(ddof=1) / np.sqrt(12):.3f}] p={t.pvalue:.4f} pos={(d > 0).sum()}",
      f"means SAGE+AG={pc['SAGE+AG'].mean():.3f} GB-CAG={pc['GB-CAG'].mean():.3f}")
