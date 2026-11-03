"""Compare the reruns of the 33 run files whose recorded source state matches no commit with the stored records.

The reruns were made from an exact copy of the tagged code (hiss-submission-v1), with the original commands.
Usage, from the repository root:
    .venv-research/bin/python paper/audit_phase1/scripts/compare_uncommitted.py <copy of the tagged tree>
Writes paper/audit_phase1/logs/m4_reruns.csv and prints a summary.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

UNCOMMITTED = {"1e7bfee25a2f3b9e", "c601208ff2218c10", "dfb870223106e371"}
stored, rerun = Path("results"), Path(sys.argv[1]) / "results"
rows = []
for f in sorted(stored.rglob("runs/*.json")):
    payload = json.load(open(f))
    if payload["provenance"]["git"].get("research_sources_sha256") not in UNCOMMITTED:
        continue
    sub = f.parts[1]
    g = rerun / sub / "runs" / f.name
    if not g.exists():
        rows.append({"file": f"{sub}/{f.name}", "method": "-", "status": "not rerun"})
        continue
    new = {r["method"]: r for r in json.load(open(g))["rows"]}
    z_old = stored / sub / "predictions" / (f.stem + ".npz")
    z_new = rerun / sub / "predictions" / (f.stem + ".npz")
    a, b = (np.load(z_old, allow_pickle=True), np.load(z_new, allow_pickle=True)) if z_old.exists() else (None, None)
    for r in payload["rows"]:
        m, n = r["method"], new.get(r["method"])
        if n is None:
            rows.append({"file": f"{sub}/{f.name}", "method": m, "status": "method missing"})
            continue
        key = "ap"
        row = {"file": f"{sub}/{f.name}", "method": m, "ap_stored": r[key], "ap_rerun": n[key],
               "ap_abs_diff": abs(r[key] - n[key])}
        if a is not None:
            row["max_abs_score_diff"] = float(np.max(np.abs(a[m].astype(float) - b[m].astype(float))))
        if "hp" in r:
            row["same_config"] = r["hp"] == n["hp"]
        row["status"] = "identical" if row["ap_abs_diff"] == 0 and row.get("max_abs_score_diff", 0) == 0 and row.get("same_config", True) else "DIFFERS"
        rows.append(row)
df = pd.DataFrame(rows)
df.to_csv("paper/audit_phase1/logs/m4_reruns.csv", index=False)
print(df.groupby("status").size().to_string())
print("files:", df.file.nunique(), "| method-runs compared:", int(df.ap_rerun.notna().sum()) if "ap_rerun" in df else 0)
if (df.status == "DIFFERS").any():
    print(df[df.status == "DIFFERS"].to_string())
