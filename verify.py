"""Check that the manuscript's result artifacts exist, carry provenance, and are internally consistent.

Run after the research pipeline (see research/README.md):
    .venv-research/bin/python verify.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
REQUIRED_TABLES = ["synthetic_ring_ap.csv", "synthetic_ring_ap_comparisons.csv", "synthetic_isolated_ap.csv",
                   "protocol_ring_ap.csv", "ablation_ring_ap.csv", "kaggle_metrics.csv", "kaggle_comparisons.csv",
                   "cascade_comparisons.csv", "reliance.csv", "kaggle_nested.csv", "hgt_arch.tex", "legacy.tex"]
REQUIRED_FIGURES = ["fig_regime_ring_ap.pdf", "fig_delta_heatmaps.pdf", "fig_ablation.pdf", "fig_kaggle_ap.pdf",
                    "fig_protocol.pdf", "fig_cascade.pdf", "fig_ring_graph.pdf"]

problems = []


def check_runs(sub: str, key_fields: list[str]) -> int:
    files = sorted((RESULTS / sub / "runs").glob("*.json"))
    if not files:
        problems.append(f"no run files in results/{sub}/runs")
    for f in files:
        payload = json.loads(f.read_text())
        prov = payload.get("provenance", {})
        for field in ["git", "config_hash", "environment", "created_utc"]:
            if field not in prov:
                problems.append(f"{f.name}: provenance missing '{field}'")
        for row in payload.get("rows", []):
            missing = [k for k in key_fields if k not in row]
            if missing:
                problems.append(f"{f.name}: row missing {missing}")
                break
    return len(files)


counts = {
    "synthetic": check_runs("synthetic", ["method", "ring_ap", "ap", "seed"]),
    "synthetic_persistent": check_runs("synthetic_persistent", ["method", "ring_ap", "seed"]),
    "ablation": check_runs("ablation", ["method", "variant", "ring_ap"]),
    "kaggle": check_runs("kaggle", ["method", "fold", "ap"]),
    "kaggle_nested": check_runs("kaggle_nested", ["method", "fold", "ap", "hp"]),
    "cascade": check_runs("cascade", ["method", "ap", "ring_ap", "isolated_ap", "precision_at_5pct"]),
    "ablation_arch": check_runs("ablation_arch", ["method", "variant", "ring_ap"]),
}
for f in sorted((RESULTS / "cascade" / "runs").glob("*.json")):
    for model, check in json.loads(f.read_text()).get("reproduction", {}).items():
        if check["max_abs_diff"] > 1e-6:
            problems.append(f"{f.name}: {model} does not reproduce the stored predictions ({check['max_abs_diff']:.2e})")
for name in ["legacy/legacy_reanalysis.json", "latency/latency.json"]:
    if not (RESULTS / name).exists():
        problems.append(f"missing results/{name}")
    elif "provenance" not in json.loads((RESULTS / name).read_text()):
        problems.append(f"results/{name}: no provenance")
for name in REQUIRED_TABLES:
    if not (RESULTS / "tables" / name).exists():
        problems.append(f"missing table results/tables/{name}")
for name in REQUIRED_FIGURES:
    if not (RESULTS / "figures" / name).exists():
        problems.append(f"missing figure results/figures/{name}")
if not (RESULTS / "tuning" / "best_hyperparameters.json").exists():
    problems.append("missing results/tuning/best_hyperparameters.json")

print("run files:", counts)
if problems:
    print("PROBLEMS:")
    for p in problems:
        print("  -", p)
    sys.exit(1)
print("All result artifacts present with provenance.")
