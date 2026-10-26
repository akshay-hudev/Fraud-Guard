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
                   "cascade_comparisons.csv", "reliance.csv", "kaggle_nested.csv", "hgt_arch.tex", "legacy.tex",
                   "lowvolume.tex", "lowvolume_comparisons.csv", "volume_signal.csv", "kaggle_bootstrap.csv",
                   "dose_response.tex", "dose_response_comparisons.csv", "hybrid_confirmation.json",
                   "kaggle_repeated.tex", "kaggle_repeated_comparisons.csv", "kaggle_nested.tex"]
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
    "synthetic_lowvolume": check_runs("synthetic_lowvolume", ["method", "ring_ap", "seed", "ring_claims_per_month"]),
    "synthetic_lowvolume_retuned": check_runs("synthetic_lowvolume_retuned",
                                              ["method", "ring_ap", "seed", "ring_claims_per_month"]),
    "synthetic_volume8": check_runs("synthetic_volume8", ["method", "ring_ap", "seed", "ring_claims_per_month"]),
    "synthetic_volume2": check_runs("synthetic_volume2", ["method", "ring_ap", "seed", "ring_claims_per_month"]),
    "synthetic_hybrid_confirm": check_runs("synthetic_hybrid_confirm", ["method", "ring_ap", "seed"]),
    "kaggle_repeated": check_runs("kaggle_repeated", ["method", "fold", "repeat", "ap"]),
}
# Second round (paper/hiss_review/analysis_plan_round2.md): the main grid is complete for seeds 1-12.
main_worlds = {(r["ring_shift"], r["ring_density"], r["seed"]) for f in (RESULTS / "synthetic" / "runs").glob("*.json")
               for r in json.loads(f.read_text())["rows"] if r["ring_shift"] in (0.0, 1.0, 2.0) and 1 <= r["seed"] <= 12}
if len(main_worlds) != 108:
    problems.append(f"main grid: {len(main_worlds)} of 108 worlds (9 regimes x seeds 1-12)")
for sub in ["synthetic_lowvolume", "synthetic_lowvolume_retuned"]:
    for f in sorted((RESULTS / sub / "runs").glob("*.json")):
        if any(row.get("ring_claims_per_month") != 4.0 for row in json.loads(f.read_text())["rows"]):
            problems.append(f"{sub}/{f.name}: not a low-volume (4 claims/month) run")
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
for name in ["tuning", "tuning_lowvolume"]:
    if not (RESULTS / name / "best_hyperparameters.json").exists():
        problems.append(f"missing results/{name}/best_hyperparameters.json")

print("run files:", counts)
if problems:
    print("PROBLEMS:")
    for p in problems:
        print("  -", p)
    sys.exit(1)
print("All result artifacts present with provenance.")
