"""Re-run stored synthetic worlds with the current code and compare test scores to the stored predictions.

Writes nothing under results/ (retrained outputs go to a temporary directory). From the repository root:
    PYTHONPATH=. .venv-research/bin/python paper/audit_phase1/scripts/repro_synth.py <subdir> <hps-json> <tag-stem> <method>...
"""
import json, sys, tempfile, time
from pathlib import Path
import numpy as np
from research.experiments.synthetic import run_dataset
from research.synth.generator import GeneratorConfig

sub, hps_path, stem, methods = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
R = Path("results") / sub
run_file = next((R / "runs").glob(f"{stem}__*.json"))
cfg_d = json.load(open(run_file))["provenance"]["config"]
cfg = GeneratorConfig(**cfg_d)
hps = json.load(open(hps_path))["best"]
stored_rows = {}
stored_preds = {}
for f in (R / "runs").glob(f"{stem}__*.json"):
    for r in json.load(open(f))["rows"]:
        stored_rows[r["method"]] = r
    z = np.load(R / "predictions" / (f.stem + ".npz"), allow_pickle=True)
    for m in z.files:
        stored_preds[m] = z[m]
out = Path(tempfile.mkdtemp(prefix="repro_synth_")) / sub
t0 = time.time()
rows = run_dataset(cfg, methods, hps, model_seed=0, out_dir=out, verbose=False, group="repro")
z = np.load(next((out / "predictions").glob(f"{stem}__repro.npz")), allow_pickle=True)
assert (z["claim_id"] == stored_preds["claim_id"]).all()
for r in rows:
    m = r["method"]; s = stored_rows[m]
    d = float(np.max(np.abs(z[m].astype(float) - stored_preds[m].astype(float))))
    print(f"{sub} {stem} {m:9s} max|dscore|={d:.2e} ringAP new={r['ring_ap']:.4f} stored={s['ring_ap']:.4f} "
          f"isoAP new={r['isolated_ap']:.4f} stored={s['isolated_ap']:.4f}", flush=True)
print(f"done in {time.time() - t0:.0f}s")
