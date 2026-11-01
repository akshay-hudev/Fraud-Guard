"""Re-run stored Kaggle rotations (tabular methods) with the current code; compare scores to stored predictions.
Run from the repository root: PYTHONPATH=. .venv-research/bin/python paper/audit_phase1/scripts/repro_kaggle.py [args]

Also prints the exact provider split sizes, disjointness and per-split fraud counts. Writes nothing under results/.
"""
import json, sys
from pathlib import Path
import numpy as np
import research.experiments.kaggle_exp as K
from research.metrics import evaluate_scores
from research.realdata.kaggle import load

K._DATA = load()
y = K._DATA.providers["y"].to_numpy()
best = json.load(open("results/kaggle/tuning/best_hyperparameters.json"))["best"]
tr, va, te = K.tuning_split()
print("tuning split sizes", len(tr), len(va), len(te), "fraud", y[tr].sum(), y[va].sum(), y[te].sum())
for repeat, fold, sub, fname, methods in [
        (0, 0, "kaggle", "fold0_fast", ["LR-P", "RF-P", "GB-P", "GB-PG", "GB-PGL", "MLP-P"]),
        (0, 3, "kaggle", "fold3_fast", ["GB-P", "GB-PG", "GB-PGL"]),
        (3, 1, "kaggle_repeated", "r3_fold1_main", ["LR-P", "RF-P", "GB-P", "GB-PG", "GB-PGL", "MLP-P"]),
        (4, 4, "kaggle_repeated", "r4_fold4_main", ["GB-P", "GB-PG", "GB-PGL"])]:
    train, val, test = K.splits(repeat)[fold]
    assert not (set(train) & set(val) or set(train) & set(test) or set(val) & set(test))
    sval, stest = set(va), set(test)
    print(f"\nrepeat {repeat} fold {fold}: n train/val/test {len(train)}/{len(val)}/{len(test)}; fraud {y[train].sum()}/{y[val].sum()}/{y[test].sum()};"
          f" test providers that were in the tuning-validation set: {len(stest & sval)}")
    z = np.load(f"results/{sub}/predictions/{fname}.npz", allow_pickle=True)
    assert (z["provider_id"] == K._DATA.providers["provider_id"].to_numpy()[test]).all()
    rows = {r["method"]: r for r in json.load(open(f"results/{sub}/runs/{fname}.json"))["rows"]}
    for m in methods:
        res = K.run_method(m, best[m], train, val, test)
        met = evaluate_scores(y[val], res["val_scores"], y[test], res["test_scores"], budget=0.10)
        d = float(np.max(np.abs(res["test_scores"].astype(np.float32).astype(float) - z[m].astype(float))))
        print(f"  {m:7s} max|dscore|={d:.2e} AP new={met['ap']:.4f} stored={rows[m]['ap']:.4f}", flush=True)
