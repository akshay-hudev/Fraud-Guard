"""Re-run one stored Kaggle GNN rotation with the current code and compare test scores. Writes nothing under results/.

Run from the repository root:
    PYTHONPATH=. .venv-research/bin/python paper/audit_phase1/scripts/repro_kaggle_gnn.py <repeat> <fold> <subdir> <file-stem> <method>
"""
import json, sys, time
import numpy as np
import research.experiments.kaggle_exp as K
from research.metrics import evaluate_scores
from research.realdata.kaggle import load
repeat, fold, sub, fname, m = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5]
K._DATA = load()
y = K._DATA.providers["y"].to_numpy()
best = json.load(open("results/kaggle/tuning/best_hyperparameters.json"))["best"]
train, val, test = K.splits(repeat)[fold]
z = np.load(f"results/{sub}/predictions/{fname}.npz", allow_pickle=True)
rows = {r["method"]: r for r in json.load(open(f"results/{sub}/runs/{fname}.json"))["rows"]}
t0 = time.time()
res = K.run_method(m, best[m], train, val, test)
met = evaluate_scores(y[val], res["val_scores"], y[test], res["test_scores"], budget=0.10)
d = float(np.max(np.abs(res["test_scores"].astype(np.float32).astype(float) - z[m].astype(float))))
print(f"{sub}/{fname} {m}: max|dscore|={d:.2e} AP new={met['ap']:.4f} stored={rows[m]['ap']:.4f} "
      f"best_epoch new={res['best_epoch']} stored={rows[m]['best_epoch']} ({time.time() - t0:.0f}s)")
