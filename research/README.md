# Research pipeline (manuscript results)

Every number, table and figure in `fraud_detection_paper.tex` is produced by this package and recorded under
`results/`, with provenance (git commit, config hash, dataset hashes, library versions) in each run file.
The legacy pipeline in `training/` serves the demo application only; its old outputs are archived in
`archive/legacy_unverified/` and must not be cited.

## Environment

```bash
python3.12 -m venv .venv-research
# Linux: install CPU torch first
#   .venv-research/bin/pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu
.venv-research/bin/pip install -r research/requirements.txt      # full resolved set: requirements-lock.txt
```

torch 2.2.2 is the last release with Intel-macOS wheels, and it requires NumPy < 2 (the app's `.venv` pairs it
with NumPy 2.3, which breaks `tensor.numpy()`). Graph models train full-batch, so `pyg-lib`/`torch-sparse` are not needed.
Set `RESEARCH_THREADS` to control torch threads per process (default 4).

## Data

* **Synthetic benchmark** — generated on the fly from a seed (`research/synth/generator.py`); nothing to download.
* **Kaggle "Healthcare Provider Fraud Detection Analysis"** (Rohit Anand Gupta, CC0 Public Domain):
  download from https://www.kaggle.com/datasets/rohitrox/healthcare-provider-fraud-detection-analysis and unzip into
  `data/external/kaggle_provider_fraud/` (git-ignored). Expected SHA-256 of the training files:

  | File | SHA-256 |
  |---|---|
  | Train-1542865627584.csv | `09315a0e0c262ddfb91df15df76b196a80de2ff1001fad46b362b562df2e7e43` |
  | Train_Beneficiarydata-1542865627584.csv | `1fb7fb14efeaae04b970bfeb3176cdf938c755748217377109081d6dae01ecfd` |
  | Train_Inpatientdata-1542865627584.csv | `395bc2d01a102bf44a26ba4b36c63e7086943eaf2d5d1489c776bf47d1613ac0` |
  | Train_Outpatientdata-1542865627584.csv | `f532ed7e25f50d6abf3ba89ceeb2a23e2a352792f90d086b13999a561fd23889` |

  Only the labelled `Train*` files are used (the `Test*` files have no labels).

## Reproducing the results

```bash
PY=.venv-research/bin/python
$PY -m pytest research/tests -q                                   # validity tests (must pass)

# 1. Hyperparameters: equal budget per method on development worlds (seeds 901, 902)
$PY -m research.experiments.tune --tag all && $PY -m research.experiments.tune --merge
# 2. Regime grid, emergent (unseen) rings: 3 feature shifts x 3 densities x 6 worlds
$PY -m research.experiments.synthetic --shifts 0.0 1.0 2.0 --seeds 1 2 3 4 5 6
# 3. Persistent (known) rings, for the memorisation comparison
$PY -m research.experiments.synthetic --persistent 4 --shifts 0.0 --densities 0.1 0.9 \
    --seeds 1 2 3 4 5 6 --methods GB-CA GB-CAG MLP-C SAGE HGT Overlap-score --out results/synthetic_persistent
# 4. Mechanism ablations (delta = 0, s = 0.9, worlds 1-6)
$PY -m research.experiments.ablation
# 5. Real data: tune, merge, 5-fold evaluation
$PY -m research.experiments.kaggle_exp --stage tune && $PY -m research.experiments.kaggle_exp --stage merge
$PY -m research.experiments.kaggle_exp --stage eval
# 6. Two-stage routing (GB-CAG -> HGT / SAGE) and reliance analyses; also checks that the
#    retrained models reproduce the predictions stored by step 2 (all 54 worlds)
$PY -m research.experiments.cascade
# 7. HGT single-factor ablations (depth, heads, relations, node inputs; delta = 0, s = 0.9, worlds 1-6)
$PY -m research.experiments.ablation --family arch
# 8. Re-analysis of the small fixed 2,100-claim benchmark (data/raw); nested tuning check on the real data
$PY -m research.experiments.legacy_reanalysis
$PY -m research.experiments.kaggle_exp --stage nested
# 9. Cost benchmark: run alone, single thread
OMP_NUM_THREADS=1 RESEARCH_THREADS=1 $PY -m research.experiments.latency
# 10. Low-volume stress test (4 instead of 16 ring claims per month, delta = 0): with the main tuning (frozen),
#     then with the tuning protocol repeated on low-volume development worlds (re-tuned)
$PY -m research.experiments.synthetic --shifts 0.0 --seeds 1 2 3 4 5 6 --ring-claims-per-month 4 \
    --out results/synthetic_lowvolume
$PY -m research.experiments.tune --ring-claims-per-month 4 --out results/tuning_lowvolume
$PY -m research.experiments.tune --merge --ring-claims-per-month 4 --out results/tuning_lowvolume
$PY -m research.experiments.synthetic --shifts 0.0 --seeds 1 2 3 4 5 6 --ring-claims-per-month 4 \
    --hps results/tuning_lowvolume/best_hyperparameters.json --out results/synthetic_lowvolume_retuned
# 11. Tables, statistics, figures; paste tables into the manuscript; then check artifacts
$PY -m research.experiments.analyze && $PY -m research.experiments.figures
$PY -m research.experiments.sync_manuscript && $PY verify.py
# 12. Second round: seeds 7-12 of the main grid, ring volumes 8 and 2, the confirmatory hybrid test, repeated
#     cross-validation and nested GNN tuning on the real data. The script lists every command (136 jobs):
PARALLEL=4 bash paper/hiss_review/run_round2.sh
#     kaggle_exp.py overwrites an existing rotation file, so rerun its commands only with the same method list.
# 13. Third round: delta = 1, 2 at volume 4; training seeds 1, 2; per-regime tuning; repeated rotations for the
#     remaining GNNs (116 jobs)
PARALLEL=4 bash paper/hiss_review/run_round3.sh
# 14. Fourth round: BWGNN and CARE-GNN (tuning, main grid, low volume, volumes 8 and 2, public data) and four
#     further ring shapes and sizes with every method (--ring-shape, --ring-doctors, --ring-patients; 107 jobs)
PARALLEL=4 bash paper/hiss_review/run_round4.sh
```

**Second experimental round.** `paper/hiss_review/analysis_plan_round2.md` pre-declares, and
`paper/hiss_review/run_round2.sh` runs, every job of the second round:

* the main grid for seeds 7–12 (final N = 12);
* ring volumes 8 and 2 at δ = 0;
* the confirmatory hybrid test on fresh seeds 7–18;
* repeated cross-validation (`kaggle_exp.py --stage eval --repeat 1..4`) on the public data;
* nested GNN tuning (`--stage nested --nested-gnns`) on the public data.

The round runs about 9 hours with four workers (`PARALLEL=4`). The third and fourth rounds
(`analysis_plan_round3.md`, `analysis_plan_round4.md`) are run the same way by `run_round3.sh` and `run_round4.sh`
(about 9 and 7 hours with four workers).

**Statistical units.** The nine regimes of one generator seed share their legitimate claims, isolated fraud and
ring members (the generator draws them before applying δ and s), so comparisons pooled over regimes test the twelve
seed means (`metrics.clustered_comparison`; seeds 1–6 for the secondary analyses). The real-data rotations share
training folds, so comparisons over them use the corrected resampled t-test of Nadeau and Bengio (2003)
(`metrics.corrected_resampled_comparison`); over the 25 repeated rotations the variance factor is 1/25 + 1/3
(Bouckaert and Frank 2004). The naive values are kept in the CSVs (`naive_*` columns) for comparison.

Steps 1–8 accept `--methods` / `--seeds` / `--shifts` / `--folds` subsets so they can be split across processes;
finished runs are skipped on restart. Wall-clock on a 4-core Intel laptop CPU: roughly 2 h tuning, 3 h synthetic
grid, 1 h ablations, 2 h Kaggle, 3 h routing and HGT ablations (run in parallel), 15 min cost benchmark.

## Layout

| Path | Content |
|---|---|
| `synth/generator.py` | regime-controllable generator (feature shift δ, structural density s, ring volume, ring shape and size, emergent/persistent rings) |
| `synth/features.py` | leakage-safe claim (C), aggregate (A) and graph (G) features per audit period |
| `synth/graph.py` | per-period heterogeneous graphs with reverse relations and degree encodings; HAN meta-path graphs |
| `models/gnn.py` | SAGE, RGCN, HAN, Simple-HGN, HGT, BWGNN, CARE-GNN and MLP with a shared input projection and head |
| `models/train.py` | full-batch training with early stopping on validation AP; tabular model factory |
| `realdata/kaggle.py` | provider-level features, label-neighbourhood features (training labels only), provider graph |
| `metrics.py` | AP, ROC-AUC, validation-selected thresholds, recall at budget, paired tests, Holm |
| `experiments/` | tuning, synthetic grid, ablations, routing, cost, legacy re-analysis, Kaggle, analysis, figures, manuscript table sync |
| `tests/test_validity.py` | leakage, label-invariance, period isolation, ring disjointness, alignment, information flow |
| `tests/test_routing_and_ablations.py` | routing uses validation only and is per-claim; ablation transforms; test-period rewiring; exact computation subgraphs |
| `tests/test_statistics.py` | regimes of a seed share their background (why the seed is the unit); seed-level and corrected resampled tests |
