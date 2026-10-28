#!/usr/bin/env bash
# Second experimental round (paper/hiss_review/analysis_plan_round2.md): every job, run with PARALLEL workers.
# Synthetic jobs skip worlds whose run files exist, so the script can be restarted; real-data jobs rerun.
#
#   nohup bash paper/hiss_review/run_round2.sh > results/round2_logs/queue.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/../.."
PARALLEL=${PARALLEL:-4}
LOGS=results/round2_logs
mkdir -p "$LOGS"
ENV="RESEARCH_THREADS=2 OMP_NUM_THREADS=2"
PY=".venv-research/bin/python -m research.experiments"
JOBS=$(mktemp)

# A. Main grid, seeds 7-12 (one job per world).
for shift in 0.0 1.0 2.0; do for dens in 0.1 0.5 0.9; do for seed in 7 8 9 10 11 12; do
  echo "$ENV $PY.synthetic --shifts $shift --densities $dens --seeds $seed --out results/synthetic --group all > $LOGS/A_d${shift}_s${dens}_g${seed}.log 2>&1"
done; done; done >> "$JOBS"
# B. Ring-volume dose-response: 8 and 2 claims per month, delta = 0, seeds 1-6, frozen hyperparameters.
for vol in 8 2; do for dens in 0.1 0.5 0.9; do for seed in 1 2 3 4 5 6; do
  echo "$ENV $PY.synthetic --shifts 0.0 --densities $dens --seeds $seed --ring-claims-per-month $vol --out results/synthetic_volume$vol --group all > $LOGS/B_v${vol}_s${dens}_g${seed}.log 2>&1"
done; done; done >> "$JOBS"
# C. Confirmatory hybrid test: fresh seeds 7-18, s = 0.1, volume 4, re-tuned configurations.
echo "$ENV $PY.synthetic --shifts 0.0 --densities 0.1 --seeds 7 8 9 10 11 12 13 14 15 16 17 18 --ring-claims-per-month 4 --hps results/tuning_lowvolume/best_hyperparameters.json --methods GB-CAG SAGE+AG --out results/synthetic_hybrid_confirm --group confirm > $LOGS/C_confirm.log 2>&1" >> "$JOBS"
# E1. Nested GNN tuning, longest jobs first (HGT, Simple-HGN), so they do not form a tail.
for fold in 0 1 2 3 4; do
  echo "$ENV $PY.kaggle_exp --stage nested --nested-gnns --folds $fold --methods HGT HGT+G --tag gnn_hgt > $LOGS/E_hgt_f$fold.log 2>&1"
  echo "$ENV $PY.kaggle_exp --stage nested --nested-gnns --folds $fold --methods SimpleHGN --tag gnn_shgn > $LOGS/E_shgn_f$fold.log 2>&1"
done >> "$JOBS"
# D. Repeated cross-validation on the public data: repeats 1-4 x five rotations.
for rep in 1 2 3 4; do for fold in 0 1 2 3 4; do
  echo "$ENV $PY.kaggle_exp --stage eval --repeat $rep --folds $fold --methods LR-P RF-P GB-P GB-PG GB-PGL MLP-P SAGE HAN HGT --tag main > $LOGS/D_r${rep}_f$fold.log 2>&1"
done; done >> "$JOBS"
# E2. Remaining nested GNN tuning.
for fold in 0 1 2 3 4; do
  echo "$ENV $PY.kaggle_exp --stage nested --nested-gnns --folds $fold --methods RGCN --tag gnn_rgcn > $LOGS/E_rgcn_f$fold.log 2>&1"
  echo "$ENV $PY.kaggle_exp --stage nested --nested-gnns --folds $fold --methods SAGE SAGE+G --tag gnn_sage > $LOGS/E_sage_f$fold.log 2>&1"
  echo "$ENV $PY.kaggle_exp --stage nested --nested-gnns --folds $fold --methods HAN --tag gnn_han > $LOGS/E_han_f$fold.log 2>&1"
done >> "$JOBS"

echo "$(wc -l < "$JOBS") jobs, $PARALLEL in parallel, started $(date)"
# Each job is passed as one argument (BSD xargs -I limits replacements to 255 bytes).
tr '\n' '\0' < "$JOBS" | xargs -0 -n 1 -P "$PARALLEL" \
  sh -c 'if eval "$1"; then echo "done $(date +%H:%M): ${1%% >*}"; else echo "FAILED $(date +%H:%M): ${1%% >*}"; fi' sh
echo "finished $(date)"
rm -f "$JOBS"
