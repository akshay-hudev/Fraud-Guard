#!/usr/bin/env bash
# Fourth experimental round (paper/hiss_review/analysis_plan_round4.md): every job, run with PARALLEL workers.
# Synthetic jobs skip worlds whose run files exist; tuning and real-data jobs rerun.
#
#   nohup bash paper/hiss_review/run_round4.sh > results/round4_logs/queue.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/../.."
PARALLEL=${PARALLEL:-4}
LOGS=results/round4_logs
mkdir -p "$LOGS"
ENV="RESEARCH_THREADS=2 OMP_NUM_THREADS=2"
PY=".venv-research/bin/python -m research.experiments"
AFTER="bash paper/hiss_review/round4_job.sh after"
JOBS=$(mktemp)
FRAUD="--methods BWGNN CARE-GNN --group fraudgnn"

# A. Tuning of the fraud-specialised GNNs: main protocol, low-volume protocol, public data (dispatched first).
for m in BWGNN CARE; do
  name=$([ "$m" = CARE ] && echo CARE-GNN || echo BWGNN)
  echo "$ENV $PY.tune --methods $name --tag $m > $LOGS/A_tune_$m.log 2>&1"
  echo "$ENV $PY.tune --ring-claims-per-month 4 --methods $name --tag $m --out results/tuning_lowvolume > $LOGS/A_tune_low_$m.log 2>&1"
done >> "$JOBS"
echo "$ENV $PY.kaggle_exp --stage tune --methods BWGNN CARE-GNN --tag fraud > $LOGS/A_tune_kaggle.log 2>&1" >> "$JOBS"
# Merges (each waits for its trial files).
for m in main lowvol kaggle; do echo "bash paper/hiss_review/round4_job.sh merge-$m"; done >> "$JOBS"
# Per-rotation (nested) selection needs no merge.
for fold in 0 1 2 3 4; do
  echo "$ENV $PY.kaggle_exp --stage nested --nested-gnns --folds $fold --methods BWGNN CARE-GNN --tag gnn_fraud > $LOGS/A_nested_f$fold.log 2>&1"
done >> "$JOBS"
# A. Synthetic runs of the fraud GNNs: main grid (one job per regime, seeds 1-12), low volume, volumes 8 and 2.
for shift in 0.0 1.0 2.0; do for dens in 0.1 0.5 0.9; do
  echo "$ENV $AFTER main $PY.synthetic --shifts $shift --densities $dens --seeds 1 2 3 4 5 6 7 8 9 10 11 12 $FRAUD --out results/synthetic > $LOGS/A_main_d${shift}_s$dens.log 2>&1"
done; done >> "$JOBS"
for dens in 0.1 0.5 0.9; do
  echo "$ENV $AFTER main $PY.synthetic --shifts 0.0 --densities $dens --seeds 1 2 3 4 5 6 --ring-claims-per-month 4 $FRAUD --out results/synthetic_lowvolume > $LOGS/A_low_s$dens.log 2>&1"
  echo "$ENV $AFTER lowvol $PY.synthetic --shifts 0.0 --densities $dens --seeds 1 2 3 4 5 6 --ring-claims-per-month 4 --hps results/tuning_lowvolume/best_hyperparameters.json $FRAUD --out results/synthetic_lowvolume_retuned > $LOGS/A_lowrt_s$dens.log 2>&1"
  for vol in 8 2; do
    echo "$ENV $AFTER main $PY.synthetic --shifts 0.0 --densities $dens --seeds 1 2 3 4 5 6 --ring-claims-per-month $vol $FRAUD --out results/synthetic_volume$vol > $LOGS/A_vol${vol}_s$dens.log 2>&1"
  done
done >> "$JOBS"
# A. Public data: the five rotations and the four repeated rotations.
for fold in 0 1 2 3 4; do
  echo "$ENV $AFTER kaggle $PY.kaggle_exp --stage eval --folds $fold --methods BWGNN CARE-GNN --tag fraud > $LOGS/A_kaggle_f$fold.log 2>&1"
done >> "$JOBS"
for rep in 1 2 3 4; do for fold in 0 1 2 3 4; do
  echo "$ENV $AFTER kaggle $PY.kaggle_exp --stage eval --repeat $rep --folds $fold --methods BWGNN CARE-GNN --tag fraud > $LOGS/A_rep_r${rep}_f$fold.log 2>&1"
done; done >> "$JOBS"
# B. Ring shapes and sizes (delta = 0, seeds 1-6, all 16 methods), full and half per-member volume.
for seed in 1 2 3 4 5 6; do
  for spec in "hub 4 12 16" "hub 4 12 8" "pairs 8 24 32" "pairs 8 24 16" "block 2 6 8" "block 2 6 4" "block 8 24 32" "block 8 24 16"; do
    set -- $spec
    echo "$ENV $AFTER main $PY.synthetic --shifts 0.0 --densities 0.5 --seeds $seed --ring-shape $1 --ring-doctors $2 --ring-patients $3 --ring-claims-per-month $4 --out results/synthetic_shapes --group all > $LOGS/B_${1}_${2}x${3}_v${4}_g$seed.log 2>&1"
  done
done >> "$JOBS"

echo "$(wc -l < "$JOBS") jobs, $PARALLEL in parallel, started $(date)"
# Each job is passed as one argument (BSD xargs -I limits replacements to 255 bytes).
tr '\n' '\0' < "$JOBS" | xargs -0 -n 1 -P "$PARALLEL" \
  sh -c 'if eval "$1"; then echo "done $(date +%H:%M): ${1%% >*}"; else echo "FAILED $(date +%H:%M): ${1%% >*}"; fi' sh
echo "finished $(date)"
rm -f "$JOBS"
