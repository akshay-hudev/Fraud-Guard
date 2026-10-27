#!/usr/bin/env bash
# Third experimental round (paper/hiss_review/analysis_plan_round3.md): every job, run with PARALLEL workers.
# Synthetic jobs skip worlds whose run files exist, so the script can be restarted; tuning and real-data jobs rerun.
#
#   nohup bash paper/hiss_review/run_round3.sh > results/round3_logs/queue.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/../.."
PARALLEL=${PARALLEL:-4}
LOGS=results/round3_logs
mkdir -p "$LOGS"
ENV="RESEARCH_THREADS=2 OMP_NUM_THREADS=2"
PY=".venv-research/bin/python -m research.experiments"
JOBS=$(mktemp)

# E-4a. Per-regime tuning at delta = 0, s = 0.1 and 0.9 (development seeds 901/902), longest jobs first.
for m in HGT SimpleHGN HAN RGCN SAGE; do for dens in 0.1 0.9; do
  echo "$ENV $PY.tune --ring-shift 0.0 --ring-density $dens --methods $m --tag $m --out results/tuning_regime_d0.0_s$dens > $LOGS/E4_tune_s${dens}_$m.log 2>&1"
done; done >> "$JOBS"
for dens in 0.1 0.9; do
  echo "$ENV $PY.tune --ring-shift 0.0 --ring-density $dens --methods LR-CA RF-CA GB-C GB-CA GB-CAG MLP-C --tag tab --out results/tuning_regime_d0.0_s$dens > $LOGS/E4_tune_s${dens}_tab.log 2>&1"
done >> "$JOBS"
# E-7. Repeated rotations on the public data for RGCN, Simple-HGN and the hybrids.
for rep in 1 2 3 4; do for fold in 0 1 2 3 4; do
  echo "$ENV $PY.kaggle_exp --stage eval --repeat $rep --folds $fold --methods RGCN SimpleHGN SAGE+G HGT+G --tag extra > $LOGS/E7_r${rep}_f$fold.log 2>&1"
done; done >> "$JOBS"
# E-2. Feature shift at ring volume 4: delta = 1 and 2, seeds 1-6, all methods, main hyperparameters.
for shift in 1.0 2.0; do for dens in 0.1 0.5 0.9; do for seed in 1 2 3 4 5 6; do
  echo "$ENV $PY.synthetic --shifts $shift --densities $dens --seeds $seed --ring-claims-per-month 4 --out results/synthetic_lowvolume_shift --group all > $LOGS/E2_d${shift}_s${dens}_g$seed.log 2>&1"
done; done; done >> "$JOBS"
# E-3. Training seeds 1 and 2 for GB-CAG, SAGE, HGT and the hybrids (delta = 0, default volume, seeds 1-6).
for ms in 1 2; do for dens in 0.1 0.5 0.9; do for seed in 1 2 3 4 5 6; do
  echo "$ENV $PY.synthetic --shifts 0.0 --densities $dens --seeds $seed --model-seed $ms --methods GB-CAG SAGE HGT SAGE+AG HGT+AG --out results/synthetic_modelseeds --group seedrep > $LOGS/E3_m${ms}_s${dens}_g$seed.log 2>&1"
done; done; done >> "$JOBS"
# E-4b. Evaluation with the per-regime configurations (each job waits for its regime's tuning and merge).
for dens in 0.1 0.9; do for seed in 1 2 3 4 5 6; do
  echo "$ENV bash paper/hiss_review/round3_regime_eval.sh 0.0 $dens $seed > $LOGS/E4_eval_s${dens}_g$seed.log 2>&1"
done; done >> "$JOBS"

echo "$(wc -l < "$JOBS") jobs, $PARALLEL in parallel, started $(date)"
# Each job is passed as one argument (BSD xargs -I limits replacements to 255 bytes).
tr '\n' '\0' < "$JOBS" | xargs -0 -n 1 -P "$PARALLEL" \
  sh -c 'if eval "$1"; then echo "done $(date +%H:%M): ${1%% >*}"; else echo "FAILED $(date +%H:%M): ${1%% >*}"; fi' sh
echo "finished $(date)"
rm -f "$JOBS"
