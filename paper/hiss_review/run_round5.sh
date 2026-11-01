#!/usr/bin/env bash
# Fifth experimental round (paper/hiss_review/analysis_plan_round5.md): every job, run with PARALLEL workers.
# Finished tuning candidates and evaluation worlds are skipped on restart.
#
#   nohup bash paper/hiss_review/run_round5.sh > results/round5_logs/queue.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/../.."
PARALLEL=${PARALLEL:-4}
LOGS=results/round5_logs
mkdir -p "$LOGS"
export RESEARCH_THREADS=${RESEARCH_THREADS:-1} OMP_NUM_THREADS=${RESEARCH_THREADS:-1}
PY=".venv-research/bin/python -m research.experiments"
LOW=results/tuning_lowvolume/best_hyperparameters.json

# C and B first: minutes, not hours.
{
  for seed in 1 2; do
    echo "$PY.synthetic --shifts 0.0 --densities 0.1 --seeds 7 8 9 10 11 12 13 14 15 16 17 18 --ring-claims-per-month 4 --methods GB-CAG SAGE+AG --hps $LOW --model-seed $seed --out results/synthetic_confirm_seeds --group confirm > $LOGS/C_seed$seed.log 2>&1"
  done
  for seed in 1 2 3 4 5 6; do
    echo "$PY.expanded --stage closure --seeds $seed > $LOGS/B_g$seed.log 2>&1"
  done
} | xargs -P "$PARALLEL" -I{} bash -c "{}"

# A. Wide search: one job per regime and method, slowest last so the fast ones free workers early.
for method in GB-CAG SAGE RGCN HGT SimpleHGN; do
  for regime in v4_s0.5 v4_s0.9 v16_s0.9; do
    echo "$PY.expanded --stage tune --regime $regime --methods $method > $LOGS/A_tune_${regime}_$method.log 2>&1"
  done
done | xargs -P "$PARALLEL" -I{} bash -c "{}"

.venv-research/bin/python -m research.experiments.expanded --stage merge > "$LOGS/A_merge.log" 2>&1

for regime in v4_s0.5 v4_s0.9 v16_s0.9; do
  for method in GB-CAG SAGE RGCN SimpleHGN HGT; do
    echo "$PY.expanded --stage eval --regime $regime --methods $method > $LOGS/A_eval_${regime}_$method.log 2>&1"
  done
done | xargs -P "$PARALLEL" -I{} bash -c "{}"
echo "round 5 finished $(date)"
