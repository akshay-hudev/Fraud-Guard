#!/usr/bin/env bash
# E-4 evaluation job of the third round (analysis_plan_round3.md): wait until every tuning job of the regime has
# written its trials, merge them once (the first job to take the lock), then evaluate one world.
#
#   bash paper/hiss_review/round3_regime_eval.sh <shift> <density> <seed>
set -euo pipefail
cd "$(dirname "$0")/../.."
SHIFT=$1 DENS=$2 SEED=$3
TUNE=results/tuning_regime_d${SHIFT}_s${DENS}
LOGS=results/round3_logs
PY=".venv-research/bin/python -m research.experiments"
N_TRIAL_FILES=6                                  # tab, SAGE, RGCN, HAN, SimpleHGN, HGT
until [ "$(ls "$TUNE"/trials_*.json 2>/dev/null | wc -l)" -ge "$N_TRIAL_FILES" ]; do sleep 60; done
if mkdir "$LOGS/merge_d${SHIFT}_s${DENS}.lock" 2>/dev/null; then
  $PY.tune --merge --ring-shift "$SHIFT" --ring-density "$DENS" --out "$TUNE" > "$LOGS/merge_d${SHIFT}_s${DENS}.log" 2>&1
  touch "$LOGS/merged_d${SHIFT}_s${DENS}.log"
fi
until [ -f "$LOGS/merged_d${SHIFT}_s${DENS}.log" ]; do sleep 10; done
$PY.synthetic --shifts "$SHIFT" --densities "$DENS" --seeds "$SEED" --hps "$TUNE/best_hyperparameters.json" \
  --out results/synthetic_regimetuned --group all
