#!/usr/bin/env bash
# Helpers for run_round4.sh (analysis_plan_round4.md): tuning merges that wait for their trial files, and jobs
# that wait for a merge.
#
#   bash paper/hiss_review/round4_job.sh merge-main | merge-lowvol | merge-kaggle
#   bash paper/hiss_review/round4_job.sh after <main|lowvol|kaggle> <command ...>
set -euo pipefail
cd "$(dirname "$0")/../.."
LOGS=results/round4_logs
PY=".venv-research/bin/python -m research.experiments"
wait_files() { for f in "$@"; do until [ -f "$f" ]; do sleep 60; done; done; }
case "$1" in
  merge-main)
    wait_files results/tuning/trials_BWGNN.json results/tuning/trials_CARE.json
    $PY.tune --merge > "$LOGS/merge_main.log" 2>&1 && touch "$LOGS/merged_main.log" ;;
  merge-lowvol)
    wait_files results/tuning_lowvolume/trials_BWGNN.json results/tuning_lowvolume/trials_CARE.json
    $PY.tune --merge --ring-claims-per-month 4 --out results/tuning_lowvolume > "$LOGS/merge_lowvol.log" 2>&1 \
      && touch "$LOGS/merged_lowvol.log" ;;
  merge-kaggle)
    wait_files results/kaggle/tuning/trials_fraud.json
    $PY.kaggle_exp --stage merge > "$LOGS/merge_kaggle.log" 2>&1 && touch "$LOGS/merged_kaggle.log" ;;
  after)
    marker=$2; shift 2
    wait_files "$LOGS/merged_$marker.log"
    "$@" ;;
  *) echo "unknown subcommand $1" >&2; exit 2 ;;
esac
