"""
Real-world experiment entry point (Kaggle Healthcare Provider Fraud Detection Analysis).

The earlier implementation in this file computed provider aggregates on the full
dataset before splitting, broadcast provider-level labels to claims (so the same
provider appeared in train and test), and silently fell back to the synthetic
data while labelling the output "real world". It has been replaced by the
leakage-safe, provider-level pipeline in ``research/experiments/kaggle_exp.py``;
this script only forwards to it.

Data: place the Kaggle CSVs in data/external/kaggle_provider_fraud/ (see
research/README.md). There is no synthetic fallback.

Usage:
    python training/train_real_world.py --stage tune
    python training/train_real_world.py --stage merge
    python training/train_real_world.py --stage eval
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    if not (REPO_ROOT / "data" / "external" / "kaggle_provider_fraud" / "Train-1542865627584.csv").exists():
        raise SystemExit("Kaggle CSVs not found in data/external/kaggle_provider_fraud/ (no synthetic fallback).")
    sys.path.insert(0, str(REPO_ROOT))
    sys.argv[0] = "research.experiments.kaggle_exp"
    runpy.run_module("research.experiments.kaggle_exp", run_name="__main__")


if __name__ == "__main__":
    main()
