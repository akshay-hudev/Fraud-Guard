"""Prepare deterministic runtime artifacts required by the training tests."""

from pathlib import Path
import sys

import pytest

TRAINING_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = TRAINING_ROOT.parent
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(TRAINING_ROOT))


@pytest.fixture(scope="session", autouse=True)
def prepare_training_artifacts(tmp_path_factory):
    required = [
        Path("data/processed/scaler.pkl"),
        Path("data/processed/label_encoders.pkl"),
        Path("models/baseline/logistic_regression.pkl"),
        Path("models/baseline/random_forest.pkl"),
        Path("models/baseline/gradient_boosting.pkl"),
    ]
    if all(path.exists() for path in required):
        return

    from src.data.preprocessor import FraudDataPreprocessor
    from src.models.baseline import BaselineTrainer

    splits = FraudDataPreprocessor("data/raw", "data/processed").run()
    trainer = BaselineTrainer(
        model_dir="models/baseline",
        log_dir=str(tmp_path_factory.mktemp("baseline-test-logs")),
    )
    trainer.train_all(
        splits["X_train"],
        splits["y_train"],
        splits["X_val"],
        splits["y_val"],
        tune=False,
    )
