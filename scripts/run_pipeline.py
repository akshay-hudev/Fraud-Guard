"""
Master Pipeline
Runs the full end-to-end ML pipeline:
  1. Generate synthetic data
  2. Preprocess & feature engineering
  3. Build graph
  4. Train baseline models
  5. Train GNN
  6. Compare & save results
"""

import os
import sys
import logging
import argparse
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("pipeline")


def step_generate():
    log.info("━━━ Step 1/5: Generating synthetic data ━━━")
    from data.generate_synthetic import generate_and_save
    generate_and_save("data/raw")


def step_preprocess():
    log.info("━━━ Step 2/5: Preprocessing & feature engineering ━━━")
    from src.data.preprocessor import FraudDataPreprocessor
    preprocessor = FraudDataPreprocessor("data/raw", "data/processed")
    return preprocessor.run()


def step_build_graph():
    log.info("━━━ Step 3/5: Building heterogeneous graph ━━━")
    from src.data.graph_builder import GraphBuilder
    builder = GraphBuilder("data/processed")
    return builder.build()


def step_train_baselines(splits: dict):
    log.info("━━━ Step 4/5: Training baseline models ━━━")
    import numpy as np
    from src.models.baseline import BaselineTrainer

    os.makedirs("models/baseline", exist_ok=True)
    trainer = BaselineTrainer("models/baseline")
    trainer.train_all(
        splits["X_train"], splits["y_train"],
        splits["X_val"],   splits["y_val"],
    )
    test_results = trainer.evaluate_test(splits["X_test"], splits["y_test"])

    # Save per-model results
    for name, metrics in test_results.items():
        with open(f"models/baseline/{name}_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

    return test_results


def step_train_gnn(data):
    log.info("━━━ Step 5/5: Training GNN ━━━")
    import torch
    from src.models.gnn import build_gnn_model
    from src.training.train_gnn import GNNTrainer

    os.makedirs("models/gnn", exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = build_gnn_model(data, hidden_dim=128, num_layers=2)
    trainer = GNNTrainer(
        model, data,
        model_dir = "models/gnn",
        lr        = 1e-3,
        epochs    = 100,
        patience  = 15,
        device    = device,
    )
    trainer.train()
    return trainer.evaluate_test()


def compare_models(baseline_results: dict, gnn_results: dict):
    from src.utils.metrics import compare_models, save_comparison
    all_results = {**baseline_results, "gnn_hgt": gnn_results}
    save_comparison(all_results, "models/comparison.json")
    df = compare_models(all_results)
    print("\n" + "="*65)
    print("MODEL COMPARISON")
    print("="*65)
    print(df.to_string())
    print("="*65)
    return all_results


def run_pipeline(skip_gnn: bool = False, skip_generate: bool = False):
    """Full pipeline runner."""
    if not skip_generate:
        step_generate()

    splits = step_preprocess()

    baseline_results = step_train_baselines(splits)

    gnn_results = {}
    if not skip_gnn:
        try:
            data = step_build_graph()
            gnn_results = step_train_gnn(data)
        except ImportError as e:
            log.warning("PyTorch Geometric not installed (%s). Skipping GNN step.", e)
            log.warning("Install with: pip install torch-geometric")

    compare_models(baseline_results, gnn_results)
    log.info("✅ Pipeline complete! All models saved in 'models/'")


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fraud Detection ML Pipeline")
    parser.add_argument("--skip-gnn",      action="store_true",
                        help="Skip GNN training (faster, for CPU-only machines)")
    parser.add_argument("--skip-generate", action="store_true",
                        help="Skip data generation (use existing raw data)")
    args = parser.parse_args()

    run_pipeline(skip_gnn=args.skip_gnn, skip_generate=args.skip_generate)
