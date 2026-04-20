"""
GNN Training Entry Point
Run this after the baseline pipeline:
  python scripts/train_gnn.py

Requirements:
  pip install torch torch-geometric
  pip install torch-scatter torch-sparse --find-links https://data.pyg.org/whl/torch-2.2.0+cpu.html
"""

import os
import sys
import logging
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def main():
    # ── Step 1: Build graph (if not already done) ──────────────────────────────
    graph_path = "data/processed/hetero_graph.pt"
    if not os.path.exists(graph_path):
        log.info("Building heterogeneous graph...")
        from src.data.graph_builder import GraphBuilder
        builder = GraphBuilder("data/processed")
        data = builder.build()
        log.info("Graph: %s", data)
    else:
        log.info("Loading existing graph from %s", graph_path)
        data = torch.load(graph_path)

    log.info("Graph summary:")
    log.info("  Node types  : %s", data.node_types)
    log.info("  Edge types  : %s", [str(e) for e in data.edge_types])
    for ntype in data.node_types:
        n = data[ntype].x.shape[0] if hasattr(data[ntype], 'x') and data[ntype].x is not None else 0
        log.info("  %-12s : %d nodes", ntype, n)

    # ── Step 2: Build model ─────────────────────────────────────────────────────
    from src.models.gnn import build_gnn_model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Device: %s", device)

    model = build_gnn_model(data, hidden_dim=128, num_layers=2)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("Model parameters: %s", f"{total_params:,}")

    # ── Step 3: Train ───────────────────────────────────────────────────────────
    from src.training.train_gnn import GNNTrainer
    os.makedirs("models/gnn", exist_ok=True)

    trainer = GNNTrainer(
        model     = model,
        data      = data,
        model_dir = "models/gnn",
        lr        = 1e-3,
        weight_decay = 1e-4,
        epochs    = 100,
        patience  = 15,
        device    = device,
    )

    history = trainer.train()
    test_metrics = trainer.evaluate_test()

    log.info("Test metrics: %s", test_metrics)

    # ── Step 4: Save embeddings ─────────────────────────────────────────────────
    import numpy as np
    embeddings = trainer.get_embeddings()
    np.save("data/processed/gnn_embeddings.npy", embeddings)
    log.info("Embeddings saved: shape=%s", embeddings.shape)

    # ── Step 5: Update comparison ───────────────────────────────────────────────
    import json
    comp_path = "models/comparison.json"
    if os.path.exists(comp_path):
        with open(comp_path) as f:
            comparison = json.load(f)
    else:
        comparison = {}

    comparison["gnn_hgt"] = test_metrics
    with open(comp_path, "w") as f:
        json.dump(comparison, f, indent=2)

    log.info("✅ GNN training complete. Results saved to models/comparison.json")

    # Print comparison table
    print("\n" + "="*60)
    print("FINAL MODEL COMPARISON")
    print("="*60)
    import pandas as pd
    df = pd.DataFrame(comparison).T.round(4)
    print(df[["accuracy", "f1", "roc_auc"]].to_string())
    print("="*60)


if __name__ == "__main__":
    main()
