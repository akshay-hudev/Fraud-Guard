"""
GNN Training Pipeline
Full training loop with:
  - Weighted cross-entropy (class imbalance)
  - Early stopping on val F1
  - Checkpoint saving
  - Training curves
"""

import os
import json
import logging
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report, confusion_matrix,
)

log = logging.getLogger(__name__)


# ── Metric helpers ─────────────────────────────────────────────────────────────

def evaluate(model, data, mask, device):
    model.eval()
    with torch.no_grad():
        logits = model(data.x_dict, data.edge_index_dict)
        probs  = F.softmax(logits, dim=-1)[:, 1]
        preds  = logits.argmax(dim=-1)

        y_true = data["claim"].y[mask].cpu().numpy()
        y_pred = preds[mask].cpu().numpy()
        y_prob = probs[mask].cpu().numpy()

    return {
        "accuracy":  round(accuracy_score (y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":    round(recall_score   (y_true, y_pred, zero_division=0), 4),
        "f1":        round(f1_score       (y_true, y_pred, zero_division=0), 4),
        "roc_auc":   round(roc_auc_score  (y_true, y_prob), 4),
    }, y_true, y_pred, y_prob


# ── Trainer ────────────────────────────────────────────────────────────────────

class GNNTrainer:

    def __init__(
        self,
        model,
        data,
        model_dir:       str   = "models/gnn",
        lr:              float = 1e-3,
        weight_decay:    float = 1e-4,
        epochs:          int   = 100,
        patience:        int   = 15,
        device:          str   = "cpu",
    ):
        self.model      = model.to(device)
        self.data       = data.to(device)
        self.model_dir  = model_dir
        self.epochs     = epochs
        self.patience   = patience
        self.device     = device

        os.makedirs(model_dir, exist_ok=True)

        # Class weights (handle imbalance)
        y_all = data["claim"].y
        n_neg = (y_all == 0).sum().item()
        n_pos = (y_all == 1).sum().item()
        total = n_neg + n_pos
        weight = torch.tensor(
            [total / (2 * n_neg), total / (2 * n_pos)], dtype=torch.float
        ).to(device)
        log.info("Class weights: %s", weight)
        self.criterion = torch.nn.CrossEntropyLoss(weight=weight)

        self.optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs, eta_min=1e-5)

        self.history = {"train_loss": [], "val_f1": [], "val_loss": []}

    def _train_step(self) -> float:
        self.model.train()
        self.optimizer.zero_grad()
        logits = self.model(self.data.x_dict, self.data.edge_index_dict)
        mask   = self.data["claim"].train_mask
        loss   = self.criterion(logits[mask], self.data["claim"].y[mask])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        return loss.item()

    def train(self) -> dict:
        best_val_f1   = -1.0
        patience_ctr  = 0
        best_epoch    = 0

        for epoch in range(1, self.epochs + 1):
            train_loss = self._train_step()
            self.scheduler.step()

            val_metrics, *_ = evaluate(
                self.model, self.data, self.data["claim"].val_mask, self.device
            )

            self.history["train_loss"].append(train_loss)
            self.history["val_f1"].append(val_metrics["f1"])

            if epoch % 10 == 0 or epoch == 1:
                log.info(
                    "Epoch %3d | loss: %.4f | val_f1: %.4f | val_auc: %.4f",
                    epoch, train_loss, val_metrics["f1"], val_metrics["roc_auc"],
                )

            # Early stopping
            if val_metrics["f1"] > best_val_f1:
                best_val_f1 = val_metrics["f1"]
                patience_ctr = 0
                best_epoch   = epoch
                torch.save(self.model.state_dict(), f"{self.model_dir}/best_model.pt")
            else:
                patience_ctr += 1
                if patience_ctr >= self.patience:
                    log.info("Early stopping at epoch %d (best val F1: %.4f @ epoch %d)",
                             epoch, best_val_f1, best_epoch)
                    break

        # Save history
        with open(f"{self.model_dir}/training_history.json", "w") as f:
            json.dump(self.history, f, indent=2)

        log.info("✅ Training complete. Best val F1: %.4f", best_val_f1)
        return self.history

    def evaluate_test(self) -> dict:
        # Load best checkpoint
        self.model.load_state_dict(
            torch.load(f"{self.model_dir}/best_model.pt", map_location=self.device)
        )
        metrics, y_true, y_pred, y_prob = evaluate(
            self.model, self.data, self.data["claim"].test_mask, self.device
        )
        log.info("TEST metrics: %s", metrics)
        print("\n" + "="*55)
        print("GNN TEST RESULTS")
        print("="*55)
        print(classification_report(y_true, y_pred, target_names=["Legit", "Fraud"]))
        print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))

        with open(f"{self.model_dir}/test_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        return metrics

    def get_embeddings(self) -> np.ndarray:
        """Return claim embeddings for visualization (UMAP/t-SNE)."""
        self.model.eval()
        with torch.no_grad():
            emb = self.model.get_embeddings(
                self.data.x_dict, self.data.edge_index_dict
            )
        return emb.cpu().numpy()


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    logging.basicConfig(level=logging.INFO)

    from src.models.gnn import build_gnn_model

    data_path = "data/processed/hetero_graph.pt"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Graph not found. Run graph_builder.py first.")

    data   = torch.load(data_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = build_gnn_model(data, hidden_dim=128, num_layers=2)

    trainer = GNNTrainer(model, data, model_dir="models/gnn",
                         lr=1e-3, epochs=150, patience=20, device=device)
    trainer.train()
    trainer.evaluate_test()
