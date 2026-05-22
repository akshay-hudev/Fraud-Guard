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
    classification_report, confusion_matrix,
)
from src.models.gnn import USE_INDUCTIVE_MODE
from src.utils.metrics import compute_binary_metrics

try:
    from torch_geometric.loader import NeighborLoader
except ImportError:  # pragma: no cover - optional runtime dependency
    NeighborLoader = None

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

    return compute_binary_metrics(y_true, y_pred, y_prob), y_true, y_pred, y_prob


def evaluate_loader(model, loader, device):
    model.eval()
    y_true_all, y_pred_all, y_prob_all = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch.x_dict, batch.edge_index_dict)
            batch_size = int(getattr(batch["claim"], "batch_size", logits.shape[0]))
            logits = logits[:batch_size]
            probs = F.softmax(logits, dim=-1)[:, 1]
            preds = logits.argmax(dim=-1)
            y_true_all.append(batch["claim"].y[:batch_size].cpu().numpy())
            y_pred_all.append(preds.cpu().numpy())
            y_prob_all.append(probs.cpu().numpy())

    y_true = np.concatenate(y_true_all) if y_true_all else np.array([])
    y_pred = np.concatenate(y_pred_all) if y_pred_all else np.array([])
    y_prob = np.concatenate(y_prob_all) if y_prob_all else np.array([])
    return compute_binary_metrics(y_true, y_pred, y_prob), y_true, y_pred, y_prob


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
        use_inductive_mode: bool = USE_INDUCTIVE_MODE,
    ):
        self.model      = model.to(device)
        self.use_inductive_mode = bool(use_inductive_mode and NeighborLoader is not None)
        if use_inductive_mode and NeighborLoader is None:
            log.warning("NeighborLoader unavailable; falling back to full-graph training.")
        self.data       = data if self.use_inductive_mode else data.to(device)
        self.model_dir  = model_dir
        self.epochs     = epochs
        self.patience   = patience
        self.device     = device

        os.makedirs(model_dir, exist_ok=True)

        # Class weights (handle imbalance)
        train_mask = data["claim"].train_mask
        y_all = data["claim"].y[train_mask] if train_mask is not None else data["claim"].y
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
        self.num_neighbors = {
            edge_type: [10, 5]
            for edge_type in data.edge_types
        }
        self.train_loader = self._build_loader(data["claim"].train_mask, shuffle=True)
        self.val_loader = self._build_loader(data["claim"].val_mask, shuffle=False)
        self.test_loader = self._build_loader(data["claim"].test_mask, shuffle=False)

    def _build_loader(self, mask, shuffle: bool):
        if not self.use_inductive_mode:
            return None
        return NeighborLoader(
            self.data,
            num_neighbors=self.num_neighbors,
            input_nodes=("claim", mask),
            batch_size=32,
            shuffle=shuffle,
        )

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

    def _train_step_inductive(self) -> float:
        self.model.train()
        losses = []
        for batch in self.train_loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            logits = self.model(batch.x_dict, batch.edge_index_dict)
            batch_size = int(getattr(batch["claim"], "batch_size", logits.shape[0]))
            loss = self.criterion(
                logits[:batch_size],
                batch["claim"].y[:batch_size],
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            losses.append(float(loss.item()))
        return float(np.mean(losses)) if losses else 0.0

    def train(self) -> dict:
        best_val_f1   = -1.0
        patience_ctr  = 0
        best_epoch    = 0

        for epoch in range(1, self.epochs + 1):
            train_loss = (
                self._train_step_inductive()
                if self.use_inductive_mode else self._train_step()
            )
            self.scheduler.step()

            if self.use_inductive_mode:
                val_metrics, *_ = evaluate_loader(self.model, self.val_loader, self.device)
            else:
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
        if self.use_inductive_mode:
            metrics, y_true, y_pred, y_prob = evaluate_loader(
                self.model, self.test_loader, self.device
            )
        else:
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

        with open(f"{self.model_dir}/test_predictions.json", "w") as f:
            json.dump({
                "y_true": y_true.tolist(),
                "y_pred": y_pred.tolist(),
                "y_prob": y_prob.tolist(),
            }, f, indent=2)

        return metrics

    def get_embeddings(self) -> np.ndarray:
        """Return claim embeddings for visualization (UMAP/t-SNE)."""
        self.model.eval()
        data = self.data.to(self.device) if self.use_inductive_mode else self.data
        with torch.no_grad():
            emb = self.model.get_embeddings(
                data.x_dict, data.edge_index_dict
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

    try:
        data = torch.load(data_path, weights_only=False)
    except TypeError:
        data = torch.load(data_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = build_gnn_model(data, hidden_dim=128, num_layers=2)

    trainer = GNNTrainer(model, data, model_dir="models/gnn",
                         lr=1e-3, epochs=150, patience=20, device=device)
    trainer.train()
    trainer.evaluate_test()
