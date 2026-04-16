"""
Graph Neural Network (GNN) for Healthcare Fraud Detection
Architecture: Heterogeneous Graph Attention Network (HAN / HGT-style)
Task: Node classification on 'claim' nodes (Fraud / Not Fraud)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import (
    HGTConv, Linear,
    GATConv, SAGEConv,
)
import logging

log = logging.getLogger(__name__)


# ── HGT-based Fraud Detector ───────────────────────────────────────────────────

class HGTFraudDetector(nn.Module):
    """
    Heterogeneous Graph Transformer (HGT) for claim-level fraud detection.

    Layers:
      1. Per-node-type linear projection (input → hidden)
      2. N × HGTConv layers (message passing across different edge types)
      3. Final MLP classifier on 'claim' node embeddings
    """

    def __init__(
        self,
        metadata: tuple,
        hidden_dim: int  = 128,
        out_dim: int     = 2,
        num_heads: int   = 4,
        num_layers: int  = 2,
        dropout: float   = 0.3,
        node_feature_dims: dict = None,  # {node_type: input_feat_dim}
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dropout    = dropout

        # Input projections (each node type may have different feature dim)
        node_types = metadata[0]
        self.lin_dict = nn.ModuleDict()
        for node_type in node_types:
            in_dim = node_feature_dims.get(node_type, hidden_dim) if node_feature_dims else hidden_dim
            self.lin_dict[node_type] = Linear(in_dim, hidden_dim)

        # HGT layers
        self.convs = nn.ModuleList([
            HGTConv(hidden_dim, hidden_dim, metadata, heads=num_heads)
            for _ in range(num_layers)
        ])

        # Batch normalisation per layer
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])

        # Classifier MLP
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, out_dim),
        )

    def forward(self, x_dict: dict, edge_index_dict: dict) -> torch.Tensor:
        # Project all node types into shared hidden space
        h_dict = {
            node_type: F.elu(self.lin_dict[node_type](x))
            for node_type, x in x_dict.items()
            if node_type in self.lin_dict
        }

        # Message passing with error handling for missing edge types
        try:
            for conv, norm in zip(self.convs, self.norms):
                h_dict_new = conv(h_dict, edge_index_dict)
                # Residual + norm for claim nodes
                for node_type in h_dict_new:
                    h = h_dict_new[node_type]
                    if node_type in h_dict and h_dict[node_type].shape == h.shape:
                        h = h + h_dict[node_type]   # residual
                    h_dict_new[node_type] = norm(F.dropout(h, p=self.dropout, training=self.training))
                h_dict = h_dict_new
        except KeyError as e:
            # Fallback: use initial projections if HGTConv fails
            import logging
            logging.warning(f"HGTConv failed with KeyError {e}, using initial projections...")
            h_dict = h_dict

        # Classify claim nodes
        claim_emb = h_dict["claim"]
        return self.classifier(claim_emb)

    def get_embeddings(self, x_dict: dict, edge_index_dict: dict) -> torch.Tensor:
        """Return claim node embeddings (before classifier) for visualization."""
        h_dict = {
            node_type: F.elu(self.lin_dict[node_type](x))
            for node_type, x in x_dict.items()
            if node_type in self.lin_dict
        }
        for conv, norm in zip(self.convs, self.norms):
            h_dict_new = conv(h_dict, edge_index_dict)
            for node_type in h_dict_new:
                h = h_dict_new[node_type]
                if node_type in h_dict and h_dict[node_type].shape == h.shape:
                    h = h + h_dict[node_type]
                h_dict_new[node_type] = norm(F.dropout(h, p=self.dropout, training=self.training))
            h_dict = h_dict_new
        return h_dict["claim"]


# ── Simpler homogeneous fallback ───────────────────────────────────────────────

class SimpleGNN(nn.Module):
    """
    Fallback homogeneous GNN (GraphSAGE + GAT) in case the dataset is
    converted to a single-type graph.
    """

    def __init__(self, in_dim: int, hidden_dim: int = 128, out_dim: int = 2,
                 dropout: float = 0.3):
        super().__init__()
        self.conv1 = SAGEConv(in_dim,     hidden_dim)
        self.conv2 = GATConv (hidden_dim, hidden_dim, heads=4, concat=False)
        self.conv3 = SAGEConv(hidden_dim, hidden_dim)
        self.clf   = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, out_dim),
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = F.elu(self.norm1(self.conv1(x, edge_index)))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.elu(self.norm2(self.conv2(h, edge_index)))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.conv3(h, edge_index)
        return self.clf(h)


# ── Model factory ──────────────────────────────────────────────────────────────

def build_gnn_model(data: HeteroData, hidden_dim: int = 128,
                    num_layers: int = 2) -> HGTFraudDetector:
    """Instantiate the HGT model from a HeteroData object."""
    node_feature_dims = {
        node_type: data[node_type].x.shape[1]
        for node_type in data.node_types
        if hasattr(data[node_type], "x") and data[node_type].x is not None
    }
    model = HGTFraudDetector(
        metadata          = data.metadata(),
        hidden_dim        = hidden_dim,
        num_layers        = num_layers,
        node_feature_dims = node_feature_dims,
    )
    log.info("GNN built: %s", model)
    log.info("Params: %s", sum(p.numel() for p in model.parameters() if p.requires_grad))
    return model
