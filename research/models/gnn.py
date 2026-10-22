"""Graph models for claim-node fraud classification.

All models share the same per-node-type input projection and the same
classification head, so differences come from the message-passing layers:

* ``sage``      : heterogeneous GraphSAGE (one SAGEConv per relation, summed);
                  neighbour aggregation 'mean' or 'sum' (tuned)
* ``rgcn``      : relational GCN (Schlichtkrull et al., 2018) with basis decomposition;
                  aggregation 'mean' or 'sum' (tuned)
* ``han``       : HAN (Wang et al., 2019) on claim nodes over meta-paths
                  claim-patient-claim and claim-doctor-claim
* ``simplehgn`` : Simple-HGN (Lv et al., 2021): GAT with edge-type embeddings in the
                  attention, residual attention, node residuals and L2-normalised output
* ``hgt``       : Heterogeneous Graph Transformer (Hu et al., 2020), PyG HGTConv
* ``mlp``       : no message passing; the claim's own features only (control)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HANConv, HeteroConv, HGTConv, RGCNConv, SAGEConv
from torch_geometric.utils import add_self_loops, scatter, softmax

GRAPH_KINDS = ["sage", "rgcn", "han", "simplehgn", "hgt"]


class Head(nn.Module):
    def __init__(self, hidden: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, h):
        return self.net(h).squeeze(-1)


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, layers: int, dropout: float):
        super().__init__()
        mods, d = [], in_dim
        for _ in range(layers):
            mods += [nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(dropout)]
            d = hidden
        self.body = nn.Sequential(*mods)
        self.head = Head(hidden, dropout)

    def forward(self, x):
        return self.head(self.body(x))


_HOMO_CACHE: dict[int, tuple] = {}


def homogeneous_structure(data: HeteroData):
    """Stacked edge_index / edge_type over node types in data.node_types order (cached per graph)."""
    key = id(data)
    if key in _HOMO_CACHE and _HOMO_CACHE[key][0] is data:
        return _HOMO_CACHE[key][1]
    offsets, total = {}, 0
    for t in data.node_types:
        offsets[t] = total
        total += data[t].num_nodes
    eis, ets = [], []
    for rel, et in enumerate(data.edge_types):
        src, _, dst = et
        ei = data[et].edge_index
        eis.append(torch.stack([ei[0] + offsets[src], ei[1] + offsets[dst]]))
        ets.append(torch.full((ei.size(1),), rel, dtype=torch.long))
    cache = (offsets, total, torch.cat(eis, 1), torch.cat(ets))
    _HOMO_CACHE[key] = (data, cache)
    return cache


class SimpleHGNConv(nn.Module):
    def __init__(self, in_dim, out_dim, num_etypes, edge_dim, heads, beta, negative_slope=0.2):
        super().__init__()
        self.heads, self.out_dim, self.beta, self.slope = heads, out_dim, beta, negative_slope
        self.W = nn.Linear(in_dim, heads * out_dim, bias=False)
        self.edge_emb = nn.Embedding(num_etypes, edge_dim)
        self.W_e = nn.Linear(edge_dim, heads * edge_dim, bias=False)
        self.a_src = nn.Parameter(torch.empty(1, heads, out_dim))
        self.a_dst = nn.Parameter(torch.empty(1, heads, out_dim))
        self.a_e = nn.Parameter(torch.empty(1, heads, edge_dim))
        self.res = nn.Linear(in_dim, heads * out_dim, bias=False)
        for p in (self.a_src, self.a_dst, self.a_e):
            nn.init.xavier_uniform_(p)

    def forward(self, x, edge_index, edge_type, prev_alpha=None):
        n = x.size(0)
        h = self.W(x).view(n, self.heads, self.out_dim)
        e = self.W_e(self.edge_emb(edge_type)).view(-1, self.heads, self.edge_emb.embedding_dim)
        src, dst = edge_index
        logit = F.leaky_relu((h * self.a_src).sum(-1)[src] + (h * self.a_dst).sum(-1)[dst] + (e * self.a_e).sum(-1), self.slope)
        alpha = softmax(logit, dst, num_nodes=n)
        if prev_alpha is not None:
            alpha = (1 - self.beta) * alpha + self.beta * prev_alpha
        out = scatter(h[src] * alpha.unsqueeze(-1), dst, dim=0, dim_size=n, reduce="sum")
        out = out + self.res(x).view(n, self.heads, self.out_dim)
        return out.reshape(n, -1), alpha


class GraphModel(nn.Module):
    def __init__(self, kind: str, data: HeteroData, hidden: int = 64, layers: int = 2, heads: int = 2,
                 dropout: float = 0.2, aggr: str = "mean", rgcn_bases: int = 4, simplehgn_beta: float = 0.05,
                 target: str = "claim"):
        super().__init__()
        assert kind in GRAPH_KINDS, kind
        self.kind, self.layers, self.dropout, self.hidden, self.target = kind, layers, dropout, hidden, target
        if kind == "han":
            in_dim = data[target].x.size(1)
            self.proj = nn.ModuleDict({target: nn.Linear(in_dim, hidden)})
            self.convs = nn.ModuleList([HANConv(hidden, hidden, data.metadata(), heads=heads, dropout=dropout)
                                        for _ in range(layers)])
        else:
            self.proj = nn.ModuleDict({t: nn.Linear(data[t].x.size(1), hidden) for t in data.node_types})
            if kind == "hgt":
                self.convs = nn.ModuleList([HGTConv(hidden, hidden, data.metadata(), heads=heads) for _ in range(layers)])
            elif kind == "sage":
                self.convs = nn.ModuleList([
                    HeteroConv({et: SAGEConv((hidden, hidden), hidden, aggr=aggr) for et in data.edge_types}, aggr="sum")
                    for _ in range(layers)])
            elif kind == "rgcn":
                n_rel = len(data.edge_types)
                self.convs = nn.ModuleList([RGCNConv(hidden, hidden, n_rel, num_bases=min(rgcn_bases, n_rel), aggr=aggr)
                                            for _ in range(layers)])
            elif kind == "simplehgn":
                n_rel = len(data.edge_types) + 1           # +1 for self-loops
                assert hidden % heads == 0
                self.convs = nn.ModuleList([SimpleHGNConv(hidden, hidden // heads, n_rel, 32, heads, simplehgn_beta)
                                            for _ in range(layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
        self.head = Head(hidden, dropout)

    def forward(self, data: HeteroData) -> torch.Tensor:
        """Return one logit per target node (claims by default)."""
        h = {t: F.relu(self.proj[t](data[t].x)) for t in self.proj}
        if self.kind in ("hgt", "sage", "han"):
            for conv, norm in zip(self.convs, self.norms):
                out = conv(h, data.edge_index_dict)
                missing = set(h) - {t for t, v in out.items() if v is not None}
                if missing:
                    raise ValueError(f"Node types without incoming messages: {sorted(missing)}")
                h = {t: norm(F.dropout(F.relu(out[t]), self.dropout, self.training) + h[t]) for t in h}
            return self.head(h[self.target])
        offsets, total, edge_index, edge_type = homogeneous_structure(data)
        x = torch.cat([h[t] for t in data.node_types])
        if self.kind == "simplehgn":
            edge_index, _ = add_self_loops(edge_index, num_nodes=total)
            edge_type = torch.cat([edge_type, torch.full((total,), len(data.edge_types), dtype=torch.long)])
        alpha = None
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            if self.kind == "rgcn":
                out = conv(x, edge_index, edge_type)
            else:
                out, alpha = conv(x, edge_index, edge_type, alpha)
            x = norm(F.dropout(F.relu(out), self.dropout, self.training) + x)
        if self.kind == "simplehgn":
            x = F.normalize(x, dim=-1)
        start = offsets[self.target]
        return self.head(x[start:start + data[self.target].num_nodes])
