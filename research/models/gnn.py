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
* ``bwgnn``     : BWGNN (Tang et al., 2022), heterogeneous variant: Beta-wavelet band-pass filters of order C
                  on each claim-claim relation (shared patient, shared doctor), summed over relations
* ``care``      : CARE-GNN (Dou et al., 2020) on the same relations: label-aware similarity, top-p neighbour
                  selection per relation with thresholds adapted each epoch, threshold-weighted aggregation and
                  an auxiliary similarity loss
* ``mlp``       : no message passing; the claim's own features only (control)

``han``, ``bwgnn`` and ``care`` operate on claim-only graphs whose claim inputs include the attributes of the
claim's patient, doctor and hospital (``synth.graph.build_metapath_graphs``), the setting these fraud models were
designed for (one node type, several relations). They keep their published layer structure and share the input
projection and classification head with the other models.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HANConv, HeteroConv, HGTConv, RGCNConv, SAGEConv
from torch_geometric.utils import add_self_loops, scatter, softmax

GRAPH_KINDS = ["sage", "rgcn", "han", "simplehgn", "hgt", "bwgnn", "care"]
TARGET_ONLY_KINDS = ("han", "bwgnn", "care")       # claim-only graphs (one node type, several relations)


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


def beta_wavelet_coefficients(order: int) -> list[list[float]]:
    """Polynomial coefficients (in powers of the normalised Laplacian L) of the order+1 Beta-wavelet filters.

    Filter i is (L/2)^i (1 - L/2)^(order-i) / B(i+1, order+1-i), as in Tang et al. (2022).
    """
    from math import comb
    from scipy.special import beta as beta_fn
    out = []
    for i in range(order + 1):
        coef = [0.0] * (order + 1)
        for j in range(order - i + 1):          # (1 - L/2)^(order-i) = sum_j comb * (-L/2)^j
            coef[i + j] += comb(order - i, j) * (-0.5) ** j * 0.5 ** i
        out.append([c / beta_fn(i + 1, order + 1 - i) for c in coef])
    return out


_LAPLACIAN_CACHE: dict[tuple, tuple] = {}


def _norm_adjacency(edge_index: torch.Tensor, n: int) -> torch.Tensor:
    """D^-1/2 A D^-1/2 as a sparse matrix (in-degree, at least one), cached per edge tensor."""
    key = (id(edge_index), n)
    if key in _LAPLACIAN_CACHE and _LAPLACIAN_CACHE[key][0] is edge_index:
        return _LAPLACIAN_CACHE[key][1]
    src, dst = edge_index
    deg = torch.bincount(dst, minlength=n).float().clamp(min=1)
    w = deg[src].pow(-0.5) * deg[dst].pow(-0.5)
    A = torch.sparse_coo_tensor(torch.stack([dst, src]), w, (n, n)).coalesce()
    _LAPLACIAN_CACHE[key] = (edge_index, A)
    return A


class BWGNNLayer(nn.Module):
    """Beta-wavelet filter bank on each relation; filter outputs concatenated, projected and summed over relations."""

    def __init__(self, hidden: int, order: int, relations: list):
        super().__init__()
        self.theta = beta_wavelet_coefficients(order)
        self.relations = relations
        self.lin = nn.Linear(hidden * (order + 1), hidden)

    def forward(self, h: torch.Tensor, edge_index_dict: dict) -> torch.Tensor:
        n, out = h.size(0), 0
        for rel in self.relations:
            A = _norm_adjacency(edge_index_dict[rel], n)
            powers, x = [h], h
            for _ in range(len(self.theta) - 1):
                x = x - torch.sparse.mm(A, x)         # L x = x - D^-1/2 A D^-1/2 x
                powers.append(x)
            filt = [sum(c * pw for c, pw in zip(coef, powers)) for coef in self.theta]
            out = out + self.lin(torch.cat(filt, dim=-1))
        return out


class CARELayer(nn.Module):
    """One CARE-GNN layer (Dou et al., 2020) for full-batch training.

    A one-unit label-aware classifier scores every node; on each relation a node keeps the fraction p_r of its
    neighbours whose scores are closest to its own, averages them, and the relation averages are combined with the
    node's own embedding, weighted by p_r. p_r starts at 0.5 and moves by tau after each epoch: up if the mean
    score distance of the kept neighbours fell, down otherwise; it stops moving once the last ten moves cancel.
    """

    def __init__(self, hidden: int, relations: list, tau: float = 0.02):
        super().__init__()
        self.relations, self.tau = relations, tau
        self.sim = nn.Linear(hidden, 1)
        self.W = nn.Linear(hidden, hidden, bias=False)
        self.register_buffer("p", torch.full((len(relations),), 0.5))
        self.register_buffer("frozen", torch.zeros((), dtype=torch.bool))
        self._dist_sum = torch.zeros(len(relations))
        self._dist_cnt = torch.zeros(len(relations))
        self._prev = [None] * len(relations)
        self._moves = [[] for _ in relations]
        self.last_logit = None

    def forward(self, h: torch.Tensor, edge_index_dict: dict) -> torch.Tensor:
        n = h.size(0)
        logit = self.sim(h).squeeze(-1)
        self.last_logit = logit
        score = torch.sigmoid(logit).detach()
        agg = 0
        for r, rel in enumerate(self.relations):
            src, dst = edge_index_dict[rel]
            dist = (score[src] - score[dst]).abs()
            # Rank each node's incoming edges by distance; keep ceil(p * degree) of them (at least one).
            order = torch.argsort(dist, stable=True)
            order = order[torch.argsort(dst[order], stable=True)]
            deg = torch.bincount(dst, minlength=n)
            start = torch.cumsum(deg, 0) - deg
            rank = torch.arange(order.numel()) - start[dst[order]]
            keep_n = torch.ceil(self.p[r] * deg.float()).clamp(min=1).long()
            kept = order[rank < keep_n[dst[order]]]
            if self.training:
                self._dist_sum[r] += float(dist[kept].sum())
                self._dist_cnt[r] += kept.numel()
            mean = scatter(h[src[kept]], dst[kept], dim=0, dim_size=n, reduce="mean")
            agg = agg + self.p[r] * self.W(F.relu(mean))
        return F.relu(self.W(h) + agg)

    @torch.no_grad()
    def end_epoch(self) -> None:
        """Adapt the thresholds from this epoch's mean distance of the kept neighbours (the bandit update)."""
        for r in range(len(self.relations)):
            if self._dist_cnt[r] == 0 or bool(self.frozen):
                continue
            d = float(self._dist_sum[r] / self._dist_cnt[r])
            if self._prev[r] is not None:
                move = 1 if d <= self._prev[r] else -1
                self.p[r] = float(min(1.0, max(self.tau, float(self.p[r]) + move * self.tau)))
                self._moves[r].append(move)
            self._prev[r] = d
        if all(len(m) >= 10 and abs(sum(m[-10:])) <= 2 for m in self._moves):
            self.frozen.fill_(True)
        self._dist_sum.zero_()
        self._dist_cnt.zero_()


class GraphModel(nn.Module):
    def __init__(self, kind: str, data: HeteroData, hidden: int = 64, layers: int = 2, heads: int = 2,
                 dropout: float = 0.2, aggr: str = "mean", rgcn_bases: int = 4, simplehgn_beta: float = 0.05,
                 target: str = "claim", order: int = 2, care_lambda: float = 2.0):
        super().__init__()
        assert kind in GRAPH_KINDS, kind
        self.kind, self.layers, self.dropout, self.hidden, self.target = kind, layers, dropout, hidden, target
        self.care_lambda = care_lambda
        if kind in TARGET_ONLY_KINDS:
            in_dim = data[target].x.size(1)
            self.proj = nn.ModuleDict({target: nn.Linear(in_dim, hidden)})
            relations = [et for et in data.edge_types if et[0] == et[2] == target]
            if kind == "han":
                self.convs = nn.ModuleList([HANConv(hidden, hidden, data.metadata(), heads=heads, dropout=dropout)
                                            for _ in range(layers)])
            elif kind == "bwgnn":
                # Published structure: two-layer input transform, one filter bank, then the classifier.
                self.pre = nn.Linear(hidden, hidden)
                self.convs = nn.ModuleList([BWGNNLayer(hidden, order, relations)])
            else:
                self.convs = nn.ModuleList([CARELayer(hidden, relations) for _ in range(layers)])
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
        if self.kind == "bwgnn":
            x = F.relu(self.pre(h[self.target]))
            x = F.relu(self.convs[0](x, data.edge_index_dict))
            return self.head(F.dropout(x, self.dropout, self.training))
        if self.kind == "care":
            x = h[self.target]
            for conv in self.convs:
                x = F.dropout(conv(x, data.edge_index_dict), self.dropout, self.training)
            return self.head(x)
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

    def aux_loss(self, data: HeteroData) -> torch.Tensor:
        """CARE-GNN's similarity loss (weight lambda) on the training nodes of the last forward pass; 0 otherwise."""
        if self.kind != "care":
            return torch.zeros(())
        mask = data[self.target].target_mask
        y = data[self.target].y[mask].float()
        pw = (len(y) - y.sum()) / y.sum().clamp(min=1)
        return self.care_lambda * sum(F.binary_cross_entropy_with_logits(c.last_logit[mask], y, pos_weight=pw)
                                      for c in self.convs) / len(self.convs)

    def end_epoch(self) -> None:
        if self.kind == "care":
            for conv in self.convs:
                conv.end_epoch()
