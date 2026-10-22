"""Per-period heterogeneous graphs for the synthetic claims.

Each audit period gets its own graph: every patient, doctor and hospital node
plus the claim nodes of that period, so graphs contain no future claims and
every split sees graphs of the same time span. Relations, each with a reverse
relation:

    patient -files-> claim, claim -billed_by-> doctor,
    doctor -affiliated-> hospital, patient -visits-> doctor (from visible claims)

Claim nodes carry claim-intrinsic attributes; patient/doctor/hospital nodes
carry their static attributes plus log(1 + degree) for each incident relation
(a label-free structural encoding, because attention/mean aggregation cannot
count neighbours). No engineered aggregates and no labels are node features.
Standardisation statistics are fitted on training data only.
"""

from __future__ import annotations

import numpy as np
import torch
import torch_geometric.transforms as T
from torch_geometric.data import HeteroData

from .features import WINDOW_ORDER
from .generator import SyntheticWorld

CLAIM_NODE_COLUMNS = ["log_amount", "num_procedures", "days_in_hospital", "inpatient", "proc_code_offset",
                      "month_sin", "month_cos"]


def _standardise(x: np.ndarray, fit_rows: np.ndarray) -> np.ndarray:
    mu, sd = x[fit_rows].mean(0), x[fit_rows].std(0)
    return ((x - mu) / np.where(sd > 0, sd, 1.0)).astype(np.float32)


def entity_features(world: SyntheticWorld) -> dict[str, np.ndarray]:
    k = world.config.n_specialties
    p = world.patients.sort_values("patient_id")
    d = world.doctors.sort_values("doctor_id")
    h = world.hospitals.sort_values("hospital_id")
    pat = p[["age", "gender", "chronic_count"]].to_numpy(float)
    doc = np.column_stack([d["years_experience"].to_numpy(float), np.eye(k)[d["specialty"].to_numpy()]])
    hosp = np.log(h[["num_beds"]].to_numpy(float))
    all_rows = lambda a: np.arange(len(a))
    return {"patient": _standardise(pat, all_rows(pat)), "doctor": _standardise(doc, all_rows(doc)),
            "hospital": _standardise(hosp, all_rows(hosp))}


def _period_graph(world: SyntheticWorld, rows: np.ndarray, x_claim: np.ndarray, ent: dict) -> HeteroData:
    vc = world.claims.iloc[rows]
    data = HeteroData()
    for t, x in ent.items():
        data[t].x = torch.from_numpy(x)
    data["claim"].x = torch.from_numpy(x_claim[rows])
    data["claim"].y = torch.from_numpy(world.y[rows].astype(np.int64))
    data["claim"].target_mask = torch.ones(len(rows), dtype=torch.bool)
    data["claim"].rows = torch.from_numpy(rows)          # row positions in world.claims
    claim_idx = torch.arange(len(rows))
    data["patient", "files", "claim"].edge_index = torch.stack([torch.from_numpy(vc["patient_id"].to_numpy()), claim_idx])
    data["claim", "billed_by", "doctor"].edge_index = torch.stack([claim_idx, torch.from_numpy(vc["doctor_id"].to_numpy())])
    d = world.doctors.sort_values("doctor_id")
    data["doctor", "affiliated", "hospital"].edge_index = torch.from_numpy(
        np.stack([d["doctor_id"].to_numpy(), d["hospital_id"].to_numpy()]))
    visits = vc[["patient_id", "doctor_id"]].drop_duplicates().to_numpy().T
    data["patient", "visits", "doctor"].edge_index = torch.from_numpy(np.ascontiguousarray(visits))
    return T.ToUndirected()(data)


def _degree_encodings(g: HeteroData) -> dict[str, np.ndarray]:
    enc = {}
    for t in ["patient", "doctor", "hospital"]:
        cols = [np.bincount(g[et].edge_index[1].numpy(), minlength=g[t].num_nodes)
                for et in g.edge_types if et[2] == t]
        enc[t] = np.log1p(np.column_stack(cols).astype(float))
    return enc


def build_graphs(world: SyntheticWorld, claim_frame) -> dict[str, list[HeteroData]]:
    """Return {split: [one graph per audit period in that split]}.

    claim_frame: claim-level features for all claims in claims order (columns CLAIM_NODE_COLUMNS).
    """
    train_rows = np.flatnonzero(world.split == "train")
    x_claim = _standardise(claim_frame[CLAIM_NODE_COLUMNS].to_numpy(float), train_rows)
    static = entity_features(world)
    periods = {p: np.flatnonzero(world.period == p) for p in np.unique(world.period)}
    raw = {p: _period_graph(world, rows, x_claim, static) for p, rows in periods.items()}
    deg = {p: _degree_encodings(g) for p, g in raw.items()}
    train_periods = sorted({int(p) for p in world.period[train_rows]})
    graphs = {name: [] for name in WINDOW_ORDER}
    for p, g in raw.items():
        for t in deg[p]:
            # Degree encodings standardised with statistics pooled over the training periods.
            pooled = np.vstack([deg[q][t] for q in train_periods])
            mu, sd = pooled.mean(0), pooled.std(0)
            enc = ((deg[p][t] - mu) / np.where(sd > 0, sd, 1.0)).astype(np.float32)
            g[t].x = torch.cat([g[t].x, torch.from_numpy(enc)], dim=1)
        split = world.split[periods[p][0]]
        graphs[split].append(g)
    return graphs


def _group_pairs(keys: np.ndarray) -> np.ndarray:
    """All ordered pairs (i, j), i != j, of positions sharing the same key."""
    order = np.argsort(keys, kind="stable")
    k = keys[order]
    bounds = np.flatnonzero(np.diff(k)) + 1
    src, dst = [], []
    for grp in np.split(order, bounds):
        if len(grp) < 2:
            continue
        a, b = np.meshgrid(grp, grp, indexing="ij")
        keep = a != b
        src.append(a[keep])
        dst.append(b[keep])
    if not src:
        return np.empty((2, 0), dtype=np.int64)
    return np.stack([np.concatenate(src), np.concatenate(dst)]).astype(np.int64)


def build_metapath_graphs(world: SyntheticWorld, graphs: dict[str, list[HeteroData]]) -> dict[str, list[HeteroData]]:
    """Claim-only graphs for HAN with meta-paths claim-patient-claim and claim-doctor-claim.

    Claim inputs are the claim's own features concatenated with the attributes of
    its patient, doctor and hospital, so HAN sees the same information as the
    other graph models.
    """
    c = world.claims
    out = {name: [] for name in graphs}
    for name in graphs:
        for g in graphs[name]:
            out[name].append(_metapath_graph(c, g))
    return out


def _metapath_graph(c, g: HeteroData) -> HeteroData:
    vc = c.iloc[g["claim"].rows.numpy()]
    x = torch.cat([g["claim"].x,
                   g["patient"].x[torch.from_numpy(vc["patient_id"].to_numpy())],
                   g["doctor"].x[torch.from_numpy(vc["doctor_id"].to_numpy())],
                   g["hospital"].x[torch.from_numpy(vc["hospital_id"].to_numpy())]], dim=1)
    data = HeteroData()
    data["claim"].x = x
    data["claim"].y = g["claim"].y
    data["claim"].target_mask = g["claim"].target_mask
    data["claim", "cpc", "claim"].edge_index = torch.from_numpy(_group_pairs(vc["patient_id"].to_numpy()))
    data["claim", "cdc", "claim"].edge_index = torch.from_numpy(_group_pairs(vc["doctor_id"].to_numpy()))
    return data
