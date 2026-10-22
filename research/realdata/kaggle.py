"""Kaggle "Healthcare Provider Fraud Detection Analysis" (CC0) — provider-level task.

Labels (``PotentialFraud``) are given per provider, so the prediction unit is
the provider. All claims of the observation year are visible; labels of
validation and test providers are hidden (an audit setting in which every
provider's claims are observed but only some providers have been investigated).

Feature groups (provider level):
* ``P`` provider aggregates of its claims and beneficiaries (no labels)
* ``G`` label-free graph features from shared beneficiaries / physicians
* ``L`` neighbourhood label rates computed from *training* labels only

Graph (heterogeneous, with reverse relations):
    provider -serves-> beneficiary, provider -employs-> physician
Physicians attached to a single provider carry no cross-provider information
and are folded into provider aggregates rather than kept as nodes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch_geometric.transforms as T
from torch_geometric.data import HeteroData

from ..ledger import KAGGLE_DIR

FILES = {
    "labels": "Train-1542865627584.csv",
    "beneficiary": "Train_Beneficiarydata-1542865627584.csv",
    "inpatient": "Train_Inpatientdata-1542865627584.csv",
    "outpatient": "Train_Outpatientdata-1542865627584.csv",
}
PHYS_COLS = ["AttendingPhysician", "OperatingPhysician", "OtherPhysician"]


@dataclass
class KaggleData:
    providers: pd.DataFrame          # provider_id, y (sorted by provider_id; row index = node index)
    claims: pd.DataFrame
    beneficiaries: pd.DataFrame
    P: pd.DataFrame                  # provider aggregate features, row-aligned with providers
    G: pd.DataFrame                  # label-free graph features
    B_pb: sp.csr_matrix              # provider x beneficiary incidence
    B_pf: sp.csr_matrix              # provider x shared-physician incidence
    bene_x: np.ndarray
    phys_x: np.ndarray


def load_raw(data_dir: Path = KAGGLE_DIR) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    lab = pd.read_csv(data_dir / FILES["labels"])
    ben = pd.read_csv(data_dir / FILES["beneficiary"])
    ip = pd.read_csv(data_dir / FILES["inpatient"]).assign(is_inpatient=1)
    op = pd.read_csv(data_dir / FILES["outpatient"]).assign(is_inpatient=0)
    claims = pd.concat([ip, op], ignore_index=True, sort=False)
    return lab, ben, claims


def _beneficiary_frame(ben: pd.DataFrame) -> pd.DataFrame:
    b = ben.copy()
    ref = pd.Timestamp("2009-12-31")
    b["age"] = (ref - pd.to_datetime(b["DOB"])).dt.days / 365.25
    b["deceased"] = b["DOD"].notna().astype(int)
    b["renal"] = (b["RenalDiseaseIndicator"].astype(str) == "Y").astype(int)
    chronic = [c for c in b.columns if c.startswith("ChronicCond_")]
    for c in chronic:
        b[c] = (b[c] == 1).astype(int)
    b["chronic_count"] = b[chronic].sum(1)
    b["gender"] = (b["Gender"] == 1).astype(int)
    for c in ["IPAnnualReimbursementAmt", "IPAnnualDeductibleAmt", "OPAnnualReimbursementAmt", "OPAnnualDeductibleAmt"]:
        b["log_" + c] = np.log1p(b[c].clip(lower=0))
    keep = (["BeneID", "age", "deceased", "renal", "chronic_count", "gender", "Race", "State",
             "NoOfMonths_PartACov", "NoOfMonths_PartBCov"] + chronic
            + [c for c in b.columns if c.startswith("log_")])
    return b[keep]


def _claim_frame(claims: pd.DataFrame) -> pd.DataFrame:
    c = claims.copy()
    c["start"] = pd.to_datetime(c["ClaimStartDt"])
    c["end"] = pd.to_datetime(c["ClaimEndDt"])
    c["duration"] = (c["end"] - c["start"]).dt.days.clip(lower=0)
    c["los"] = (pd.to_datetime(c["DischargeDt"]) - pd.to_datetime(c["AdmissionDt"])).dt.days.clip(lower=0)
    c["n_diag"] = c[[f"ClmDiagnosisCode_{i}" for i in range(1, 11)]].notna().sum(1)
    c["n_proc"] = c[[f"ClmProcedureCode_{i}" for i in range(1, 7)]].notna().sum(1)
    c["has_operating"] = c["OperatingPhysician"].notna().astype(int)
    c["reimb"] = c["InscClaimAmtReimbursed"].fillna(0)
    c["log_reimb"] = np.log1p(c["reimb"].clip(lower=0))
    c["deductible"] = c["DeductibleAmtPaid"].fillna(0)
    return c


def provider_aggregates(c: pd.DataFrame, b: pd.DataFrame, providers: pd.Series) -> pd.DataFrame:
    cb = c.merge(b, on="BeneID", how="left")
    g = cb.groupby("Provider")
    P = pd.DataFrame({
        "log_n_claims": np.log1p(g.size()),
        "ip_share": g["is_inpatient"].mean(),
        "log_n_benes": np.log1p(g["BeneID"].nunique()),
        "claims_per_bene": g.size() / g["BeneID"].nunique(),
        "log_n_attending": np.log1p(g["AttendingPhysician"].nunique()),
        "log_n_operating": np.log1p(g["OperatingPhysician"].nunique()),
        "log_total_reimb": np.log1p(g["reimb"].sum()),
        "mean_log_reimb": g["log_reimb"].mean(),
        "max_log_reimb": g["log_reimb"].max(),
        "std_log_reimb": g["log_reimb"].std().fillna(0),
        "mean_deductible": g["deductible"].mean(),
        "mean_duration": g["duration"].mean(),
        "mean_los": g["los"].mean().fillna(0),
        "mean_n_diag": g["n_diag"].mean(),
        "mean_n_proc": g["n_proc"].mean(),
        "operating_share": g["has_operating"].mean(),
        "bene_mean_age": g["age"].mean(),
        "bene_mean_chronic": g["chronic_count"].mean(),
        "bene_deceased_share": g["deceased"].mean(),
        "bene_renal_share": g["renal"].mean(),
        "bene_mean_log_ip_reimb": g["log_IPAnnualReimbursementAmt"].mean(),
        "bene_mean_log_op_reimb": g["log_OPAnnualReimbursementAmt"].mean(),
        "log_n_states": np.log1p(g["State"].nunique()),
    })
    return P.reindex(providers).reset_index(drop=True)


def _incidence(rows: pd.Series, cols: pd.Series, row_index: pd.Index, col_index: pd.Index) -> sp.csr_matrix:
    r = row_index.get_indexer(rows)
    k = col_index.get_indexer(cols)
    m = sp.csr_matrix((np.ones(len(r)), (r, k)), shape=(len(row_index), len(col_index)))
    m.data[:] = 1.0
    return m


def _overlap_features(B: sp.csr_matrix, prefix: str) -> pd.DataFrame:
    """Shared-neighbour statistics between providers through a bipartite incidence B."""
    C = (B @ B.T).tocsr()
    deg = np.asarray(B.sum(1)).ravel()
    C.setdiag(0)
    C.eliminate_zeros()
    n_co = np.diff(C.indptr)
    coo = C.tocoo()
    jac = coo.data / (deg[coo.row] + deg[coo.col] - coo.data)
    max_j = np.zeros(B.shape[0])
    np.maximum.at(max_j, coo.row, jac)
    col_deg = np.asarray(B.sum(0)).ravel()
    shared_share = np.asarray(B @ (col_deg > 1).astype(float)).ravel() / np.maximum(deg, 1)
    mean_col_deg = np.asarray(B @ col_deg).ravel() / np.maximum(deg, 1)
    return pd.DataFrame({
        f"{prefix}_log_n_coproviders": np.log1p(n_co),
        f"{prefix}_max_jaccard": max_j,
        f"{prefix}_shared_share": shared_share,
        f"{prefix}_mean_degree": mean_col_deg,
    })


def load(data_dir: Path = KAGGLE_DIR) -> KaggleData:
    lab, ben, claims = load_raw(data_dir)
    providers = lab.rename(columns={"Provider": "provider_id"}).sort_values("provider_id").reset_index(drop=True)
    providers["y"] = (providers["PotentialFraud"] == "Yes").astype(int)
    b = _beneficiary_frame(ben)
    c = _claim_frame(claims)
    P = provider_aggregates(c, b, providers["provider_id"])
    prov_idx = pd.Index(providers["provider_id"])
    bene_idx = pd.Index(b["BeneID"])
    pb = c[["Provider", "BeneID"]].drop_duplicates()
    B_pb = _incidence(pb["Provider"], pb["BeneID"], prov_idx, bene_idx)
    pf = pd.concat([c[["Provider", col]].rename(columns={col: "phys"}) for col in PHYS_COLS]).dropna().drop_duplicates()
    shared = pf.groupby("phys")["Provider"].nunique()
    shared = shared[shared > 1].index
    pf = pf[pf["phys"].isin(shared)]
    phys_idx = pd.Index(sorted(shared))
    B_pf = _incidence(pf["Provider"], pf["phys"], prov_idx, phys_idx)
    G = pd.concat([_overlap_features(B_pb, "bene"), _overlap_features(B_pf, "phys")], axis=1)
    bene_cols = [col for col in b.columns if col not in ("BeneID", "State", "Race")]
    bene_x = b[bene_cols].to_numpy(float)
    bene_x = (bene_x - bene_x.mean(0)) / np.where(bene_x.std(0) > 0, bene_x.std(0), 1)
    phys_claims = pd.concat([c[[col]].rename(columns={col: "phys"}) for col in PHYS_COLS]).dropna()
    phys_claims = phys_claims[phys_claims["phys"].isin(shared)].groupby("phys").size().reindex(phys_idx)
    phys_x = np.column_stack([np.log1p(phys_claims.to_numpy(float)), np.log1p(np.asarray(B_pf.sum(0)).ravel())])
    phys_x = (phys_x - phys_x.mean(0)) / np.where(phys_x.std(0) > 0, phys_x.std(0), 1)
    return KaggleData(providers, c, b, P, G, B_pb, B_pf, bene_x.astype(np.float32), phys_x.astype(np.float32))


def label_features(data: KaggleData, train_idx: np.ndarray) -> pd.DataFrame:
    """Share of co-providers (through beneficiaries / physicians) that are known-fraud training providers.

    Only training labels are used; a provider's own label never enters its feature.
    """
    known = np.zeros(len(data.providers))
    known[train_idx] = data.providers["y"].to_numpy()[train_idx]
    is_train = np.zeros(len(data.providers))
    is_train[train_idx] = 1
    out = {}
    for name, B in [("bene", data.B_pb), ("phys", data.B_pf)]:
        C = (B @ B.T).tocsr()
        C.setdiag(0)
        C.eliminate_zeros()
        C.data[:] = 1.0
        n_known = C @ is_train
        out[f"{name}_known_fraud_share"] = (C @ known) / np.maximum(n_known, 1)
        out[f"{name}_has_known_neighbour"] = (n_known > 0).astype(float)
    return pd.DataFrame(out)


def build_graph(data: KaggleData, provider_x: np.ndarray, y: np.ndarray) -> HeteroData:
    g = HeteroData()
    g["provider"].x = torch.tensor(provider_x, dtype=torch.float32)
    g["provider"].y = torch.tensor(y, dtype=torch.long)
    g["beneficiary"].x = torch.from_numpy(data.bene_x)
    g["physician"].x = torch.from_numpy(data.phys_x)
    pb = data.B_pb.tocoo()
    pf = data.B_pf.tocoo()
    g["provider", "serves", "beneficiary"].edge_index = torch.from_numpy(np.stack([pb.row, pb.col]).astype(np.int64))
    g["provider", "employs", "physician"].edge_index = torch.from_numpy(np.stack([pf.row, pf.col]).astype(np.int64))
    return T.ToUndirected()(g)


def build_metapath_graph(data: KaggleData, provider_x: np.ndarray, y: np.ndarray) -> HeteroData:
    """Provider-only graph for HAN: meta-paths provider-beneficiary-provider and provider-physician-provider."""
    g = HeteroData()
    g["provider"].x = torch.tensor(provider_x, dtype=torch.float32)
    g["provider"].y = torch.tensor(y, dtype=torch.long)
    for name, B in [("pbp", data.B_pb), ("pfp", data.B_pf)]:
        C = (B @ B.T).tocoo()
        keep = C.row != C.col
        g["provider", name, "provider"].edge_index = torch.from_numpy(np.stack([C.row[keep], C.col[keep]]).astype(np.int64))
    return g
