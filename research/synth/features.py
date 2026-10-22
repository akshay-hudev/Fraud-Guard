"""Leakage-safe tabular features for the synthetic claims.

Scoring protocol (retrospective audit): each audit period is analysed on its
own. A claim's features use only claims filed in the same period (the
"visible" claims), so every split sees windows of the same length and
count-based features keep the same meaning. Labels are never used to build
features. Feature groups:

* ``C`` claim-level: the claim's own attributes plus static attributes of its
  patient, doctor and hospital.
* ``A`` entity aggregates over visible claims: utilisation, distinct
  counterparties, amount statistics and amount-vs-history ratios.
* ``G`` graph features over the visible patient-doctor bipartite graph:
  shared-patient overlap between doctors and triadic closure around the claim.

Train, validation and test claims use identical definitions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .generator import SyntheticWorld

C_NUMERIC = ["log_amount", "num_procedures", "days_in_hospital", "inpatient", "proc_code_offset",
             "month_sin", "month_cos", "age", "gender", "chronic_count", "years_experience", "log_beds"]
A_COLUMNS = ["pat_n_claims", "pat_n_doctors", "pat_mean_log_amount", "pat_max_log_amount",
             "doc_n_claims", "doc_n_patients", "doc_mean_log_amount", "hosp_n_claims", "pair_n_claims",
             "amt_vs_pat", "amt_vs_doc", "amt_vs_specialty"]
G_COLUMNS = ["doc_max_jaccard", "doc_n_overlap_docs", "doc_mean_patient_degree",
             "pat_mean_doc_jaccard", "pat_doc_pair_overlap", "claim_closure", "claim_closure_norm"]
WINDOW_ORDER = ["train", "val", "test"]


def claim_level(world: SyntheticWorld) -> pd.DataFrame:
    c, n_spec = world.claims, world.config.n_specialties
    doc = world.doctors.set_index("doctor_id").loc[c["doctor_id"]].reset_index(drop=True)
    pat = world.patients.set_index("patient_id").loc[c["patient_id"]].reset_index(drop=True)
    hosp = world.hospitals.set_index("hospital_id").loc[c["hospital_id"]].reset_index(drop=True)
    month = (c["day"] // 30) % 12
    out = pd.DataFrame({
        "log_amount": np.log(c["claim_amount"]),
        "num_procedures": c["num_procedures"],
        "days_in_hospital": c["days_in_hospital"],
        "inpatient": c["inpatient"],
        "proc_code_offset": c["procedure_code"] % 10,
        "month_sin": np.sin(2 * np.pi * month / 12),
        "month_cos": np.cos(2 * np.pi * month / 12),
        "age": pat["age"],
        "gender": pat["gender"],
        "chronic_count": pat["chronic_count"],
        "years_experience": doc["years_experience"],
        "log_beds": np.log(hosp["num_beds"]),
    })
    for k in range(n_spec):
        out[f"spec_{k}"] = (doc["specialty"].to_numpy() == k).astype(float)
    return out


def c_columns(world: SyntheticWorld) -> list[str]:
    return C_NUMERIC + [f"spec_{k}" for k in range(world.config.n_specialties)]


def _aggregates(c: pd.DataFrame, visible: np.ndarray, rows: np.ndarray, spec_mean: pd.Series,
                specialty_of_doc: np.ndarray) -> pd.DataFrame:
    v = c.loc[visible]
    la = np.log(v["claim_amount"])
    pat = v.assign(la=la).groupby("patient_id").agg(pat_n_claims=("claim_id", "size"),
                                                     pat_n_doctors=("doctor_id", "nunique"),
                                                     pat_mean_log_amount=("la", "mean"),
                                                     pat_max_log_amount=("la", "max"))
    doc = v.assign(la=la).groupby("doctor_id").agg(doc_n_claims=("claim_id", "size"),
                                                    doc_n_patients=("patient_id", "nunique"),
                                                    doc_mean_log_amount=("la", "mean"))
    hosp = v.groupby("hospital_id").size().rename("hosp_n_claims")
    pair = v.groupby(["patient_id", "doctor_id"]).size().rename("pair_n_claims")
    r = c.loc[rows]
    out = (r[["patient_id", "doctor_id", "hospital_id"]]
           .join(pat, on="patient_id").join(doc, on="doctor_id").join(hosp, on="hospital_id")
           .join(pair, on=["patient_id", "doctor_id"]))
    lr = np.log(r["claim_amount"])
    out["amt_vs_pat"] = lr - out["pat_mean_log_amount"]
    out["amt_vs_doc"] = lr - out["doc_mean_log_amount"]
    out["amt_vs_specialty"] = lr.to_numpy() - spec_mean.reindex(specialty_of_doc[r["doctor_id"]]).to_numpy()
    return out[A_COLUMNS].reset_index(drop=True)


def _graph_features(c: pd.DataFrame, visible: np.ndarray, rows: np.ndarray, n_pat: int, n_doc: int) -> pd.DataFrame:
    v = c.loc[visible, ["patient_id", "doctor_id"]].drop_duplicates()
    B = sp.csr_matrix((np.ones(len(v)), (v["patient_id"], v["doctor_id"])), shape=(n_pat, n_doc))
    D = (B.T @ B).toarray()                       # doctor x doctor shared-patient counts
    deg = np.diag(D).copy()
    np.fill_diagonal(D, 0)
    union = deg[:, None] + deg[None, :] - D
    J = np.divide(D, union, out=np.zeros_like(D, dtype=float), where=union > 0)
    pat_deg = np.asarray(B.sum(1)).ravel()
    doc_max_j = J.max(1)
    doc_overlap = (D >= 2).sum(1)
    doc_mean_pdeg = np.asarray((B.T @ sp.diags(pat_deg)).sum(1)).ravel() / np.maximum(deg, 1)
    pat_mean_j = np.asarray(B @ doc_max_j).ravel() / np.maximum(pat_deg, 1)
    # Pairs of the patient's doctors that share >= 2 patients, normalised by possible pairs.
    Dov = sp.csr_matrix((D >= 2).astype(float))
    pair_cnt = np.asarray((B @ Dov).multiply(B).sum(1)).ravel() / 2
    pat_pair_overlap = pair_cnt / np.maximum(pat_deg * (pat_deg - 1) / 2, 1)
    M = np.asarray((B @ sp.csr_matrix(D)).todense())   # patients x doctors: sum_{d' in N(p)} D[d', d]
    r = c.loc[rows]
    p, d = r["patient_id"].to_numpy(), r["doctor_id"].to_numpy()
    closure = M[p, d]
    return pd.DataFrame({
        "doc_max_jaccard": doc_max_j[d],
        "doc_n_overlap_docs": doc_overlap[d],
        "doc_mean_patient_degree": doc_mean_pdeg[d],
        "pat_mean_doc_jaccard": pat_mean_j[p],
        "pat_doc_pair_overlap": pat_pair_overlap[p],
        "claim_closure": closure,
        "claim_closure_norm": closure / np.maximum(pat_deg[p], 1),
    })


def build_tabular(world: SyntheticWorld) -> dict[str, pd.DataFrame]:
    """Return {split: DataFrame with C, A and G columns plus claim_id}, rows in claims order."""
    c = world.claims
    cl = claim_level(world)
    specialty_of_doc = world.doctors.set_index("doctor_id")["specialty"].reindex(np.arange(world.config.n_doctors)).to_numpy()
    train_rows = np.flatnonzero(world.split == "train")
    spec_mean = (pd.Series(np.log(c.loc[train_rows, "claim_amount"]).to_numpy(),
                           index=specialty_of_doc[c.loc[train_rows, "doctor_id"]]).groupby(level=0).mean())
    per_period = []
    for period in np.unique(world.period):
        rows = np.flatnonzero(world.period == period)
        per_period.append(pd.concat([
            pd.DataFrame({"claim_id": c.loc[rows, "claim_id"].to_numpy()}),
            cl.loc[rows].reset_index(drop=True),
            _aggregates(c, rows, rows, spec_mean, specialty_of_doc),
            _graph_features(c, rows, rows, world.config.n_patients, world.config.n_doctors),
        ], axis=1))
    frame = pd.concat(per_period, ignore_index=True)     # periods are date-ordered, so this is claims order
    return {name: frame[world.split == name].reset_index(drop=True) for name in WINDOW_ORDER}
