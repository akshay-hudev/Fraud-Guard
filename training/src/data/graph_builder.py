"""
Graph Builder
Converts the processed healthcare dataset into a heterogeneous graph
using PyTorch Geometric (PyG) for GNN training.

Node types  : Patient, Doctor, Hospital, Claim
Edge types  :
  patient  → claim       (filed_claim)
  claim    → doctor      (treated_by)
  doctor   → hospital    (works_at)
  patient  → doctor      (visited)
"""

import os
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData
import logging

log = logging.getLogger(__name__)


class GraphBuilder:
    """Builds a PyG HeteroData graph from processed CSVs."""

    def __init__(self, processed_dir: str = "data/processed"):
        self.processed_dir = processed_dir

    # ── Node feature builders ──────────────────────────────────────────────────

    def _patient_features(self, patients: pd.DataFrame) -> torch.Tensor:
        feats = patients[["age", "member_since_years"]].copy()
        feats["gender"]       = (patients["gender"] == "M").astype(float)
        feats["has_chronic"]  = (patients["chronic_condition"] != "None").astype(float)
        return torch.tensor(feats.values, dtype=torch.float)

    def _doctor_features(self, doctors: pd.DataFrame) -> torch.Tensor:
        feats = doctors[["years_experience", "avg_claims_per_month"]].copy()
        feats["license_valid"] = doctors["license_valid"].astype(float)
        return torch.tensor(feats.values, dtype=torch.float)

    def _hospital_features(self, hospitals: pd.DataFrame) -> torch.Tensor:
        feats = hospitals[["num_beds"]].copy()
        feats["is_accredited"] = hospitals["is_accredited"].astype(float)
        return torch.tensor(feats.values, dtype=torch.float)

    def _claim_features(self, claims: pd.DataFrame) -> torch.Tensor:
        num_cols = [
            "claim_amount", "num_procedures", "days_in_hospital",
            "patient_total_claims", "patient_avg_amount",
            "doctor_total_claims",  "doctor_avg_amount",
            "amount_vs_patient_avg", "amount_vs_doctor_avg",
        ]
        # Use only columns that exist
        available = [c for c in num_cols if c in claims.columns]
        feats = claims[available].fillna(0).values
        # Normalize claim_amount
        feats = feats.astype(np.float32)
        feats = (feats - feats.mean(0)) / (feats.std(0) + 1e-8)
        return torch.tensor(feats, dtype=torch.float)

    # ── Index mapping helpers ──────────────────────────────────────────────────

    @staticmethod
    def _build_index(series: pd.Series) -> dict:
        """Map string IDs → contiguous integers."""
        return {v: i for i, v in enumerate(series.unique())}

    # ── Main builder ──────────────────────────────────────────────────────────

    def build(self) -> HeteroData:
        # Load artefacts
        patients  = pd.read_csv(f"{self.processed_dir}/../../data/raw/patients.csv")
        doctors   = pd.read_csv(f"{self.processed_dir}/../../data/raw/doctors.csv")
        hospitals = pd.read_csv(f"{self.processed_dir}/../../data/raw/hospitals.csv")

        # Use enriched claims (has engineered features)
        claims_path = f"{self.processed_dir}/claims_engineered.csv"
        if os.path.exists(claims_path):
            claims = pd.read_csv(claims_path)
        else:
            claims = pd.read_csv(f"{self.processed_dir}/../../data/raw/claims.csv")

        # Ensure fraud_label exists
        if "fraud_label" not in claims.columns:
            claims["fraud_label"] = 0

        # ── Index maps ────────────────────────────────────────────────────────
        patient_idx  = self._build_index(patients["patient_id"])
        doctor_idx   = self._build_index(doctors["doctor_id"])
        hospital_idx = self._build_index(hospitals["hospital_id"])
        claim_idx    = {v: i for i, v in enumerate(claims["claim_id"])}

        log.info("Graph nodes — patients: %d | doctors: %d | hospitals: %d | claims: %d",
                 len(patient_idx), len(doctor_idx), len(hospital_idx), len(claim_idx))

        # ── Build HeteroData ──────────────────────────────────────────────────
        data = HeteroData()

        # Node features
        data["patient"].x  = self._patient_features(patients)
        data["doctor"].x   = self._doctor_features(doctors)
        data["hospital"].x = self._hospital_features(hospitals)
        data["claim"].x    = self._claim_features(claims)

        # Node labels (only claims have fraud labels)
        data["claim"].y = torch.tensor(claims["fraud_label"].values, dtype=torch.long)

        # ── Edges ─────────────────────────────────────────────────────────────

        # patient → claim  (filed_claim)
        src, dst = [], []
        for _, row in claims.iterrows():
            if row["patient_id"] in patient_idx and row["claim_id"] in claim_idx:
                src.append(patient_idx[row["patient_id"]])
                dst.append(claim_idx[row["claim_id"]])
        data["patient", "filed_claim", "claim"].edge_index = torch.tensor(
            [src, dst], dtype=torch.long
        )

        # claim → doctor  (treated_by)
        src, dst = [], []
        for _, row in claims.iterrows():
            if row["claim_id"] in claim_idx and row["doctor_id"] in doctor_idx:
                src.append(claim_idx[row["claim_id"]])
                dst.append(doctor_idx[row["doctor_id"]])
        data["claim", "treated_by", "doctor"].edge_index = torch.tensor(
            [src, dst], dtype=torch.long
        )

        # doctor → hospital  (works_at)
        src, dst = [], []
        for _, row in doctors.iterrows():
            if row["doctor_id"] in doctor_idx and row["hospital_id"] in hospital_idx:
                src.append(doctor_idx[row["doctor_id"]])
                dst.append(hospital_idx[row["hospital_id"]])
        data["doctor", "works_at", "hospital"].edge_index = torch.tensor(
            [src, dst], dtype=torch.long
        )

        # patient → doctor  (visited) — derived from claims
        visited = claims[["patient_id", "doctor_id"]].drop_duplicates()
        src, dst = [], []
        for _, row in visited.iterrows():
            if row["patient_id"] in patient_idx and row["doctor_id"] in doctor_idx:
                src.append(patient_idx[row["patient_id"]])
                dst.append(doctor_idx[row["doctor_id"]])
        data["patient", "visited", "doctor"].edge_index = torch.tensor(
            [src, dst], dtype=torch.long
        )

        # ── Train / val / test masks on claims ────────────────────────────────
        n_claims = len(claims)
        perm     = torch.randperm(n_claims, generator=torch.Generator().manual_seed(42))
        n_train  = int(0.70 * n_claims)
        n_val    = int(0.15 * n_claims)

        train_mask = torch.zeros(n_claims, dtype=torch.bool)
        val_mask   = torch.zeros(n_claims, dtype=torch.bool)
        test_mask  = torch.zeros(n_claims, dtype=torch.bool)

        train_mask[perm[:n_train]]            = True
        val_mask  [perm[n_train:n_train+n_val]] = True
        test_mask [perm[n_train+n_val:]]      = True

        data["claim"].train_mask = train_mask
        data["claim"].val_mask   = val_mask
        data["claim"].test_mask  = test_mask

        log.info("Graph built. Saving...")
        torch.save(data, f"{self.processed_dir}/hetero_graph.pt")
        log.info("✅ Graph saved to '%s/hetero_graph.pt'", self.processed_dir)

        # Also save mappings for inference
        import json
        mappings = {
            "patient_idx":  {k: v for k, v in patient_idx.items()},
            "doctor_idx":   {k: v for k, v in doctor_idx.items()},
            "hospital_idx": {k: v for k, v in hospital_idx.items()},
            "claim_idx":    {k: v for k, v in claim_idx.items()},
        }
        with open(f"{self.processed_dir}/graph_mappings.json", "w") as f:
            json.dump(mappings, f)

        return data

    def load(self) -> HeteroData:
        path = f"{self.processed_dir}/hetero_graph.pt"
        if not os.path.exists(path):
            raise FileNotFoundError(f"Graph not found at {path}. Run build() first.")
        return torch.load(path)


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    builder = GraphBuilder(processed_dir="data/processed")
    data    = builder.build()
    print(data)
