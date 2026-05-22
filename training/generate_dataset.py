"""
Publication synthetic dataset generator.

Creates the IJISA-scale dataset used in the paper appendix and logs all fraud
injection parameters for exact reproducibility.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


TRAINING_DIR = Path(__file__).resolve().parent
REPO_ROOT = TRAINING_DIR.parent

DATASET_CONFIG = {
    "n_claims": 2100,
    "n_patients": 500,
    "n_doctors": 50,
    "n_hospitals": 20,
    "n_rings": 5,
    "ring_size": 4,
    "patients_per_ring": 10,
    "isolated_fraud_n": 200,
    "ring_fraud_n": 178,
    "temporal_span_months": 24,
    "seed": 42,
}


def _date_for(rng: np.random.Generator, months: int) -> str:
    start = datetime(2024, 1, 1)
    days = int(months * 30.4375)
    return (start + timedelta(days=int(rng.integers(0, days)))).strftime("%Y-%m-%d")


def generate_hospitals(config: dict, rng: np.random.Generator) -> pd.DataFrame:
    states = ["CA", "TX", "NY", "FL", "IL", "PA", "OH", "GA"]
    specialties = ["General", "Cardiology", "Orthopedics", "Neurology", "Oncology"]
    rows = []
    for i in range(config["n_hospitals"]):
        rows.append({
            "hospital_id": f"H{i:04d}",
            "hospital_name": f"Hospital_{i}",
            "specialty": rng.choice(specialties),
            "state": rng.choice(states),
            "num_beds": int(rng.integers(60, 700)),
            "is_accredited": bool(rng.random() > 0.08),
            "fraud_prone": bool(rng.random() < 0.12),
        })
    return pd.DataFrame(rows)


def generate_doctors(config: dict, hospitals: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    specialties = [
        "General Practitioner", "Cardiologist", "Orthopedic Surgeon",
        "Neurologist", "Oncologist", "Pediatrician", "Radiologist",
    ]
    ring_doctor_total = config["n_rings"] * config["ring_size"]
    rows = []
    for i in range(config["n_doctors"]):
        hospital = hospitals.sample(1, random_state=int(rng.integers(0, 1_000_000))).iloc[0]
        ring_id = (i // config["ring_size"]) + 1 if i < ring_doctor_total else 0
        rows.append({
            "doctor_id": f"D{i:04d}",
            "doctor_name": f"Dr_Doctor_{i}",
            "specialty": rng.choice(specialties),
            "hospital_id": hospital["hospital_id"],
            "years_experience": int(rng.integers(1, 41)),
            "avg_claims_per_month": int(rng.integers(5, 80)),
            "license_valid": bool(rng.random() > 0.05),
            "fraud_prone": bool(ring_id > 0 or rng.random() < 0.08),
            "ring_id": ring_id,
        })
    return pd.DataFrame(rows)


def generate_patients(config: dict, rng: np.random.Generator) -> pd.DataFrame:
    conditions = ["Diabetes", "Hypertension", "Asthma", "Cancer", "Heart Disease", "Arthritis", "None"]
    insurance = ["Medicare", "Medicaid", "Private", "Self-Pay"]
    states = ["CA", "TX", "NY", "FL", "IL"]
    ring_patient_total = config["n_rings"] * config["patients_per_ring"]
    rows = []
    for i in range(config["n_patients"]):
        ring_id = (i // config["patients_per_ring"]) + 1 if i < ring_patient_total else 0
        rows.append({
            "patient_id": f"P{i:05d}",
            "age": int(rng.integers(18, 86)),
            "gender": rng.choice(["M", "F"]),
            "chronic_condition": rng.choice(conditions),
            "insurance_type": rng.choice(insurance),
            "state": rng.choice(states),
            "member_since_years": int(rng.integers(0, 21)),
            "fraud_prone": bool(ring_id > 0 or rng.random() < 0.06),
            "ring_id": ring_id,
        })
    return pd.DataFrame(rows)


def _claim_row(
    claim_id: int,
    patient: pd.Series,
    doctor: pd.Series,
    hospital: pd.Series,
    rng: np.random.Generator,
    config: dict,
    fraud_label: int,
    ring_id: int,
) -> dict:
    if fraud_label:
        amount = float(rng.uniform(4500, 22000))
        procedures = int(rng.integers(4, 12))
        days = int(rng.integers(1, 16))
        approved = bool(rng.random() < 0.35)
    else:
        amount = float(rng.uniform(120, 8500))
        procedures = int(rng.integers(1, 6))
        days = int(rng.integers(0, 6))
        approved = bool(rng.random() < 0.94)
    return {
        "claim_id": f"C{claim_id:06d}",
        "patient_id": patient["patient_id"],
        "doctor_id": doctor["doctor_id"],
        "hospital_id": hospital["hospital_id"],
        "claim_date": _date_for(rng, config["temporal_span_months"]),
        "claim_amount": round(amount, 2),
        "num_procedures": procedures,
        "procedure_code": f"PC{int(rng.integers(1, 51)):03d}",
        "diagnosis_code": f"DX{int(rng.integers(1, 41)):03d}",
        "days_in_hospital": days,
        "approved": approved,
        "fraud_label": fraud_label,
        "ring_id": int(ring_id),
    }


def generate_claims(config: dict, patients: pd.DataFrame, doctors: pd.DataFrame, hospitals: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    claim_id = 0
    n_legit = config["n_claims"] - config["isolated_fraud_n"] - config["ring_fraud_n"]

    for _ in range(n_legit):
        patient = patients.sample(1, random_state=int(rng.integers(0, 1_000_000))).iloc[0]
        doctor = doctors.sample(1, random_state=int(rng.integers(0, 1_000_000))).iloc[0]
        hospital = hospitals[hospitals["hospital_id"] == doctor["hospital_id"]].iloc[0]
        rows.append(_claim_row(claim_id, patient, doctor, hospital, rng, config, 0, 0))
        claim_id += 1

    non_ring_patients = patients[patients["ring_id"] == 0]
    non_ring_doctors = doctors[doctors["ring_id"] == 0]
    for _ in range(config["isolated_fraud_n"]):
        patient = non_ring_patients.sample(1, random_state=int(rng.integers(0, 1_000_000))).iloc[0]
        doctor = non_ring_doctors.sample(1, random_state=int(rng.integers(0, 1_000_000))).iloc[0]
        hospital = hospitals[hospitals["hospital_id"] == doctor["hospital_id"]].iloc[0]
        rows.append(_claim_row(claim_id, patient, doctor, hospital, rng, config, 1, 0))
        claim_id += 1

    for i in range(config["ring_fraud_n"]):
        ring_id = (i % config["n_rings"]) + 1
        ring_patients = patients[patients["ring_id"] == ring_id]
        ring_doctors = doctors[doctors["ring_id"] == ring_id]
        patient = ring_patients.sample(1, random_state=int(rng.integers(0, 1_000_000))).iloc[0]
        doctor = ring_doctors.sample(1, random_state=int(rng.integers(0, 1_000_000))).iloc[0]
        hospital = hospitals[hospitals["hospital_id"] == doctor["hospital_id"]].iloc[0]
        rows.append(_claim_row(claim_id, patient, doctor, hospital, rng, config, 1, ring_id))
        claim_id += 1

    claims = pd.DataFrame(rows)
    claims["claim_date"] = pd.to_datetime(claims["claim_date"])
    claims = claims.sample(frac=1.0, random_state=config["seed"]).sort_values("claim_date").reset_index(drop=True)
    claims["claim_date"] = claims["claim_date"].dt.strftime("%Y-%m-%d")
    return claims


def _dataset_hash(output_dir: Path) -> str:
    hasher = hashlib.sha256()
    for name in ["hospitals.csv", "doctors.csv", "patients.csv", "claims.csv"]:
        hasher.update((output_dir / name).read_bytes())
    return hasher.hexdigest()


def generate_and_save(output_dir: str | Path | None = None, seed: int = 42) -> str:
    config = dict(DATASET_CONFIG)
    config["seed"] = int(seed)
    rng = np.random.default_rng(seed)
    random.seed(seed)

    output_dir = Path(output_dir) if output_dir is not None else TRAINING_DIR / "data" / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)
    LOG_DIR = REPO_ROOT / "logs"
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    hospitals = generate_hospitals(config, rng)
    doctors = generate_doctors(config, hospitals, rng)
    patients = generate_patients(config, rng)
    claims = generate_claims(config, patients, doctors, hospitals, rng)

    hospitals.to_csv(output_dir / "hospitals.csv", index=False)
    doctors.to_csv(output_dir / "doctors.csv", index=False)
    patients.to_csv(output_dir / "patients.csv", index=False)
    claims.to_csv(output_dir / "claims.csv", index=False)

    dataset_hash = _dataset_hash(output_dir)
    config["dataset_sha256"] = dataset_hash
    with open(LOG_DIR / "dataset_config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"dataset_config.json written with {len(config)} keys")
    print(f"SHA-256: {dataset_hash}")
    return dataset_hash


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate publication synthetic fraud dataset.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default=str(TRAINING_DIR / "data" / "raw"))
    args = parser.parse_args()
    generate_and_save(args.output_dir, seed=args.seed)


if __name__ == "__main__":
    main()
