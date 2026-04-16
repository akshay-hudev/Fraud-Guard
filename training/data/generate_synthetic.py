"""
Synthetic Healthcare Fraud Dataset Generator
Generates realistic patient, doctor, hospital, and claim data
with embedded fraud patterns for model training.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import os

np.random.seed(42)
random.seed(42)

# ── Configuration ──────────────────────────────────────────────────────────────
N_PATIENTS  = 2_000
N_DOCTORS   = 200
N_HOSPITALS = 50
N_CLAIMS    = 10_000
FRAUD_RATE  = 0.12   # 12% fraud


def _date_range(start_days_ago: int, end_days_ago: int = 0) -> datetime:
    delta = start_days_ago - end_days_ago
    return datetime.now() - timedelta(days=random.randint(end_days_ago, start_days_ago))


# ── Entity generators ──────────────────────────────────────────────────────────

def generate_hospitals(n: int) -> pd.DataFrame:
    specialties = ["General", "Cardiology", "Orthopedics", "Neurology", "Oncology",
                   "Pediatrics", "Emergency", "Surgery", "Psychiatry", "Dermatology"]
    states = ["CA", "TX", "NY", "FL", "IL", "PA", "OH", "GA", "NC", "MI"]
    records = []
    for i in range(n):
        records.append({
            "hospital_id": f"H{i:04d}",
            "hospital_name": f"Hospital_{i}",
            "specialty": random.choice(specialties),
            "state": random.choice(states),
            "num_beds": random.randint(50, 800),
            "is_accredited": random.random() > 0.1,
            "fraud_prone": random.random() < 0.15,   # 15% hospitals are high-risk
        })
    return pd.DataFrame(records)


def generate_doctors(n: int, hospitals: pd.DataFrame) -> pd.DataFrame:
    specialties = ["General Practitioner", "Cardiologist", "Orthopedic Surgeon",
                   "Neurologist", "Oncologist", "Pediatrician", "Psychiatrist",
                   "Dermatologist", "Emergency Physician", "Radiologist"]
    records = []
    for i in range(n):
        hospital = hospitals.sample(1).iloc[0]
        fraud_prone = hospital["fraud_prone"] or (random.random() < 0.10)
        records.append({
            "doctor_id": f"D{i:04d}",
            "doctor_name": f"Dr_Doctor_{i}",
            "specialty": random.choice(specialties),
            "hospital_id": hospital["hospital_id"],
            "years_experience": random.randint(1, 40),
            "avg_claims_per_month": random.randint(5, 80),
            "license_valid": random.random() > 0.05,
            "fraud_prone": fraud_prone,
        })
    return pd.DataFrame(records)


def generate_patients(n: int) -> pd.DataFrame:
    conditions = ["Diabetes", "Hypertension", "Asthma", "Cancer", "Heart Disease",
                  "Arthritis", "None", "None", "None", "None"]
    records = []
    for i in range(n):
        age = random.randint(18, 85)
        records.append({
            "patient_id": f"P{i:05d}",
            "age": age,
            "gender": random.choice(["M", "F"]),
            "chronic_condition": random.choice(conditions),
            "insurance_type": random.choice(["Medicare", "Medicaid", "Private", "Self-Pay"]),
            "state": random.choice(["CA", "TX", "NY", "FL", "IL"]),
            "member_since_years": random.randint(0, 20),
            "fraud_prone": random.random() < 0.08,
        })
    return pd.DataFrame(records)


def generate_claims(
    n: int,
    patients: pd.DataFrame,
    doctors: pd.DataFrame,
    hospitals: pd.DataFrame,
) -> pd.DataFrame:
    procedure_codes = [f"PC{i:03d}" for i in range(1, 51)]
    diagnosis_codes = [f"DX{i:03d}" for i in range(1, 41)]

    records = []
    for i in range(n):
        # Determine if this claim will be fraudulent
        is_fraud = random.random() < FRAUD_RATE
        
        # Random selection (no fraud_prone flags used — they're removed!)
        patient = patients.sample(1).iloc[0]
        doctor  = doctors.sample(1).iloc[0]
        hospital = hospitals[hospitals["hospital_id"] == doctor["hospital_id"]].iloc[0]

        # Realistic fraud patterns — SUBTLE to avoid leakage
        if is_fraud:
            # Fraudsters don't always bill excessively — make patterns subtle
            claim_amount = random.uniform(2_000, 15_000)    # Somewhat higher, not 50k
            num_procedures = random.randint(3, 10)           # Slightly higher, not obviously fraudulent
            days_in_hospital = random.randint(1, 14)
            # ~70% of fraud gets caught/flagged, ~30% slips through
            approved = random.random() < 0.3
        else:
            claim_amount = random.uniform(100, 8_000)
            num_procedures = random.randint(1, 5)
            days_in_hospital = random.randint(0, 5)
            approved = random.random() < 0.95  # Most legit claims approved

        claim_date = _date_range(730)
        records.append({
            "claim_id":        f"C{i:06d}",
            "patient_id":      patient["patient_id"],
            "doctor_id":       doctor["doctor_id"],
            "hospital_id":     hospital["hospital_id"],
            "claim_date":      claim_date.strftime("%Y-%m-%d"),
            "claim_amount":    round(claim_amount, 2),
            "num_procedures":  num_procedures,
            "procedure_code":  random.choice(procedure_codes),
            "diagnosis_code":  random.choice(diagnosis_codes),
            "days_in_hospital": days_in_hospital,
            "approved":        approved,  # DO NOT use this as target — it's correlated
            "fraud_label":     int(is_fraud),
        })

    return pd.DataFrame(records)


# ── Main ───────────────────────────────────────────────────────────────────────

def generate_and_save(output_dir: str = "data/raw") -> None:
    os.makedirs(output_dir, exist_ok=True)

    print("Generating hospitals...")
    hospitals = generate_hospitals(N_HOSPITALS)

    print("Generating doctors...")
    doctors = generate_doctors(N_DOCTORS, hospitals)

    print("Generating patients...")
    patients = generate_patients(N_PATIENTS)

    print("Generating claims...")
    claims = generate_claims(N_CLAIMS, patients, doctors, hospitals)

    # Save
    hospitals.to_csv(f"{output_dir}/hospitals.csv", index=False)
    doctors.to_csv(f"{output_dir}/doctors.csv",     index=False)
    patients.to_csv(f"{output_dir}/patients.csv",   index=False)
    claims.to_csv(f"{output_dir}/claims.csv",       index=False)

    print(f"\n✅ Dataset saved to '{output_dir}/'")
    print(f"   Hospitals : {len(hospitals)}")
    print(f"   Doctors   : {len(doctors)}")
    print(f"   Patients  : {len(patients)}")
    print(f"   Claims    : {len(claims)}  (fraud: {claims['fraud_label'].sum()}, "
          f"{claims['fraud_label'].mean():.1%})")


if __name__ == "__main__":
    generate_and_save()
