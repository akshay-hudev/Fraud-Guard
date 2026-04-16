"""
Data Preprocessor
Handles loading, cleaning, feature engineering, and train/val/test splits
for the healthcare fraud detection dataset.
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


class FraudDataPreprocessor:
    """Full preprocessing pipeline: raw CSVs → feature matrix ready for ML/GNN."""

    def __init__(self, data_dir: str = "data/raw", processed_dir: str = "data/processed"):
        self.data_dir      = data_dir
        self.processed_dir = processed_dir
        self.scaler        = StandardScaler()
        self.label_encoders: dict[str, LabelEncoder] = {}
        os.makedirs(processed_dir, exist_ok=True)

    # ── Load ───────────────────────────────────────────────────────────────────

    def load_raw(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        hospitals = pd.read_csv(f"{self.data_dir}/hospitals.csv")
        doctors   = pd.read_csv(f"{self.data_dir}/doctors.csv")
        patients  = pd.read_csv(f"{self.data_dir}/patients.csv")
        claims    = pd.read_csv(f"{self.data_dir}/claims.csv")
        log.info("Raw data loaded: %d claims, %d patients, %d doctors, %d hospitals",
                 len(claims), len(patients), len(doctors), len(hospitals))
        return hospitals, doctors, patients, claims

    # ── Clean ──────────────────────────────────────────────────────────────────

    def clean(
        self,
        hospitals: pd.DataFrame,
        doctors: pd.DataFrame,
        patients: pd.DataFrame,
        claims: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        # Drop exact duplicates
        claims    = claims.drop_duplicates(subset=["claim_id"])
        patients  = patients.drop_duplicates(subset=["patient_id"])
        doctors   = doctors.drop_duplicates(subset=["doctor_id"])
        hospitals = hospitals.drop_duplicates(subset=["hospital_id"])

        # Fill numeric nulls with median
        for df in [claims, patients, doctors, hospitals]:
            num_cols = df.select_dtypes(include=np.number).columns
            df[num_cols] = df[num_cols].fillna(df[num_cols].median())

        # Fill categorical nulls with mode
        for df in [claims, patients, doctors, hospitals]:
            cat_cols = df.select_dtypes(include="object").columns
            for col in cat_cols:
                df[col] = df[col].fillna(df[col].mode()[0])

        log.info("Cleaning done.")
        return hospitals, doctors, patients, claims

    # ── Feature Engineering ────────────────────────────────────────────────────

    def engineer_features(
        self,
        hospitals: pd.DataFrame,
        doctors: pd.DataFrame,
        patients: pd.DataFrame,
        claims: pd.DataFrame,
    ) -> pd.DataFrame:
        """Merge all entities into one wide feature dataframe."""

        df = claims.copy()
        df["claim_date"] = pd.to_datetime(df["claim_date"])

        # ── Patient-level aggregations ─────────────────────────────────────────
        patient_stats = (
            df.groupby("patient_id")
            .agg(
                patient_total_claims      = ("claim_id",     "count"),
                patient_total_amount      = ("claim_amount", "sum"),
                patient_avg_amount        = ("claim_amount", "mean"),
                patient_max_amount        = ("claim_amount", "max"),
                patient_unique_doctors    = ("doctor_id",    "nunique"),
                patient_unique_hospitals  = ("hospital_id",  "nunique"),
            )
            .reset_index()
        )
        df = df.merge(patient_stats, on="patient_id", how="left")

        # ── Doctor-level aggregations ──────────────────────────────────────────
        doctor_stats = (
            df.groupby("doctor_id")
            .agg(
                doctor_total_claims     = ("claim_id",     "count"),
                doctor_total_amount     = ("claim_amount", "sum"),
                doctor_avg_amount       = ("claim_amount", "mean"),
                doctor_unique_patients  = ("patient_id",   "nunique"),
                doctor_fraud_rate       = ("fraud_label",  "mean"),
            )
            .reset_index()
        )
        df = df.merge(doctor_stats, on="doctor_id", how="left")

        # ── Hospital-level aggregations ────────────────────────────────────────
        hosp_stats = (
            df.groupby("hospital_id")
            .agg(
                hosp_total_claims  = ("claim_id",     "count"),
                hosp_avg_amount    = ("claim_amount", "mean"),
                hosp_fraud_rate    = ("fraud_label",  "mean"),
            )
            .reset_index()
        )
        df = df.merge(hosp_stats, on="hospital_id", how="left")

        # ── Merge entity metadata ──────────────────────────────────────────────
        df = df.merge(
            patients[["patient_id", "age", "gender", "chronic_condition",
                       "insurance_type", "member_since_years"]],
            on="patient_id", how="left",
        )
        df = df.merge(
            doctors[["doctor_id", "specialty", "years_experience",
                      "avg_claims_per_month", "license_valid"]],
            on="doctor_id", how="left",
        )
        df = df.merge(
            hospitals[["hospital_id", "num_beds", "is_accredited", "state"]],
            on="hospital_id", how="left",
        )

        # ── Anomaly / ratio features ───────────────────────────────────────────
        df["amount_vs_patient_avg"]  = df["claim_amount"] / (df["patient_avg_amount"]  + 1)
        df["amount_vs_doctor_avg"]   = df["claim_amount"] / (df["doctor_avg_amount"]   + 1)
        df["amount_vs_hosp_avg"]     = df["claim_amount"] / (df["hosp_avg_amount"]     + 1)
        df["claim_month"]            = df["claim_date"].dt.month
        df["claim_dayofweek"]        = df["claim_date"].dt.dayofweek
        df["license_valid"]          = df["license_valid"].astype(int)
        df["is_accredited"]          = df["is_accredited"].astype(int)

        log.info("Feature engineering done. Shape: %s", df.shape)
        return df

    # ── Encode & Scale ─────────────────────────────────────────────────────────

    def encode_and_scale(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        cat_cols = ["gender", "chronic_condition", "insurance_type",
                    "specialty", "state", "procedure_code", "diagnosis_code"]

        for col in cat_cols:
            if col not in df.columns:
                continue
            if fit:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
            else:
                le = self.label_encoders[col]
                df[col] = df[col].map(
                    lambda x, le=le: le.transform([str(x)])[0]
                    if str(x) in le.classes_ else -1
                )

        # Drop non-numeric / id columns
        drop_cols = ["claim_id", "patient_id", "doctor_id", "hospital_id",
                     "claim_date", "approved"]
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])

        # Scale numeric features (exclude label)
        feature_cols = [c for c in df.columns if c != "fraud_label"]
        if fit:
            df[feature_cols] = self.scaler.fit_transform(df[feature_cols])
        else:
            df[feature_cols] = self.scaler.transform(df[feature_cols])

        return df

    # ── Full pipeline ──────────────────────────────────────────────────────────

    def run(self, test_size: float = 0.15, val_size: float = 0.15) -> dict:
        hospitals, doctors, patients, claims = self.load_raw()
        hospitals, doctors, patients, claims = self.clean(hospitals, doctors, patients, claims)
        features_df = self.engineer_features(hospitals, doctors, patients, claims)

        # Save processed before encoding (for graph builder)
        features_df.to_csv(f"{self.processed_dir}/features_raw.csv", index=False)

        encoded = self.encode_and_scale(features_df.copy(), fit=True)

        X = encoded.drop(columns=["fraud_label"]).values
        y = encoded["fraud_label"].values

        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=test_size + val_size, random_state=42, stratify=y
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=test_size / (test_size + val_size),
            random_state=42, stratify=y_temp,
        )

        log.info("Split sizes — train: %d | val: %d | test: %d",
                 len(X_train), len(X_val), len(X_test))

        # Persist artefacts
        joblib.dump(self.scaler,         f"{self.processed_dir}/scaler.pkl")
        joblib.dump(self.label_encoders, f"{self.processed_dir}/label_encoders.pkl")
        np.save(f"{self.processed_dir}/X_train.npy", X_train)
        np.save(f"{self.processed_dir}/X_val.npy",   X_val)
        np.save(f"{self.processed_dir}/X_test.npy",  X_test)
        np.save(f"{self.processed_dir}/y_train.npy", y_train)
        np.save(f"{self.processed_dir}/y_val.npy",   y_val)
        np.save(f"{self.processed_dir}/y_test.npy",  y_test)

        # Also save the original (un-encoded) claims with IDs for graph builder
        claims_with_features = features_df.copy()
        claims_with_features.to_csv(f"{self.processed_dir}/claims_engineered.csv", index=False)

        log.info("✅ Preprocessing complete. Artefacts saved to '%s'", self.processed_dir)
        return {
            "X_train": X_train, "X_val": X_val, "X_test": X_test,
            "y_train": y_train, "y_val": y_val,  "y_test": y_test,
            "feature_names": list(encoded.drop(columns=["fraud_label"]).columns),
            "hospitals": hospitals,
            "doctors":   doctors,
            "patients":  patients,
            "claims":    claims,
        }


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    preprocessor = FraudDataPreprocessor()
    preprocessor.run()
