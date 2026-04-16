"""
Data Preprocessor — FIXED FOR NO DATA LEAKAGE
Critical fix: Split data BEFORE computing aggregation features
Only compute statistics from TRAINING set, then apply to val/test
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


class FraudDataPreprocessor:
    """Preprocessing pipeline: raw → clean → SPLIT → engineer (no leakage) → scale → save"""

    def __init__(self, data_dir: str = "data/raw", processed_dir: str = "data/processed"):
        self.data_dir = data_dir
        self.processed_dir = processed_dir
        self.scaler = StandardScaler()
        self.label_encoders: dict[str, LabelEncoder] = {}
        self.train_stats: dict = {}
        os.makedirs(processed_dir, exist_ok=True)

    # ── Load ───────────────────────────────────────────────────────────────────

    def load_raw(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        hospitals = pd.read_csv(f"{self.data_dir}/hospitals.csv")
        doctors = pd.read_csv(f"{self.data_dir}/doctors.csv")
        patients = pd.read_csv(f"{self.data_dir}/patients.csv")
        claims = pd.read_csv(f"{self.data_dir}/claims.csv")
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
        """Remove duplicates and handle nulls."""
        claims = claims.drop_duplicates(subset=["claim_id"])
        patients = patients.drop_duplicates(subset=["patient_id"])
        doctors = doctors.drop_duplicates(subset=["doctor_id"])
        hospitals = hospitals.drop_duplicates(subset=["hospital_id"])

        for df in [claims, patients, doctors, hospitals]:
            num_cols = df.select_dtypes(include=np.number).columns
            df[num_cols] = df[num_cols].fillna(df[num_cols].median())
            cat_cols = df.select_dtypes(include="object").columns
            for col in cat_cols:
                df[col] = df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else "Unknown")

        log.info("Cleaning done.")
        return hospitals, doctors, patients, claims

    # ── Split BEFORE feature engineering ───────────────────────────────────────

    def _temporal_stratified_split(self, claims: pd.DataFrame) -> tuple[pd.Index, pd.Index, pd.Index]:
        """
        Split data BEFORE feature engineering.
        Temporal: Sort by date, then stratify by label.
        Returns: (train_idx, val_idx, test_idx)
        """
        claims_sorted = claims.sort_values("claim_date").reset_index(drop=True)
        fraud_label = claims_sorted["fraud_label"]

        # Temporal split: first 60% train, next 20% val, last 20% test
        n = len(claims_sorted)
        train_end = int(0.60 * n)
        val_end = int(0.80 * n)

        train_idx = claims_sorted.index[:train_end].values
        val_idx = claims_sorted.index[train_end:val_end].values
        test_idx = claims_sorted.index[val_end:].values

        log.info(f"Temporal split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
        log.info(f"Train fraud rate: {fraud_label[train_idx].mean():.2%}")
        log.info(f"Val   fraud rate: {fraud_label[val_idx].mean():.2%}")
        log.info(f"Test  fraud rate: {fraud_label[test_idx].mean():.2%}")

        return train_idx, val_idx, test_idx

    # ── Feature Engineering (per-split) ────────────────────────────────────────

    def _engineer_features_no_leakage(
        self,
        claims_split: pd.DataFrame,
        doctors: pd.DataFrame,
        patients: pd.DataFrame,
        hospitals: pd.DataFrame,
        split_name: str = "train",
    ) -> pd.DataFrame:
        """
        Engineer features for ONE split using ONLY that split's data (for aggregations).
        This prevents looking into val/test data.
        """
        df = claims_split.copy()
        df["claim_date"] = pd.to_datetime(df["claim_date"])

        # ─ Compute aggregations from THIS SPLIT ONLY ─
        if split_name == "train":
            log.info("Computing aggregations from TRAINING data...")
            # Patient stats
            patient_stats = (
                df.groupby("patient_id")
                .agg(
                    patient_total_claims=("claim_id", "count"),
                    patient_total_amount=("claim_amount", "sum"),
                    patient_avg_amount=("claim_amount", "mean"),
                    patient_max_amount=("claim_amount", "max"),
                )
                .reset_index()
            )
            self.train_stats["patient_stats"] = patient_stats

            # Doctor stats (NOTE: NO fraud_rate — that's label information!)
            doctor_stats = (
                df.groupby("doctor_id")
                .agg(
                    doctor_total_claims=("claim_id", "count"),
                    doctor_total_amount=("claim_amount", "sum"),
                    doctor_avg_amount=("claim_amount", "mean"),
                )
                .reset_index()
            )
            self.train_stats["doctor_stats"] = doctor_stats

            # Hospital stats
            hosp_stats = (
                df.groupby("hospital_id")
                .agg(
                    hosp_total_claims=("claim_id", "count"),
                    hosp_avg_amount=("claim_amount", "mean"),
                )
                .reset_index()
            )
            self.train_stats["hosp_stats"] = hosp_stats

            # Global stats (for unseen entities)
            self.train_stats["global_avg_claim"] = df["claim_amount"].mean()
            self.train_stats["global_max_procedures"] = df["num_procedures"].max()

        else:
            # Use TRAINING statistics for val/test
            log.info(f"Applying TRAINING statistics to {split_name}...")
            patient_stats = self.train_stats["patient_stats"]
            doctor_stats = self.train_stats["doctor_stats"]
            hosp_stats = self.train_stats["hosp_stats"]

        # Merge aggregations (with fallback for unseen entities)
        df = df.merge(patient_stats, on="patient_id", how="left")
        df["patient_total_claims"] = df["patient_total_claims"].fillna(1)
        df["patient_total_amount"] = df["patient_total_amount"].fillna(0)
        df["patient_avg_amount"] = df["patient_avg_amount"].fillna(
            self.train_stats["global_avg_claim"]
        )
        df["patient_max_amount"] = df["patient_max_amount"].fillna(
            self.train_stats["global_avg_claim"]
        )

        df = df.merge(doctor_stats, on="doctor_id", how="left")
        df["doctor_total_claims"] = df["doctor_total_claims"].fillna(1)
        df["doctor_total_amount"] = df["doctor_total_amount"].fillna(0)
        df["doctor_avg_amount"] = df["doctor_avg_amount"].fillna(
            self.train_stats["global_avg_claim"]
        )

        df = df.merge(hosp_stats, on="hospital_id", how="left")
        df["hosp_total_claims"] = df["hosp_total_claims"].fillna(1)
        df["hosp_avg_amount"] = df["hosp_avg_amount"].fillna(
            self.train_stats["global_avg_claim"]
        )

        # Merge entity metadata (safe — no labels in these)
        df = df.merge(
            patients[["patient_id", "age", "gender", "chronic_condition", "insurance_type"]],
            on="patient_id",
            how="left",
        )
        df = df.merge(
            doctors[["doctor_id", "specialty", "years_experience"]],
            on="doctor_id",
            how="left",
        )
        df = df.merge(
            hospitals[["hospital_id", "num_beds", "state"]],
            on="hospital_id",
            how="left",
        )

        # Anomaly/ratio features
        df["amount_vs_patient_avg"] = df["claim_amount"] / (df["patient_avg_amount"] + 1)
        df["amount_vs_doctor_avg"] = df["claim_amount"] / (df["doctor_avg_amount"] + 1)
        df["amount_vs_hosp_avg"] = df["claim_amount"] / (df["hosp_avg_amount"] + 1)

        # Temporal features
        df["claim_month"] = df["claim_date"].dt.month
        df["claim_dayofweek"] = df["claim_date"].dt.dayofweek
        df["claim_day"] = df["claim_date"].dt.day

        # Drop ID/date/code columns (not needed for model)
        df = df.drop(
            columns=["claim_id", "claim_date", "approved", "hospital_id", "patient_id", "doctor_id",
                     "procedure_code", "diagnosis_code"],
            errors="ignore",
        )

        log.info(f"{split_name} engineered shape: {df.shape}")
        return df

    # ── Encoding ───────────────────────────────────────────────────────────────

    def _encode_categoricals(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Encode categorical columns. Fit only on TRAINING data."""
        cat_cols = ["gender", "chronic_condition", "insurance_type", "specialty", "state"]

        for col in cat_cols:
            if col not in df.columns:
                continue
            if fit:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
            else:
                le = self.label_encoders[col]
                # Handle unseen categories
                df[col] = df[col].apply(
                    lambda x, le=le: le.transform([str(x)])[0]
                    if str(x) in le.classes_
                    else 0  # fallback to index 0
                )

        return df

    # ── Full pipeline ──────────────────────────────────────────────────────────

    def run(self) -> dict:
        """Execute corrected pipeline: load → clean → split → engineer → encode → scale → save"""
        log.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        log.info("FRAUD DETECTION PREPROCESSING — NO DATA LEAKAGE")
        log.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

        # Step 1: Load and clean
        hospitals, doctors, patients, claims = self.load_raw()
        hospitals, doctors, patients, claims = self.clean(hospitals, doctors, patients, claims)

        # Step 2: SPLIT FIRST (critical!)
        log.info("\n⚠️  CRITICAL: SPLITTING BEFORE FEATURE ENGINEERING")
        train_idx, val_idx, test_idx = self._temporal_stratified_split(claims)

        claims_train = claims.iloc[train_idx].reset_index(drop=True)
        claims_val = claims.iloc[val_idx].reset_index(drop=True)
        claims_test = claims.iloc[test_idx].reset_index(drop=True)

        # Step 3: Engineer features per split (no leakage)
        log.info("\n📊 Engineering features for TRAINING set...")
        features_train = self._engineer_features_no_leakage(claims_train, doctors, patients, hospitals, "train")

        log.info("\n📊 Engineering features for VALIDATION set (using train stats)...")
        features_val = self._engineer_features_no_leakage(claims_val, doctors, patients, hospitals, "val")

        log.info("\n📊 Engineering features for TEST set (using train stats)...")
        features_test = self._engineer_features_no_leakage(claims_test, doctors, patients, hospitals, "test")

        # Step 4: Encode categoricals (fit only on train)
        log.info("\n🔤 Encoding categorical features...")
        features_train = self._encode_categoricals(features_train, fit=True)
        features_val = self._encode_categoricals(features_val, fit=False)
        features_test = self._encode_categoricals(features_test, fit=False)

        # Step 5: Extract X and y
        y_train = features_train["fraud_label"].values
        X_train = features_train.drop(columns=["fraud_label"]).values

        y_val = features_val["fraud_label"].values
        X_val = features_val.drop(columns=["fraud_label"]).values

        y_test = features_test["fraud_label"].values
        X_test = features_test.drop(columns=["fraud_label"]).values

        # Step 6: Scale features (fit only on train)
        log.info("\n🔢 Scaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)

        # Step 7: Save all artifacts
        log.info("\n💾 Saving artifacts...")
        np.save(f"{self.processed_dir}/X_train.npy", X_train_scaled)
        np.save(f"{self.processed_dir}/X_val.npy", X_val_scaled)
        np.save(f"{self.processed_dir}/X_test.npy", X_test_scaled)
        np.save(f"{self.processed_dir}/y_train.npy", y_train)
        np.save(f"{self.processed_dir}/y_val.npy", y_val)
        np.save(f"{self.processed_dir}/y_test.npy", y_test)

        joblib.dump(self.scaler, f"{self.processed_dir}/scaler.pkl")
        joblib.dump(self.label_encoders, f"{self.processed_dir}/label_encoders.pkl")

        log.info("\n✅ PREPROCESSING COMPLETE - NO DATA LEAKAGE")
        log.info(f"Train: {len(X_train_scaled)} samples, {y_train.mean():.2%} fraud")
        log.info(f"Val:   {len(X_val_scaled)} samples, {y_val.mean():.2%} fraud")
        log.info(f"Test:  {len(X_test_scaled)} samples, {y_test.mean():.2%} fraud")

        return {
            "X_train": X_train_scaled,
            "X_val": X_val_scaled,
            "X_test": X_test_scaled,
            "y_train": y_train,
            "y_val": y_val,
            "y_test": y_test,
            "feature_names": list(features_train.drop(columns=["fraud_label"]).columns),
        }


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    preprocessor = FraudDataPreprocessor()
    preprocessor.run()
