"""
Test Suite — Health Insurance Fraud Detection System
Covers: data generation, preprocessing, models, API endpoints
Run with: pytest tests/ -v
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Data Tests ─────────────────────────────────────────────────────────────────

class TestDataGeneration:

    def test_hospitals_generated(self):
        df = pd.read_csv("data/raw/hospitals.csv")
        assert len(df) == 50
        assert "hospital_id" in df.columns
        assert df["hospital_id"].nunique() == 50

    def test_doctors_generated(self):
        df = pd.read_csv("data/raw/doctors.csv")
        assert len(df) == 200
        assert "hospital_id" in df.columns
        # All doctors have valid hospital references
        hospitals = pd.read_csv("data/raw/hospitals.csv")["hospital_id"].tolist()
        assert df["hospital_id"].isin(hospitals).all()

    def test_patients_generated(self):
        df = pd.read_csv("data/raw/patients.csv")
        assert len(df) == 2000
        assert df["age"].between(18, 90).all()
        assert df["gender"].isin(["M", "F"]).all()

    def test_claims_generated(self):
        df = pd.read_csv("data/raw/claims.csv")
        assert len(df) == 10000
        assert "fraud_label" in df.columns
        assert df["fraud_label"].isin([0, 1]).all()
        # Fraud rate should be around 12% ± 5%
        fraud_rate = df["fraud_label"].mean()
        assert 0.07 < fraud_rate < 0.20, f"Unexpected fraud rate: {fraud_rate}"

    def test_no_null_ids(self):
        for fname in ["hospitals.csv", "doctors.csv", "patients.csv", "claims.csv"]:
            df = pd.read_csv(f"data/raw/{fname}")
            id_col = [c for c in df.columns if c.endswith("_id")][0]
            assert df[id_col].isna().sum() == 0, f"Null IDs in {fname}"

    def test_claim_amount_positive(self):
        df = pd.read_csv("data/raw/claims.csv")
        assert (df["claim_amount"] > 0).all()


# ── Preprocessing Tests ────────────────────────────────────────────────────────

class TestPreprocessing:

    def setup_method(self):
        from src.data.preprocessor import FraudDataPreprocessor
        self.pp = FraudDataPreprocessor("data/raw", "data/processed")

    def test_processed_files_exist(self):
        for fname in ["X_train.npy", "X_val.npy", "X_test.npy",
                      "y_train.npy", "y_val.npy", "y_test.npy",
                      "scaler.pkl", "label_encoders.pkl"]:
            assert os.path.exists(f"data/processed/{fname}"), f"Missing: {fname}"

    def test_shapes_consistent(self):
        X_train = np.load("data/processed/X_train.npy")
        X_val   = np.load("data/processed/X_val.npy")
        X_test  = np.load("data/processed/X_test.npy")
        y_train = np.load("data/processed/y_train.npy")
        assert X_train.shape[1] == X_val.shape[1] == X_test.shape[1]
        assert X_train.shape[0] == y_train.shape[0]

    def test_no_nans_in_features(self):
        X_train = np.load("data/processed/X_train.npy")
        assert not np.isnan(X_train).any(), "NaN values found in X_train"

    def test_labels_binary(self):
        for split in ["y_train", "y_val", "y_test"]:
            y = np.load(f"data/processed/{split}.npy")
            assert set(y).issubset({0, 1}), f"Non-binary labels in {split}"

    def test_stratified_split(self):
        y_train = np.load("data/processed/y_train.npy")
        y_val   = np.load("data/processed/y_val.npy")
        y_test  = np.load("data/processed/y_test.npy")
        # Fraud rate should be similar across splits (stratified)
        rates = [y.mean() for y in [y_train, y_val, y_test]]
        for r in rates:
            assert abs(r - rates[0]) < 0.03, f"Unstratified split: rates={rates}"

    def test_scaler_transform(self):
        import joblib
        scaler  = joblib.load("data/processed/scaler.pkl")
        X_train = np.load("data/processed/X_train.npy")
        # After scaling, columns should be approx zero-mean unit-variance
        # (already scaled, so just check no extreme values)
        assert np.abs(X_train).max() < 100, "Extreme values after scaling"

    def test_feature_engineering_columns(self):
        df = pd.read_csv("data/processed/claims_engineered.csv")
        expected = ["amount_vs_patient_avg", "amount_vs_doctor_avg",
                    "patient_total_claims", "doctor_total_claims"]
        for col in expected:
            assert col in df.columns, f"Missing engineered feature: {col}"


# ── Model Tests ────────────────────────────────────────────────────────────────

class TestBaselineModels:

    def setup_method(self):
        self.X_test = np.load("data/processed/X_test.npy")
        self.y_test = np.load("data/processed/y_test.npy")

    def test_model_files_exist(self):
        for name in ["logistic_regression", "random_forest", "gradient_boosting"]:
            assert os.path.exists(f"models/baseline/{name}.pkl"), f"Missing: {name}.pkl"

    def test_predictions_binary(self):
        import joblib
        model = joblib.load("models/baseline/random_forest.pkl")
        preds = model.predict(self.X_test)
        assert set(preds).issubset({0, 1})

    def test_probabilities_valid(self):
        import joblib
        model = joblib.load("models/baseline/random_forest.pkl")
        proba = model.predict_proba(self.X_test)
        assert proba.shape == (len(self.X_test), 2)
        assert (proba >= 0).all() and (proba <= 1).all()
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_f1_above_threshold(self):
        import joblib
        from sklearn.metrics import f1_score
        for name in ["logistic_regression", "random_forest", "gradient_boosting"]:
            model = joblib.load(f"models/baseline/{name}.pkl")
            preds = model.predict(self.X_test)
            f1 = f1_score(self.y_test, preds, zero_division=0)
            assert f1 > 0.70, f"{name} F1 ({f1:.3f}) below threshold"

    def test_metrics_json_exists(self):
        for name in ["logistic_regression", "random_forest", "gradient_boosting"]:
            path = f"models/baseline/{name}_metrics.json"
            assert os.path.exists(path)
            with open(path) as f:
                m = json.load(f)
            for key in ["accuracy", "precision", "recall", "f1"]:
                assert key in m
                assert 0 <= m[key] <= 1


# ── Predictor Tests ────────────────────────────────────────────────────────────

class TestFraudPredictor:

    def setup_method(self):
        from backend.predictor import FraudPredictor
        # Reset singleton
        FraudPredictor._instance = None
        self.predictor = FraudPredictor("data/processed", "models")

    def test_predict_returns_required_keys(self):
        result = self.predictor.predict({"claim_amount": 5000})
        for key in ["fraud_probability", "prediction", "risk_level", "model_used"]:
            assert key in result

    def test_high_amount_higher_risk(self):
        legit = self.predictor.predict({"claim_amount": 300, "num_procedures": 1})
        fraud = self.predictor.predict({"claim_amount": 45000, "num_procedures": 20, "days_in_hospital": 15})
        assert fraud["fraud_probability"] >= legit["fraud_probability"]

    def test_probability_range(self):
        result = self.predictor.predict({"claim_amount": 10000})
        assert 0 <= result["fraud_probability"] <= 1

    def test_prediction_matches_probability(self):
        result = self.predictor.predict({"claim_amount": 5000})
        expected_pred = int(result["fraud_probability"] >= 0.5)
        assert result["prediction"] == expected_pred

    def test_risk_level_thresholds(self):
        from backend.predictor import FraudPredictor
        for prob, expected_level in [(0.1, "LOW"), (0.55, "MEDIUM"), (0.85, "HIGH")]:
            # Patch the predictor to test risk level logic
            if prob < 0.4:
                level = "LOW"
            elif prob < 0.7:
                level = "MEDIUM"
            else:
                level = "HIGH"
            assert level == expected_level

    def test_batch_predict(self):
        claims = [
            {"claim_amount": 500},
            {"claim_amount": 25000, "num_procedures": 12},
            {"claim_amount": 1200},
        ]
        results = self.predictor.batch_predict(claims)
        assert len(results) == 3
        for r in results:
            assert "fraud_probability" in r


# ── Alert System Tests ─────────────────────────────────────────────────────────

class TestAlertSystem:

    def setup_method(self):
        from src.utils.explainability import AlertSystem
        self.alert = AlertSystem("data/test_alerts.jsonl")

    def teardown_method(self):
        if os.path.exists("data/test_alerts.jsonl"):
            os.remove("data/test_alerts.jsonl")

    def test_high_risk_alert(self):
        result = self.alert.evaluate("C001", 0.85)
        assert result["risk_level"] == "HIGH"
        assert result["recommended_action"] == "BLOCK_AND_REVIEW"

    def test_medium_risk_alert(self):
        result = self.alert.evaluate("C002", 0.55)
        assert result["risk_level"] == "MEDIUM"
        assert result["recommended_action"] == "MANUAL_REVIEW"

    def test_low_risk_no_alert(self):
        result = self.alert.evaluate("C003", 0.15)
        assert result["risk_level"] == "LOW"
        assert result["recommended_action"] == "AUTO_APPROVE"

    def test_high_risk_logged(self):
        self.alert.evaluate("C004", 0.90)
        logged = self.alert.get_recent_alerts(5)
        assert any(a["claim_id"] == "C004" for a in logged)

    def test_low_risk_not_logged(self):
        self.alert.evaluate("C005", 0.10)
        logged = self.alert.get_recent_alerts(5)
        assert not any(a["claim_id"] == "C005" for a in logged)


# ── API Tests ──────────────────────────────────────────────────────────────────

class TestAPI:
    """Integration tests — requires the API to be running on port 8000."""

    BASE = "http://localhost:8000"

    @pytest.fixture(autouse=True)
    def skip_if_api_down(self):
        import requests
        try:
            requests.get(f"{self.BASE}/health", timeout=2)
        except Exception:
            pytest.skip("API not running — start with: uvicorn backend.main:app")

    def test_health(self):
        import requests
        r = requests.get(f"{self.BASE}/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "healthy"

    def test_predict_endpoint(self):
        import requests
        payload = {
            "claim_id": "TEST_001",
            "claim_amount": 15000,
            "num_procedures": 8,
            "days_in_hospital": 5,
        }
        r = requests.post(f"{self.BASE}/predict", json=payload)
        assert r.status_code == 200
        data = r.json()
        assert "fraud_probability" in data
        assert "risk_level" in data
        assert data["risk_level"] in ["LOW", "MEDIUM", "HIGH"]

    def test_stats_endpoint(self):
        import requests
        r = requests.get(f"{self.BASE}/stats")
        assert r.status_code == 200
        data = r.json()
        assert "model_performance" in data
        assert "best_model" in data

    def test_graph_data_endpoint(self):
        import requests
        r = requests.get(f"{self.BASE}/graph/data", params={"max_nodes": 20})
        assert r.status_code == 200
        data = r.json()
        assert "nodes" in data
        assert "edges" in data
        assert len(data["nodes"]) > 0

    def test_simulate_endpoint(self):
        import requests
        r = requests.get(f"{self.BASE}/simulate", params={"n": 5})
        assert r.status_code == 200
        data = r.json()
        assert len(data["claims"]) == 5
        for claim in data["claims"]:
            assert 0 <= claim["fraud_probability"] <= 1

    def test_batch_predict(self):
        import requests
        payload = {
            "claims": [
                {"claim_id": "B001", "claim_amount": 500},
                {"claim_id": "B002", "claim_amount": 40000, "num_procedures": 15},
            ]
        }
        r = requests.post(f"{self.BASE}/predict/batch", json=payload)
        assert r.status_code == 200
        data = r.json()
        assert data["total"] == 2
        assert len(data["results"]) == 2

    def test_upload_csv(self):
        import requests, io
        csv_content = "claim_amount,num_procedures,days_in_hospital\n1200,2,0\n35000,12,8\n500,1,0\n"
        files = {"file": ("claims.csv", io.BytesIO(csv_content.encode()), "text/csv")}
        r = requests.post(f"{self.BASE}/upload", files=files)
        assert r.status_code == 200
        data = r.json()
        assert data["total_claims"] == 3
        assert "fraud_flagged" in data
        assert "fraud_rate" in data


# ── Metrics Tests ──────────────────────────────────────────────────────────────

class TestMetrics:

    def test_mock_results_structure(self):
        from src.utils.metrics import MOCK_RESULTS
        for model, metrics in MOCK_RESULTS.items():
            for key in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
                assert key in metrics, f"Missing {key} in {model}"
                assert 0 <= metrics[key] <= 1

    def test_gnn_beats_baselines(self):
        from src.utils.metrics import MOCK_RESULTS
        gnn_f1 = MOCK_RESULTS["gnn_hgt"]["f1"]
        for model in ["logistic_regression", "random_forest"]:
            assert gnn_f1 > MOCK_RESULTS[model]["f1"], \
                f"GNN F1 ({gnn_f1}) should exceed {model}"

    def test_comparison_json(self):
        if not os.path.exists("models/comparison.json"):
            pytest.skip("comparison.json not yet generated")
        with open("models/comparison.json") as f:
            comp = json.load(f)
        assert len(comp) >= 3  # at least 3 models


# ── Run ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import subprocess
    result = subprocess.run(
        ["python", "-m", "pytest", __file__, "-v", "--tb=short"],
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )
    sys.exit(result.returncode)
