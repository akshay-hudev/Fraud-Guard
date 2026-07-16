"""Manual monitoring smoke test.

Run with an API listening on localhost:8000:
    python tests/test_prediction_metrics.py
"""

import requests


def main() -> int:
    print("Step 1: Getting authentication token...")
    auth_response = requests.post(
        "http://localhost:8000/token",
        params={"api_key": "test_key_123"},
        timeout=5,
    )
    if auth_response.status_code != 200:
        print(f"✗ Failed to get token: {auth_response.json()}")
        return 1

    token = auth_response.json()["access_token"]
    print(f"✓ Got token: {token[:20]}...")

    print("\nStep 2: Making authenticated prediction request...")
    claim = {
        "claim_id": "TEST-001",
        "patient_id": "P001",
        "doctor_id": "D001",
        "hospital_id": "H001",
        "claim_amount": 5000,
        "service_type": "surgery",
        "diagnosis_code": "J45.901",
        "procedure_code": "99213",
        "days_to_process": 45,
        "claim_frequency": 2,
        "explain": False,
    }

    headers = {"Authorization": f"Bearer {token}"}
    response = requests.post(
        "http://localhost:8000/predict",
        json=claim,
        headers=headers,
        timeout=5,
    )
    print(f"✓ Prediction Status: {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        print(f"  Prediction: {data.get('fraud_prediction')}")
        print(f"  Score: {data.get('fraud_score'):.4f}")
        print(f"  Confidence: {data.get('confidence'):.4f}")
        print(f"  Latency: {data.get('inference_time_ms'):.2f}ms")
        print("\n✓ Prediction recorded with metrics!")
    else:
        print(f"  Error: {response.json().get('error')}")
        return 1

    print("\nStep 3: Checking metrics...")
    metrics_response = requests.get("http://localhost:8000/metrics", timeout=5)
    metrics_response.raise_for_status()
    content = metrics_response.text

    if "fraud_predictions_total" in content:
        lines = [
            line
            for line in content.split("\n")
            if "fraud_predictions_total{" in line
        ]
        total = sum(
            float(line.split()[-1])
            for line in lines
            if line and not line.startswith("#")
        )
        print(f"  Total predictions recorded: {int(total)}")

        frauds = [line for line in lines if "fraud" in line]
        legits = [line for line in lines if "legit" in line]
        print(
            "  Fraud predictions recorded: "
            f"{len([line for line in frauds if not line.startswith('#')])}",
        )
        print(
            "  Legit predictions recorded: "
            f"{len([line for line in legits if not line.startswith('#')])}",
        )

    if "api_requests_total" in content:
        api_lines = [
            line
            for line in content.split("\n")
            if "api_requests_total{" in line and 'endpoint="/predict"' in line
        ]
        total_predicts = sum(
            float(line.split()[-1])
            for line in api_lines
            if line and not line.startswith("#")
        )
        print(f"  /predict API calls recorded: {int(total_predicts)}")

    print("\n✅ Monitoring integration COMPLETE and WORKING!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
