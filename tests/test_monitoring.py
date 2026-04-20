"""
Quick test to verify monitoring integration is working.
Run after starting the API with: python -m uvicorn backend.main:app --reload
"""

import requests
import json
from typing import Dict

BASE_URL = "http://localhost:8000"

def test_health_endpoint():
    """Test /health endpoint with updated metrics."""
    print("🔍 Testing /health endpoint...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"   Status: {response.status_code}")
    print(f"   Response: {json.dumps(response.json(), indent=2)}\n")

def test_metrics_endpoint():
    """Test /metrics endpoint returns Prometheus format."""
    print("🔍 Testing /metrics endpoint...")
    response = requests.get(f"{BASE_URL}/metrics")
    print(f"   Status: {response.status_code}")
    print(f"   Content-Type: {response.headers.get('content-type')}")
    
    # Check for key metrics
    content = response.text
    metrics_to_check = [
        'fraud_predictions_total',
        'fraud_prediction_latency_ms',
        'fraud_api_requests_total',
        'fraud_model_loaded',
        'fraud_database_connected',
    ]
    
    for metric in metrics_to_check:
        if metric in content:
            print(f"   ✓ Found metric: {metric}")
        else:
            print(f"   ✗ Missing metric: {metric}")
    
    print(f"\n   Sample output (first 500 chars):\n")
    print("   " + "\n   ".join(content[:500].split("\n")))
    print()

def test_prediction_with_metrics():
    """Test /predict endpoint records metrics."""
    print("🔍 Testing /predict endpoint (should record metrics)...")
    
    sample_claim = {
        "claim_id": "CLM-TEST-001",
        "patient_id": "PAT-001",
        "doctor_id": "DOC-001",
        "hospital_id": "HOSP-001",
        "claim_amount": 5000.0,
        "service_type": "surgery",
        "diagnosis_code": "J45.901",
        "procedure_code": "99213",
        "days_to_process": 45,
        "claim_frequency": 2,
        "explain": False
    }
    
    # Need API key for this
    headers = {
        "X-API-Key": "test-key-12345"  # You may need to check actual key in .env
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            json=sample_claim,
            headers=headers
        )
        print(f"   Status: {response.status_code}")
        
        if response.status_code == 200:
            print(f"   Response: {json.dumps(response.json(), indent=2)}\n")
        else:
            print(f"   Error: {response.text}\n")
    except Exception as e:
        print(f"   ✗ Request failed: {e}")
        print(f"   (This is expected if API key is not configured)\n")

def test_metrics_after_request():
    """Check if metrics changed after API request."""
    print("🔍 Checking if metrics increased...")
    response = requests.get(f"{BASE_URL}/metrics")
    content = response.text
    
    # Count lines with fraud_api_requests metrics
    request_lines = [line for line in content.split("\n") if "fraud_api_requests_total" in line and not line.startswith("#")]
    
    print(f"   Found {len(request_lines)} API request metric entries")
    if request_lines:
        print(f"   Sample entries:")
        for line in request_lines[:3]:
            print(f"      {line}")
    print()

def main():
    print("=" * 60)
    print("MONITORING INTEGRATION TEST")
    print("=" * 60)
    print()
    
    try:
        test_health_endpoint()
        test_metrics_endpoint()
        test_prediction_with_metrics()
        test_metrics_after_request()
        
        print("=" * 60)
        print("✓ All tests completed!")
        print("=" * 60)
        
    except requests.exceptions.ConnectionError:
        print("✗ Could not connect to API at http://localhost:8000")
        print("  Make sure the API is running:")
        print("  python -m uvicorn backend.main:app --reload")

if __name__ == "__main__":
    main()
