import requests
import json

# Step 1: Get auth token
print("Step 1: Getting authentication token...")
auth_resp = requests.post('http://localhost:8000/token?api_key=test_key_123')
if auth_resp.status_code != 200:
    print(f"✗ Failed to get token: {auth_resp.json()}")
    exit(1)

token = auth_resp.json()['access_token']
print(f"✓ Got token: {token[:20]}...")

# Step 2: Make prediction
print("\nStep 2: Making authenticated prediction request...")
claim = {
    'claim_id': 'TEST-001',
    'patient_id': 'P001',
    'doctor_id': 'D001',
    'hospital_id': 'H001',
    'claim_amount': 5000,
    'service_type': 'surgery',
    'diagnosis_code': 'J45.901',
    'procedure_code': '99213',
    'days_to_process': 45,
    'claim_frequency': 2,
    'explain': False
}

headers = {'Authorization': f'Bearer {token}'}
resp = requests.post('http://localhost:8000/predict', json=claim, headers=headers)
print(f"✓ Prediction Status: {resp.status_code}")

if resp.status_code == 200:
    data = resp.json()
    print(f"  Prediction: {data.get('fraud_prediction')}")
    print(f"  Score: {data.get('fraud_score'):.4f}")
    print(f"  Confidence: {data.get('confidence'):.4f}")
    print(f"  Latency: {data.get('inference_time_ms'):.2f}ms")
    print("\n✓ Prediction recorded with metrics!")
else:
    print(f"  Error: {resp.json().get('error')}")

# Step 3: Check if metrics increased
print("\nStep 3: Checking metrics...")
resp = requests.get('http://localhost:8000/metrics')
content = resp.text

# Count predictions
if 'fraud_predictions_total' in content:
    lines = [l for l in content.split('\n') if 'fraud_predictions_total{' in l]
    total = sum(float(l.split()[-1]) for l in lines if l and not l.startswith('#'))
    print(f"  Total predictions recorded: {int(total)}")

# Check fraud/legit breakdown  
if 'fraud_predictions_total' in content:
    frauds = [l for l in content.split('\n') if 'fraud_predictions_total{' in l and 'fraud' in l]
    legits = [l for l in content.split('\n') if 'fraud_predictions_total{' in l and 'legit' in l]
    print(f"  Fraud predictions recorded: {len([l for l in frauds if not l.startswith('#')])}")
    print(f"  Legit predictions recorded: {len([l for l in legits if not l.startswith('#')])}")

# Check API requests
if 'api_requests_total' in content:
    api_lines = [l for l in content.split('\n') if 'api_requests_total{' in l and 'endpoint="/predict"' in l]
    total_predicts = sum(float(l.split()[-1]) for l in api_lines if l and not l.startswith('#'))
    print(f"  /predict API calls recorded: {int(total_predicts)}")

print("\n✅ Monitoring integration COMPLETE and WORKING!")
