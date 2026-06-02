# Health Insurance Fraud Detection

Production-style ML system for detecting fraudulent healthcare claims. The repo includes a leakage-safe training pipeline (tabular baselines + HGT-style GNN), a FastAPI scoring API, and a Streamlit dashboard. The API currently serves the Random Forest baseline model; GNN training is implemented for offline evaluation and comparison.

## Key capabilities
- Synthetic dataset generator and temporal split preprocessing (no leakage).
- Baseline models: Logistic Regression, Random Forest, Gradient Boosting.
- Heterogeneous GNN (HGT-style) training and graph construction.
- FastAPI backend with JWT auth, model registry, drift checks, Prometheus metrics, and audit logging.
- Data quality monitoring, performance caching, interpretability, explainability, compliance, and resilience modules.
- Streamlit UI for single-claim scoring, bulk upload, data quality, performance, and resilience views.

## Repository layout
```
backend/            FastAPI app, auth, monitoring, compliance, resilience
frontend/           Streamlit dashboard
training/           Training module (data, models, scripts, notebooks)
data/               Raw and processed datasets (generated)
models/             Trained model artifacts for API runtime
logs/               Metrics, comparisons, and reports
figures/            Generated plots
docker/             Dockerfiles and docker-compose
tests/              Retraining and monitoring checks
paper/              Paper notes and artifacts
```

Top-level scripts:
- generate_dataset.py, run_pipeline.py, threshold_analysis.py, train_real_world.py
- verify.py (checks expected outputs after running the pipeline)

## Setup
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

# Optional: edit defaults
copy .env.example .env
```

## Training and reproducibility
Run from the repository root unless noted.

Full pipeline (data generation, preprocessing, baselines, optional GNN):
```bash
python run_pipeline.py
python run_pipeline.py --skip-gnn
python run_pipeline.py --skip-generate
```

Threshold sweep (writes logs/threshold_sweep.json and figures/threshold_sensitivity.png):
```bash
python threshold_analysis.py
```

Kaggle provider fraud experiment:
```bash
python train_real_world.py --data-dir path/to/kaggle_healthcare_provider_fraud
```

Verify expected output files:
```bash
python verify.py
```

Notes:
- GNN training requires PyTorch Geometric and related packages. The pipeline skips GNN if they are missing.
- Training outputs are written to data/, models/, logs/, and figures/ in the repo root.

## Run the API
```bash
uvicorn backend.main:app --reload --port 8000
```

Get a token and make a prediction (default dev key is test_key_123 in backend/security.py):
```bash
curl -X POST "http://localhost:8000/token?api_key=test_key_123"

curl -X POST "http://localhost:8000/predict" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "claim_id": "CLM_001",
    "patient_id": "PAT_001",
    "doctor_id": "DOC_001",
    "hospital_id": "HOS_001",
    "claim_amount": 15000,
    "num_procedures": 3,
    "days_in_hospital": 2,
    "age": 45,
    "gender": "M",
    "insurance_type": "PPO",
    "specialty": "Cardiology",
    "explain": true
  }'
```

API docs:
- http://localhost:8000/docs
- http://localhost:8000/redoc

Prometheus metrics:
- http://localhost:8000/metrics

## Streamlit dashboard
```bash
streamlit run frontend/app.py --server.port 8501
```

Note: The API base URL is set in frontend/app.py. Update it if your API is not on localhost.

## Configuration
Environment variables are loaded from .env (see .env.example).

Core settings:
- DATABASE_URL: SQLAlchemy database URL (default uses SQLite in training/data/).
- JWT_SECRET, JWT_ALGORITHM, ACCESS_TOKEN_EXPIRE_MINUTES: JWT settings.
- LOG_LEVEL, LOG_FILE: structured logging configuration.
- API_HOST, API_PORT, WORKERS, RELOAD: FastAPI server settings.
- MODEL_DIR, DATA_DIR, PROCESSED_DATA_DIR: training/runtime data locations.
- DRIFT_CHECK_INTERVAL_DAYS, DRIFT_THRESHOLD_KL_DIVERGENCE, DRIFT_THRESHOLD_JS_DIVERGENCE.
- ENABLE_PROMETHEUS, PROMETHEUS_PORT.

Runtime flags:
- USE_INDUCTIVE_MODE=true|false (enables 2-hop subgraph inference path when graph artifacts exist).

## Docker
All-in-one container (API + frontend):
```bash
docker build -t health-fraud-detection .
docker run -p 8000:8000 -p 8501:8501 health-fraud-detection
```

Compose (separate API and frontend images):
```bash
docker compose -f docker/docker-compose.yml up --build
# Optional training profile
# docker compose -f docker/docker-compose.yml --profile train up --build
```

## Tests
Unit-style retraining checks:
```bash
pytest tests/test_retraining.py -v
```

Monitoring and prediction checks (require API running locally):
```bash
python tests/test_monitoring.py
python tests/test_prediction_metrics.py
```

## Related docs
- training/README.md
- DEPLOYMENT_HF_SPACES.md

## License
No license file is included yet.
