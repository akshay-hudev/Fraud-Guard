# 🏥 Health Insurance Fraud Detection System
### Graph Neural Networks + Machine Learning | FastAPI + Streamlit | Production-Ready

[![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109-green?logo=fastapi)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2-orange?logo=pytorch)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.31-red?logo=streamlit)](https://streamlit.io)
[![Tests](https://img.shields.io/badge/Tests-39%20passing-brightgreen)](tests/)
[![Docker](https://img.shields.io/badge/Docker-ready-blue?logo=docker)](docker/)

> Detect fraudulent healthcare insurance claims in real-time using a Heterogeneous Graph Transformer (HGT) that models relationships between **patients → claims → doctors → hospitals**.

---

## 📸 System Preview

| Feature | Screenshot |
|---|---|
| Dashboard | Model metrics, KPI cards, architecture overview |
| Single Claim | Gauge + SHAP feature contributions |
| Graph Explorer | Interactive fraud network (NetworkX + Plotly) |
| Bulk Upload | CSV scoring with summary analytics |
| Live Feed | Real-time claim simulation stream |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     DATA LAYER                                  │
│  patients.csv  doctors.csv  hospitals.csv  claims.csv           │
└──────────────────────┬──────────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────────┐
│                  PROCESSING LAYER                               │
│  FraudDataPreprocessor → Feature Engineering → GraphBuilder     │
│  (36 features, stratified splits, HeteroData graph)            │
└──────────┬───────────────────────────────┬──────────────────────┘
           │                               │
┌──────────▼──────────┐         ┌──────────▼──────────────────────┐
│   BASELINE ML       │         │       GNN PIPELINE              │
│  Logistic Regression│         │  HGTConv × 2 layers             │
│  Random Forest      │         │  Patient / Doctor /             │
│  Gradient Boosting  │         │  Hospital / Claim nodes         │
└──────────┬──────────┘         └──────────┬──────────────────────┘
           └─────────────┬─────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│                     FastAPI BACKEND                             │
│  /predict  /predict/batch  /upload  /stats  /alerts            │
│  /graph/data  /simulate  /health                               │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│                  Streamlit FRONTEND                             │
│  Dashboard · Single Claim · Bulk Upload · Graph Explorer       │
│  Model Analytics · Live Feed                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📁 Folder Structure

```
health-fraud-detection/
├── data/
│   ├── raw/                        # Generated CSVs (hospitals, doctors, patients, claims)
│   ├── processed/                  # Scaled features, graph, scaler/encoder pickles
│   └── generate_synthetic.py       # Synthetic dataset generator
│
├── src/
│   ├── data/
│   │   ├── preprocessor.py         # Clean → engineer → encode → split
│   │   └── graph_builder.py        # Tabular → PyG HeteroData graph
│   ├── models/
│   │   ├── baseline.py             # LR + RF + GBM training & evaluation
│   │   └── gnn.py                  # HGTFraudDetector (PyTorch Geometric)
│   ├── training/
│   │   └── train_gnn.py            # GNN training loop, early stopping
│   └── utils/
│       ├── explainability.py       # SHAP + AlertSystem
│       └── metrics.py              # Metrics helpers + comparison
│
├── backend/
│   ├── main.py                     # FastAPI app (8 endpoints)
│   └── predictor.py                # Singleton inference class
│
├── frontend/
│   └── app.py                      # Streamlit dashboard (6 pages)
│
├── scripts/
│   ├── run_pipeline.py             # Master pipeline runner
│   └── train_gnn.py                # GNN training entry point
│
├── tests/
│   └── test_all.py                 # 39 tests (data + models + API)
│
├── notebooks/
│   └── exploration.ipynb           # EDA + visualization notebook
│
├── docs/
│   └── DEPLOYMENT.md               # Render / AWS / Docker deployment
│
├── docker/
│   ├── Dockerfile.api              # Backend container
│   ├── Dockerfile.frontend         # Frontend container
│   └── docker-compose.yml          # Full stack orchestration
│
└── requirements.txt
```

---

## ⚡ Quick Start (Local)

### 1. Clone & Install

```bash
git clone https://github.com/YOUR/health-fraud-detection.git
cd health-fraud-detection

# Create virtual environment
python -m venv venv && source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install PyTorch (CPU)
pip install torch==2.2.0 --index-url https://download.pytorch.org/whl/cpu

# Install PyTorch Geometric (for GNN)
pip install torch-geometric
pip install torch-scatter torch-sparse \
  --find-links https://data.pyg.org/whl/torch-2.2.0+cpu.html
```

### 2. Run the Full ML Pipeline

```bash
# Generate data + preprocess + train all baseline models
python scripts/run_pipeline.py

# Include GNN training (requires PyTorch Geometric)
python scripts/run_pipeline.py  # GNN runs automatically if PyG is installed

# Skip GNN (faster, CPU-only machines)
python scripts/run_pipeline.py --skip-gnn

# Just train GNN (after baseline pipeline has run)
python scripts/train_gnn.py
```

### 3. Start the API

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

# Verify
curl http://localhost:8000/health
# → {"status":"healthy","model_loaded":true}
```

### 4. Start the Dashboard

```bash
streamlit run frontend/app.py
# → Opens at http://localhost:8501
```

### 5. Run Tests

```bash
pytest tests/test_all.py -v
# 39 tests: data / preprocessing / models / predictor / alerts / API
```

---

## 🐳 Docker Deployment

```bash
# Train models first (one-time)
docker compose -f docker/docker-compose.yml --profile train up pipeline

# Start full stack (API + Frontend)
docker compose -f docker/docker-compose.yml up -d

# Verify
curl http://localhost:8000/health       # API
open http://localhost:8501              # Dashboard
```

---

## 📡 API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/health` | Liveness check |
| `POST` | `/predict` | Single claim fraud prediction |
| `POST` | `/predict/batch` | Batch claim prediction |
| `POST` | `/upload` | CSV bulk scoring |
| `GET`  | `/stats` | Model performance metrics |
| `GET`  | `/alerts` | Recent high-risk alerts |
| `GET`  | `/graph/data` | Graph data for visualization |
| `GET`  | `/simulate` | Real-time claim simulation |

### Example: Predict a Claim

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "claim_id":       "C_001",
    "claim_amount":   35000,
    "num_procedures": 18,
    "days_in_hospital": 12,
    "age":            67,
    "insurance_type": "Medicare",
    "explain":        true
  }'
```

```json
{
  "claim_id": "C_001",
  "fraud_probability": 0.8731,
  "prediction": 1,
  "risk_level": "HIGH",
  "model_used": "random_forest",
  "alert": {
    "recommended_action": "BLOCK_AND_REVIEW"
  },
  "contributions": [
    {"feature": "claim_amount",      "shap_value": 0.3241},
    {"feature": "num_procedures",    "shap_value": 0.2812},
    {"feature": "days_in_hospital",  "shap_value": 0.1934}
  ]
}
```

---

## 📊 Model Performance

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|----|---------|
| Logistic Regression | 99.87% | 98.92% | 100.0% | 99.46% | 1.000 |
| Random Forest | 99.93% | 100.0% | 99.45% | 99.73% | 1.000 |
| Gradient Boosting | 99.93% | 100.0% | 99.45% | 99.73% | 0.997 |
| **GNN (HGT)** | **95.7%** | **91.0%** | **88.4%** | **89.7%** | **0.980** |

> **Note:** Near-perfect baseline scores are expected on synthetic data with strong fraud signals. On real-world data (e.g. Kaggle Healthcare Provider Fraud), expect F1 ≈ 0.75–0.90. The GNN scores reflect realistic cross-entity patterns.

---

## 🧠 GNN Architecture

```
Input Node Features:
  patient  (4)  → Linear → 128
  doctor   (3)  → Linear → 128
  hospital (2)  → Linear → 128
  claim    (9)  → Linear → 128
       ↓
  HGTConv Layer 1  (heads=4)   ← multi-relational attention
  LayerNorm + Residual + Dropout
       ↓
  HGTConv Layer 2  (heads=4)
  LayerNorm + Residual + Dropout
       ↓
  claim node embeddings [N_claims, 128]
       ↓
  MLP Classifier: 128 → 64 → ReLU → Dropout → 2
       ↓
  Softmax → Fraud Probability
```

**Edge types modelled:**
- `patient → filed_claim → claim`
- `claim → treated_by → doctor`
- `doctor → works_at → hospital`
- `patient → visited → doctor`

---

## 🔬 Explainability

- **SHAP TreeExplainer** on Random Forest → per-claim feature contributions
- **Alert System** → auto-flags HIGH (>75%) and MEDIUM (>45%) risk claims
- **Risk levels**: `LOW` → approve | `MEDIUM` → manual review | `HIGH` → block

---

## 🎯 Resume Highlights

- ✅ End-to-end ML system: data generation → preprocessing → training → API → UI
- ✅ Heterogeneous GNN (HGT) on multi-entity healthcare graph
- ✅ Production FastAPI with 8 endpoints, input validation, error handling
- ✅ Interactive Streamlit dashboard with 6 pages including real-time feed
- ✅ SHAP explainability + automated alert system
- ✅ 39 automated tests (unit + integration)
- ✅ Docker Compose deployment ready
- ✅ Cloud deployment guide (Render / AWS ECS + ECR)

---

## 📚 References

- [Heterogeneous Graph Transformer (HGT)](https://arxiv.org/abs/2003.01332) — Hu et al., 2020
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io)
- [Kaggle Healthcare Fraud Dataset](https://www.kaggle.com/datasets/rohitrox/healthcare-provider-fraud-detection-analysis)
- [SHAP](https://shap.readthedocs.io)

---

## 📄 License

MIT License — free to use for learning and portfolio projects.
