# 📜 Scripts — Training & Deployment Entry Points

Quick reference for running different parts of the ML pipeline.

## Core Pipeline

### Full End-to-End Training
```bash
python scripts/run_pipeline.py
```
**Runs the complete ML pipeline:**
1. Generate synthetic data
2. Preprocess & feature engineering
3. Build heterogeneous graph
4. Train baseline models (Logistic Regression, Random Forest, Gradient Boosting)
5. Train Graph Neural Network (HGT)
6. Compare models & save results

**Output:** Trained models + metrics in `models/` and `data/processed/`

---

### GNN Training Only
```bash
python scripts/train_gnn.py
```
**Runs just the Graph Neural Network training** (assumes data is preprocessed).

Use this if you:
- Already have preprocessed data and graph
- Want to experiment with GNN hyperparameters
- Need to retrain GNN only

**Output:** `models/gnn/hgt_model.pth` + metrics

---

## Deployment

### Start Backend API
```bash
cd backend && python main.py
```
Starts FastAPI server at `http://localhost:8000`

**Endpoints:**
- `POST /predict` — Single prediction
- `POST /predict/batch` — Batch predictions
- `GET /health`  — Health check
- `GET /models` — Available models

---

### Start Frontend Dashboard
```bash
cd frontend && streamlit run app.py
```
Opens Streamlit dashboard at `http://localhost:8501`

---

### Docker Deployment
```bash
docker-compose -f docker/docker-compose.yml up
```
Runs both API and frontend in containers.

---

## Development & Testing

### Run Tests
```bash
pytest tests/test_all.py -v
```

---

## File Structure

```
scripts/
├── run_pipeline.py     ← Master pipeline (all models)
├── train_gnn.py        ← GNN-only training wrapper
└── README.md           ← This file
```

---

## Tips

**Reduce training time for testing:**
- Edit `run_pipeline.py` to train on smaller sample
- Reduce epochs in `src/training/train_gnn.py`
- Use `--sample` flag (if supported)

**Monitor training:**
- Check logs in `logs/` folder
- Use `tensorboard --logdir logs/` for TensorBoard visualization
- Watch `models/` folder for saved artifacts

