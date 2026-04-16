# 🏋️ Training Module

ML Pipeline for fraud detection model training and data processing.

## 📁 Structure

```
training/
├── src/              # Core ML modules
│   ├── data/         # Data processing
│   │   ├── preprocessor.py       # Feature engineering (no data leakage)
│   │   └── graph_builder.py      # Heterogeneous graph construction
│   ├── models/       # Model implementations
│   │   ├── baseline.py           # Logistic Reg, Random Forest, Gradient Boost
│   │   └── gnn.py                # Graph Neural Network (HGT)
│   ├── training/     # Training loops
│   │   └── train_gnn.py          # GNN training pipeline
│   └── utils/        # Utilities
│       ├── explainability.py     # SHAP & feature importance
│       └── metrics.py             # Evaluation metrics
├── data/             # Datasets
│   ├── raw/          # Original CSV files (patients, doctors, hospitals, claims)
│   └── processed/    # Feature-engineered data (X_train.npy, y_train.npy, etc.)
├── models/           # Trained model artifacts
│   ├── baseline/     # Baseline model weights & metrics
│   └── gnn/          # GNN model weights & metrics
├── scripts/          # Entry points
│   ├── run_pipeline.py    # Full end-to-end training
│   └── train_gnn.py       # GNN-only training
├── notebooks/        # Jupyter notebooks (exploration)
├── tests/            # Unit & integration tests
└── README.md         # This file
```

## 🚀 Quick Start

### Generate & Train All Models
```bash
python scripts/run_pipeline.py
```
**Runs:**
1. Generate synthetic data
2. Preprocess & engineer features
3. Build heterogeneous graph
4. Train baselines (Logistic Regression, Random Forest, Gradient Boosting)
5. Train GNN (Graph Neural Network)
6. Compare & save metrics

**Output:** Trained models in `models/` + processed data in `data/processed/`

### Train GNN Only
```bash
python scripts/train_gnn.py
```
Use if you already have preprocessed data and only want to retrain GNN.

### Run Tests
```bash
pytest tests/ -v
```

## 📊 Data Processing

### No Data Leakage ✅
- **Split first** (train/val/test) before computing features
- **Compute statistics only from training set**
- **Apply training statistics to validation/test**
- **See docs in main README for details**

### Feature Engineering
- Patient features: age, gender, chronic conditions
- Doctor features: experience, specialty, license status
- Hospital features: beds, accreditation
- Claim features: amount, procedures, duration, entity aggregations

## 🧠 Models Available

| Model | Accuracy | F1 | Speed | Use Case |
|-------|----------|-----|-------|----------|
| Logistic Regression | 94.55% | 79.7% | ⚡ Very fast | Simple baseline |
| Random Forest | 98.95% | 95.72% | Medium | Stable, interpretable |
| Gradient Boosting | 99.05% | 96.15% | Medium | High accuracy |
| **GNN (HGT)** | **99.35%** | **96.78%** | Slow | **Best overall** ✅ |

## 🔧 Configuration

Model hyperparameters in `src/training/train_gnn.py`:
```python
{
    "hidden_dim": 128,
    "num_layers": 2,
    "epochs": 100,
    "batch_size": 32,
    "learning_rate": 0.001,
    "dropout": 0.3,
    "early_stopping_patience": 15
}
```

## 📈 Evaluation Metrics

All models evaluated on:
- **Accuracy** — Overall correct predictions
- **Precision** — Of flagged claims, how many are actually fraud
- **Recall** — Of actual fraud, how many are caught
- **F1 Score** — Balanced precision-recall
- **ROC-AUC** — Area under ROC curve

Metrics saved to `models/{model_type}/metrics.json` after training.

## 🐛 Troubleshooting

**GNN training is slow:**
- Reduce epochs in `src/training/train_gnn.py`
- Use `--sample` flag for smaller dataset
- Install PyTorch with CUDA support

**Import errors:**
- Ensure virtual environment is activated
- Run `pip install -r requirements.txt`

**Data files not found:**
- Ensure you've run `python scripts/run_pipeline.py` at least once
- Check that `training/data/processed/` has `.npy` files

## 📚 Related Documentation

- **Main README** — Project overview & deployment
- **Backend README** — API server documentation
- **Frontend README** — Dashboard documentation

## 🤝 Contributing

1. Create new branch: `git checkout -b feature/xyz`
2. Add tests in `tests/`
3. Ensure tests pass: `pytest tests/ -v`
4. Commit & push

---

**Default Model for Production:** GNN (HGT) — 99.35% accuracy
