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
├── scripts/          # Entry points
│   ├── run_pipeline.py    # Full end-to-end training
│   └── train_gnn.py       # GNN-only training
├── tests/            # Unit & integration tests
└── README.md         # This file
```

Data and trained artifacts are written to the repository root: `data/raw/`, `data/processed/`,
`models/baseline/`, `models/gnn/` and `logs/` (run the scripts from the repository root).

## 🚀 Quick Start

### Generate & Train All Models
From the repository root:
```bash
python run_pipeline.py            # add --skip-gnn to skip the GNN step
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

Logistic Regression, Random Forest and Gradient Boosting (`src/models/baseline.py`) and an HGT model
(`src/models/gnn.py`). This pipeline trains the models served by the demo application on the legacy
2,100-claim dataset, which is separable by construction, so its scores are not meaningful estimates of
real-world performance. For evaluated results see the research pipeline (`research/README.md`) and the
manuscript; earlier performance tables in this README were unsupported and have been removed
(see `paper/research_audit.md`).

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
- Ensure you've run `python run_pipeline.py` from the repository root at least once
- Check that `data/processed/` has `.npy` files

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

**Model served by the demo API:** Random Forest (`models/baseline/random_forest.pkl`).
