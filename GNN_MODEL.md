# 🧠 GNN HGT Model — Best Performer

## Model Overview

**Heterogeneous Graph Transformer (HGT)** — Advanced fraud detection using Graph Neural Networks

### Architecture
```
Input Graph (Heterogeneous):
├── Patient Nodes (features: age, gender, chronic condition, insurance type)
├── Doctor Nodes (features: experience, specialty, license status)
├── Hospital Nodes (features: beds, accreditation status)
└── Claim Nodes (features: amount, procedures, days in hospital, aggregated stats)

Edge Types:
├── Patient → Claim (filed_claim) — which patient filed which claim
├── Claim → Doctor (treated_by) — which doctor treated the claim
├── Doctor → Hospital (works_at) — which hospital employs doctor
└── Patient → Doctor (visited) — which patients visited which doctors
```

### 3-Layer Architecture

**Layer 1: Input Projection**
- Each node type projected to 128-dim hidden space
- Patient: 4 features → 128 dims
- Doctor: 3 features → 128 dims
- Hospital: 2 features → 128 dims
- Claim: 9 features → 128 dims

**Layer 2: Graph Attention (2 HGT Conv layers)**
- Multi-head attention (4 heads) over heterogeneous edges
- Message passing learns relationships between entities
- Batch normalization after each layer
- Residual connections for gradient flow

**Layer 3: Classification MLP**
- 128 → 64 (ReLU) → 2 (Fraud/Legit)
- Dropout (0.3) for regularization
- Softmax output → fraud probability

### Why Graphs for Fraud Detection?

Traditional ML sees isolated claims. **GNN sees relationships:**

```
Example 1: Ring Fraud
┌─────────┐
│PatientA │────→ ┌──────────┐    ┌─────────┐
└─────────┘      │ Claim 1  │→ Doctor  X  │
                 └──────────┘    └─────────┘
                      ↓              ↓
┌─────────┐       ┌──────────┐      │
│PatientB │─────→ │ Claim 2  │→     │
└─────────┘       └──────────┘      │
                       ↓              ↓
        ┌──────────────────────────────
        │
        └─→ Hospital Y (all from same hospital)

GNN detects: Same doctor treating multiple patients with suspicious patterns + same hospital
RF detects: Individual claim is suspicious (but misses the pattern)
```

### Performance

**Test Set Results (2000 claims):**

| Metric | Value |
|--------|-------|
| **Accuracy** | 99.35% |
| **Precision** | 99.10% |
| **Recall** | 94.53% (catches fraud) |
| **F1 Score** | 96.78% |
| **ROC-AUC** | 0.9884 |

### Comparison vs Baselines

| Model | Accuracy | F1 | ROC-AUC | Why GNN Wins |
|-------|----------|-----|---------|---|
| Logistic Regression | 94.55% | 79.7% | 0.954 | Can't handle non-linear patterns |
| Random Forest | 98.95% | 95.72% | 0.981 | Sees trees, not patterns |
| Gradient Boosting | 99.05% | 96.15% | 0.9815 | Good, but misses graph structure |
| **GNN HGT** | **99.35%** | **96.78%** | **0.9884** | ✅ Captures entitity relationships |

### Key Advantages

✅ **Relationship-aware**: Detects connected fraud patterns (rings, collusion)  
✅ **Heterogeneous**: Handles different entity types with different features  
✅ **Explainable**: Follow the suspicious paths in the graph  
✅ **Scalable**: Graph structure stays same; just update node features  
✅ **Transferable**: Trained patterns apply to new doctors/patients/hospitals  

### Implementation

**Framework**: PyTorch Geometric (torch-geometric)  
**Graph Type**: Heterogeneous (multiple node & edge types)  
**Convolution**: HGTConv (Heterogeneous Graph Transformer)  
**Loss**: Weighted Cross-Entropy (handles class imbalance: 12% fraud)  
**Optimizer**: AdamW with CosineAnnealing schedule  
**Training**: 100 epochs with early stopping (patience=15)  

### Files

- [src/models/gnn.py](../src/models/gnn.py) — HGT model architecture
- [src/data/graph_builder.py](../src/data/graph_builder.py) — Graph construction
- [src/training/train_gnn.py](../src/training/train_gnn.py) — Training pipeline
- [models/gnn/](../models/gnn/) — Saved weights & metrics

### Production Deployment

The GNN model is integrated into the FastAPI backend:
```python
predictor = ProductionFraudPredictor(use_gnn=True)
prediction = predictor.predict(claim_features, graph_data)
```

- Caching layer for graph embeddings
- Batch inference support
- Fallback to Random Forest if GNN unavailable
- Drift detection on learned representations

### Future Improvements

🔄 **Temporal Dynamics**: Track graphs over time (new edges appear)  
📊 **Interpretability**: SHAP for GNN, graph attention visualization  
⚡ **Optimization**: Sampling for larger graphs, knowledge distillation  
🔐 **Adversarial**: Test against sophisticated fraud patterns  

---

**Best Model for Production**: GNN HGT  
**Accuracy**: 99.35% on test set  
**Inference Latency**: ~120ms per claim (with graph context)  
**Throughput**: 2K-3K predictions/sec
