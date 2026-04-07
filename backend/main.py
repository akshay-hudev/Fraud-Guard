"""
FastAPI Backend
Health Insurance Fraud Detection API

Endpoints:
  POST /predict        — single claim fraud prediction
  POST /predict/batch  — batch prediction
  POST /upload         — CSV upload for bulk scoring
  GET  /stats          — model performance statistics
  GET  /alerts         — recent high-risk alerts
  GET  /graph/data     — graph data for frontend visualisation
  GET  /health         — liveness check
"""

import os
import io
import json
import logging
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger(__name__)

# ── Import internal modules (graceful fallback if models not trained yet) ──────
try:
    import sys; sys.path.insert(0, ".")
    from backend.predictor import FraudPredictor
    from src.utils.explainability import AlertSystem
    from src.utils.metrics import MOCK_RESULTS
    _PREDICTOR_AVAILABLE = True
except ImportError as e:
    log.warning("Predictor import failed (%s). Using mock responses.", e)
    _PREDICTOR_AVAILABLE = False

# ── Lifespan ───────────────────────────────────────────────────────────────────
predictor: Optional[object]  = None
alert_system: Optional[object] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor, alert_system
    log.info("Starting up — loading models...")
    if _PREDICTOR_AVAILABLE:
        try:
            predictor    = FraudPredictor()
            alert_system = AlertSystem()
            log.info("Models loaded successfully.")
        except Exception as e:
            log.warning("Model load failed: %s. API will use mock responses.", e)
    yield
    log.info("Shutting down.")


# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "Health Insurance Fraud Detection API",
    description = "GNN-powered fraud detection for healthcare claims.",
    version     = "1.0.0",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ── Schemas ────────────────────────────────────────────────────────────────────

class ClaimRequest(BaseModel):
    claim_id:       str              = Field("C_TEST", description="Unique claim identifier")
    patient_id:     Optional[str]    = None
    doctor_id:      Optional[str]    = None
    hospital_id:    Optional[str]    = None
    claim_amount:   float            = Field(..., ge=0, description="Claim amount in USD")
    num_procedures: int              = Field(1, ge=1)
    days_in_hospital: int            = Field(0, ge=0)
    age:            Optional[int]    = 40
    gender:         Optional[str]    = "M"
    insurance_type: Optional[str]    = "Private"
    specialty:      Optional[str]    = "General Practitioner"
    chronic_condition: Optional[str] = "None"
    explain:        bool             = False

    class Config:
        json_schema_extra = {
            "example": {
                "claim_id":       "C_001",
                "claim_amount":   25000,
                "num_procedures": 12,
                "days_in_hospital": 15,
                "age":            45,
                "gender":         "M",
                "insurance_type": "Medicare",
                "explain":        True,
            }
        }


class PredictionResponse(BaseModel):
    claim_id:          str
    fraud_probability: float
    prediction:        int
    risk_level:        str
    model_used:        str
    alert:             Optional[dict] = None
    contributions:     Optional[list] = None


class BatchRequest(BaseModel):
    claims: list[ClaimRequest]


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["system"])
def health_check():
    return {
        "status":           "healthy",
        "model_loaded":     predictor is not None,
        "predictor_type":   type(predictor).__name__ if predictor else "none",
    }


@app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
def predict_claim(req: ClaimRequest):
    """Predict fraud probability for a single claim."""
    claim_dict = req.model_dump(exclude={"explain", "claim_id"})

    if predictor:
        result = predictor.predict(claim_dict, explain=req.explain)
    else:
        # Mock response for demo
        import random
        amount = req.claim_amount
        prob   = min(amount / 40_000 + random.uniform(-0.05, 0.05), 0.99)
        result = {
            "fraud_probability": round(prob, 4),
            "prediction":        int(prob >= 0.5),
            "risk_level":        "HIGH" if prob > 0.7 else "MEDIUM" if prob > 0.4 else "LOW",
            "model_used":        "mock_heuristic",
        }

    # Fire alert if needed
    alert_data = None
    if alert_system:
        alert_data = alert_system.evaluate(
            req.claim_id, result["fraud_probability"],
            metadata={"amount": req.claim_amount, "patient": req.patient_id},
        )

    return PredictionResponse(
        claim_id          = req.claim_id,
        fraud_probability = result["fraud_probability"],
        prediction        = result["prediction"],
        risk_level        = result["risk_level"],
        model_used        = result["model_used"],
        alert             = alert_data,
        contributions     = result.get("contributions"),
    )


@app.post("/predict/batch", tags=["prediction"])
def predict_batch(req: BatchRequest):
    """Predict fraud for a batch of claims."""
    results = []
    for claim_req in req.claims:
        try:
            results.append(predict_claim(claim_req).model_dump())
        except Exception as e:
            results.append({"claim_id": claim_req.claim_id, "error": str(e)})
    return {"results": results, "total": len(results)}


@app.post("/upload", tags=["data"])
async def upload_claims(file: UploadFile = File(...)):
    """
    Upload a CSV of claims for bulk fraud scoring.
    Returns a scored CSV with fraud_probability column added.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(400, "Only CSV files are accepted.")

    content = await file.read()
    df = pd.read_csv(io.StringIO(content.decode("utf-8")))

    required = ["claim_amount"]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise HTTPException(400, f"Missing required columns: {missing}")

    records = df.to_dict(orient="records")
    scored  = []
    for rec in records:
        if predictor:
            result = predictor.predict(rec)
        else:
            amount = float(rec.get("claim_amount", 0))
            prob   = min(amount / 40_000, 0.99)
            result = {
                "fraud_probability": round(prob, 4),
                "prediction": int(prob >= 0.5),
                "risk_level": "HIGH" if prob > 0.7 else "LOW",
            }
        scored.append({**rec, **result})

    result_df = pd.DataFrame(scored)
    summary = {
        "total_claims":    len(result_df),
        "fraud_flagged":   int(result_df["prediction"].sum()),
        "fraud_rate":      round(result_df["prediction"].mean(), 4),
        "high_risk":       int((result_df["risk_level"] == "HIGH").sum()),
        "total_amount":    round(float(result_df["claim_amount"].sum()), 2),
        "flagged_amount":  round(float(result_df.loc[result_df["prediction"]==1, "claim_amount"].sum()), 2),
        "preview":         scored[:10],
    }
    return summary


@app.get("/stats", tags=["analytics"])
def model_stats():
    """Return model performance metrics for all trained models."""
    # Try to load real results; fall back to mock
    results_path = "models/comparison.json"
    if os.path.exists(results_path):
        with open(results_path) as f:
            results = json.load(f)
    else:
        from src.utils.metrics import MOCK_RESULTS
        results = MOCK_RESULTS

    # Compute summary stats from alerts log
    alert_summary = {}
    if alert_system:
        recent = alert_system.get_recent_alerts(100)
        alert_summary = {
            "total_alerts":  len(recent),
            "high_risk":     sum(1 for a in recent if a.get("risk_level") == "HIGH"),
            "medium_risk":   sum(1 for a in recent if a.get("risk_level") == "MEDIUM"),
        }

    return {
        "model_performance": results,
        "alert_summary":     alert_summary,
        "best_model": max(results, key=lambda k: results[k].get("f1", 0)),
    }


@app.get("/alerts", tags=["analytics"])
def recent_alerts(n: int = Query(20, ge=1, le=100)):
    """Return the most recent high/medium risk alerts."""
    if alert_system:
        return {"alerts": alert_system.get_recent_alerts(n)}
    # Return mock alerts
    mock = [
        {"claim_id": f"C{i:04d}", "fraud_probability": 0.85 - i*0.02,
         "risk_level": "HIGH", "recommended_action": "BLOCK_AND_REVIEW"}
        for i in range(5)
    ]
    return {"alerts": mock}


@app.get("/graph/data", tags=["visualization"])
def graph_data(max_nodes: int = Query(200, ge=10, le=1000)):
    """
    Return a lightweight graph snapshot for frontend visualization.
    Samples up to max_nodes claims + their connected entities.
    """
    claims_path  = "data/raw/claims.csv"
    patients_path = "data/raw/patients.csv"
    doctors_path  = "data/raw/doctors.csv"

    if not os.path.exists(claims_path):
        # Return mock graph
        nodes = [
            {"id": "P001", "type": "patient",  "label": "Patient 1"},
            {"id": "D001", "type": "doctor",   "label": "Dr. Smith"},
            {"id": "H001", "type": "hospital", "label": "Hospital A"},
            {"id": "C001", "type": "claim",    "label": "Claim $5000", "fraud": True},
        ]
        edges = [
            {"source": "P001", "target": "C001", "relation": "filed_claim"},
            {"source": "C001", "target": "D001", "relation": "treated_by"},
            {"source": "D001", "target": "H001", "relation": "works_at"},
        ]
        return {"nodes": nodes, "edges": edges}

    claims   = pd.read_csv(claims_path  ).sample(min(max_nodes, 500), random_state=42)
    patients = pd.read_csv(patients_path)
    doctors  = pd.read_csv(doctors_path )

    nodes, edges = [], []
    seen = set()

    for _, row in claims.iterrows():
        # Claim node
        c_id = row["claim_id"]
        if c_id not in seen:
            nodes.append({
                "id":    c_id,
                "type":  "claim",
                "label": f"${row['claim_amount']:,.0f}",
                "fraud": bool(row.get("fraud_label", 0)),
                "amount": float(row["claim_amount"]),
            })
            seen.add(c_id)

        # Patient node
        p_id = row["patient_id"]
        if p_id not in seen:
            p_row = patients[patients["patient_id"] == p_id]
            nodes.append({
                "id":    p_id,
                "type":  "patient",
                "label": p_id,
                "age":   int(p_row["age"].values[0]) if len(p_row) else 0,
            })
            seen.add(p_id)

        # Doctor node
        d_id = row["doctor_id"]
        if d_id not in seen:
            nodes.append({"id": d_id, "type": "doctor", "label": d_id})
            seen.add(d_id)

        edges.append({"source": p_id, "target": c_id, "relation": "filed_claim"})
        edges.append({"source": c_id, "target": d_id, "relation": "treated_by"})

    return {
        "nodes": nodes[:max_nodes],
        "edges": edges[:max_nodes * 2],
        "stats": {
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "fraud_nodes":  sum(1 for n in nodes if n.get("fraud")),
        }
    }


@app.get("/simulate", tags=["realtime"])
def simulate_realtime(n: int = Query(5, ge=1, le=50)):
    """Simulate real-time incoming claims (for frontend live feed demo)."""
    import random
    import time

    simulated = []
    for i in range(n):
        amount    = random.uniform(200, 45_000)
        is_fraud  = amount > 20_000 and random.random() < 0.6
        prob      = random.uniform(0.65, 0.95) if is_fraud else random.uniform(0.02, 0.35)
        simulated.append({
            "claim_id":          f"SIM_{int(time.time())}_{i}",
            "claim_amount":      round(amount, 2),
            "fraud_probability": round(prob, 4),
            "prediction":        int(prob >= 0.5),
            "risk_level":        "HIGH" if prob > 0.7 else "MEDIUM" if prob > 0.4 else "LOW",
            "timestamp":         pd.Timestamp.utcnow().isoformat(),
        })
    return {"claims": simulated}


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
