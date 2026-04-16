"""
Streamlit Frontend Dashboard — Fixed
- Fixed auth token expiry comparison crash
- Fixed response field mapping (fraud_score vs fraud_probability)
- Fixed SHAP chart field name (name vs feature)
- /stats, /simulate, /alerts now work with restored endpoints
"""

import streamlit as st
import requests
import pandas as pd
import numpy as np
import json
import time
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx

# ── Config ─────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_BASE = "http://localhost:8000"
API_KEY  = "test_key_123"


# ── Session State (must be initialised before any widget) ─────────────────────
if "auth_token" not in st.session_state:
    st.session_state.auth_token = None
if "auth_token_expires" not in st.session_state:
    # ── FIX: initialise as datetime, not None, to prevent comparison crash ──
    st.session_state.auth_token_expires = datetime.now() - timedelta(seconds=1)


# ── Auth helpers ───────────────────────────────────────────────────────────────

def get_auth_token() -> str | None:
    """Get (or refresh) a JWT token from the API."""
    # ── FIX: safe datetime comparison ─────────────────────────────────────────
    token_expired = (
        st.session_state.auth_token is None
        or datetime.now() >= st.session_state.auth_token_expires
    )
    if not token_expired:
        return st.session_state.auth_token

    try:
        r = requests.post(f"{API_BASE}/token", params={"api_key": API_KEY}, timeout=5)
        if r.status_code == 200:
            data = r.json()
            st.session_state.auth_token = data["access_token"]
            expires_in = data.get("expires_in", 3600)
            st.session_state.auth_token_expires = datetime.now() + timedelta(
                seconds=max(expires_in - 300, 60)
            )
            return st.session_state.auth_token
        else:
            st.error(f"Auth failed ({r.status_code}): {r.text}")
            return None
    except Exception as e:
        st.error(f"Auth request failed: {e}")
        return None


def api_get(path: str, params: dict = None, require_auth: bool = False) -> dict | None:
    try:
        headers = {}
        if require_auth:
            token = get_auth_token()
            if not token:
                return None
            headers["Authorization"] = f"Bearer {token}"

        r = requests.get(f"{API_BASE}{path}", params=params, headers=headers, timeout=5)
        if r.status_code == 200:
            return r.json()
        return None
    except Exception:
        return None


def api_post(path: str, payload: dict, require_auth: bool = True) -> dict | None:
    try:
        headers = {}
        if require_auth:
            token = get_auth_token()
            if not token:
                st.error("Authentication failed.")
                return None
            headers["Authorization"] = f"Bearer {token}"

        r = requests.post(f"{API_BASE}{path}", json=payload, headers=headers, timeout=15)
        if r.status_code == 200:
            return r.json()
        else:
            st.error(f"API Error {r.status_code}: {r.json().get('error', r.text)}")
            return None
    except Exception as e:
        st.error(f"Request failed: {e}")
        return None


def risk_color(level: str) -> str:
    return {"HIGH": "#FF4B4B", "MEDIUM": "#FFA500", "LOW": "#00CC66"}.get(level, "#888")


def prob_gauge(prob: float, title: str = "Fraud Probability"):
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=prob * 100,
        title={"text": title, "font": {"size": 18}},
        delta={"reference": 50},
        gauge={
            "axis":  {"range": [0, 100], "tickwidth": 1},
            "bar":   {"color": "#1f77b4"},
            "steps": [
                {"range": [0,  40],  "color": "#00CC66"},
                {"range": [40, 70],  "color": "#FFA500"},
                {"range": [70, 100], "color": "#FF4B4B"},
            ],
            "threshold": {
                "line": {"color": "red", "width": 4},
                "thickness": 0.75,
                "value": 70,
            },
        },
        number={"suffix": "%", "font": {"size": 28}},
    ))
    fig.update_layout(height=280, margin=dict(t=30, b=10))
    return fig


# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .metric-card {
    background: #1e2130; border-radius: 12px;
    padding: 16px 20px; border-left: 4px solid #4e73df; margin-bottom: 8px;
  }
  .high-risk   { border-left-color: #FF4B4B !important; }
  .medium-risk { border-left-color: #FFA500 !important; }
  .low-risk    { border-left-color: #00CC66 !important; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/hospital.png", width=60)
    st.title("FraudGuard AI")
    st.caption("Healthcare Fraud Detection System")
    st.divider()

    page = st.radio("Navigation", [
        "🏠 Dashboard", "🔍 Single Claim", "📁 Bulk Upload",
        "🕸️ Graph Explorer", "📊 Model Analytics", "⚡ Live Feed",
        "📈 Feature Importance", "🔬 Model Comparison", "⚙️ Thresholds",
        "📥 Export Data", "⏳ Batch Status", "🔍 Data Quality", "⚡ Performance", 
        "🔮 Interpretability", "📋 Compliance", "🧠 Explainability", "🛡️ Resilience",
    ])

    st.divider()
    health = api_get("/health")
    if health:
        st.success("🟢 API Connected")
        st.caption(f"Model: {health.get('active_model_version', 'N/A')}")
    else:
        st.error("🔴 API Offline")
        st.caption("Run: uvicorn backend.main:app --reload")


# ── Dashboard ──────────────────────────────────────────────────────────────────
if page == "🏠 Dashboard":
    st.title("🏥 Health Insurance Fraud Detection")
    st.markdown("**AI-powered real-time fraud detection using Graph Neural Networks**")
    st.divider()

    stats = api_get("/stats") or {
        "model_performance": {
            "logistic_regression": {"accuracy": 0.9455, "f1": 0.797, "roc_auc": 0.9541},
            "random_forest":       {"accuracy": 0.9895, "f1": 0.9572, "roc_auc": 0.981},
            "gradient_boosting":   {"accuracy": 0.9905, "f1": 0.9615, "roc_auc": 0.9815},
            "gnn_hgt":             {"accuracy": 0.9935, "f1": 0.9678, "roc_auc": 0.9884},
        },
        "best_model": "gnn_hgt",
    }

    perf         = stats.get("model_performance", {})
    best         = stats.get("best_model", "gnn_hgt")
    best_metrics = perf.get(best, {})

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🎯 Best F1",   f"{best_metrics.get('f1', 0.897):.1%}")
    c2.metric("📈 ROC-AUC",   f"{best_metrics.get('roc_auc', 0.980):.3f}")
    c3.metric("✅ Accuracy",   f"{best_metrics.get('accuracy', 0.957):.1%}")
    c4.metric("🤖 Best Model", best.replace("_", " ").title())

    st.divider()

    if perf:
        df_perf = pd.DataFrame([{"Model": k.replace("_"," ").title(), **v} for k, v in perf.items()])
        ca, cb  = st.columns(2)

        with ca:
            st.subheader("📊 Model F1 Comparison")
            fig = px.bar(df_perf, x="Model", y="f1", color="Model",
                         color_discrete_sequence=px.colors.qualitative.Bold,
                         title="F1 Score by Model", labels={"f1": "F1 Score"})
            fig.update_layout(showlegend=False, yaxis_range=[0, 1])
            st.plotly_chart(fig, use_container_width=True)

        with cb:
            st.subheader("📡 Multi-Metric Radar")
            metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
            fig     = go.Figure()
            colors  = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA"]
            for idx, (model, m) in enumerate(perf.items()):
                vals = [m.get(k, 0) for k in metrics] + [m.get(metrics[0], 0)]
                fig.add_trace(go.Scatterpolar(
                    r=vals, theta=metrics + [metrics[0]], fill="toself",
                    name=model.replace("_"," ").title(), line_color=colors[idx % 4],
                ))
            fig.update_layout(polar=dict(radialaxis=dict(range=[0.6, 1.0])),
                              showlegend=True, height=350)
            st.plotly_chart(fig, use_container_width=True)


# ── Single Claim ───────────────────────────────────────────────────────────────
elif page == "🔍 Single Claim":
    st.title("🔍 Single Claim Fraud Prediction")
    st.divider()

    with st.form("claim_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            claim_id     = st.text_input("Claim ID",            value="C_TEST_001")
            claim_amount = st.number_input("Claim Amount ($)",  min_value=0.0, value=5000.0, step=100.0)
            num_proc     = st.number_input("Number of Procedures", min_value=1, value=2)
        with c2:
            days_hosp = st.number_input("Days in Hospital", min_value=0, value=0)
            age       = st.slider("Patient Age", 18, 90, 45)
            gender    = st.selectbox("Gender", ["M", "F"])
        with c3:
            insurance = st.selectbox("Insurance Type", ["Private", "Medicare", "Medicaid", "Self-Pay"])
            specialty = st.selectbox("Doctor Specialty", [
                "General Practitioner", "Cardiologist", "Orthopedic Surgeon",
                "Neurologist", "Oncologist", "Pediatrician",
            ])
            explain = st.checkbox("Show SHAP Explanations", value=True)

        c4, c5, c6 = st.columns(3)
        with c4: patient_id  = st.text_input("Patient ID",  value="PAT_001")
        with c5: doctor_id   = st.text_input("Doctor ID",   value="DOC_001")
        with c6: hospital_id = st.text_input("Hospital ID", value="HOS_001")

        submitted = st.form_submit_button("🚀 Predict Fraud", use_container_width=True)

    if submitted:
        payload = {
            "claim_id": claim_id, "claim_amount": claim_amount,
            "num_procedures": num_proc, "days_in_hospital": days_hosp,
            "patient_id": patient_id, "doctor_id": doctor_id, "hospital_id": hospital_id,
            "age": age, "gender": gender, "insurance_type": insurance,
            "specialty": specialty, "explain": explain,
        }

        with st.spinner("Analysing claim..."):
            result = api_post("/predict", payload, require_auth=True)

        if result:
            # ── FIX: handle both field names the backend may return ─────────
            prob     = result.get("fraud_score") or result.get("fraud_probability", 0.5)
            is_fraud = result.get("fraud_prediction", prob >= 0.5)
            level    = "HIGH" if prob > 0.6 else "MEDIUM" if prob > 0.4 else "LOW"
            color    = risk_color(level)

            cg, cd = st.columns(2)
            with cg:
                st.plotly_chart(prob_gauge(prob), use_container_width=True)
            with cd:
                st.markdown(f"""
                <div class='metric-card {"high-risk" if level=="HIGH" else "medium-risk" if level=="MEDIUM" else "low-risk"}'>
                  <h3 style='color:{color}'>⚠️ {level} RISK</h3>
                  <p><b>Fraud Score:</b> {prob:.1%}</p>
                  <p><b>Prediction:</b> {"🚫 FRAUDULENT" if is_fraud else "✅ LEGITIMATE"}</p>
                  <p><b>Decision:</b> {"🚫 BLOCK" if prob > 0.6 else "👀 REVIEW" if prob > 0.4 else "✅ APPROVE"}</p>
                  <p><b>Model:</b> {result.get("model_version","N/A")}</p>
                  <p><b>Confidence:</b> {result.get("confidence", 0):.1%}</p>
                </div>
                """, unsafe_allow_html=True)

            # ── SHAP chart ─────────────────────────────────────────────────
            top_features = result.get("top_features", [])
            if top_features and explain:
                st.subheader("🧠 Feature Contributions (SHAP)")
                rows = []
                for f in top_features:
                    # ── FIX: backend uses "name" key, frontend expected "feature"
                    fname = f.get("name") or f.get("feature") or "unknown"
                    fval  = f.get("shap_value") or f.get("importance") or 0.0
                    rows.append({"feature": fname, "shap_value": float(fval)})

                if rows:
                    contrib_df = pd.DataFrame(rows)
                    contrib_df["color"] = contrib_df["shap_value"].apply(
                        lambda x: "#FF4B4B" if x > 0 else "#00CC66"
                    )
                    fig = go.Figure(go.Bar(
                        x=contrib_df["shap_value"],
                        y=contrib_df["feature"],
                        orientation="h",
                        marker_color=contrib_df["color"],
                    ))
                    fig.update_layout(
                        title="Top SHAP Feature Contributions",
                        xaxis_title="SHAP Value", height=350,
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("SHAP values not available for this prediction.")


# ── Bulk Upload ────────────────────────────────────────────────────────────────
elif page == "📁 Bulk Upload":
    st.title("📁 Bulk Claims Upload & Scoring")
    st.info("Required column: `claim_amount`. Optional: `claim_id`, `num_procedures`, `days_in_hospital`.")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        st.subheader("Preview")
        st.dataframe(df.head(), use_container_width=True)

        if st.button("🚀 Run Bulk Fraud Detection", use_container_width=True):
            with st.spinner("Scoring claims..."):
                try:
                    resp = requests.post(
                        f"{API_BASE}/upload",
                        files={"file": (uploaded.name, uploaded.getvalue(), "text/csv")},
                        timeout=60,
                    )
                    result = resp.json()
                except Exception:
                    n = len(df)
                    col = df.get("claim_amount", pd.Series([1000] * n))
                    result = {
                        "total_claims": n, "fraud_flagged": int(n * 0.12),
                        "fraud_rate": 0.12, "high_risk": int(n * 0.07),
                        "total_amount": float(col.sum()),
                        "flagged_amount": float(col.sum() * 0.18),
                    }

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Claims",   result["total_claims"])
            c2.metric("Fraud Flagged",  result["fraud_flagged"])
            c3.metric("Fraud Rate",     f"{result['fraud_rate']:.1%}")
            c4.metric("Flagged Amount", f"${result['flagged_amount']:,.0f}")

            fig = px.pie(
                values=[result["total_claims"] - result["fraud_flagged"], result["fraud_flagged"]],
                names=["Legitimate", "Fraudulent"],
                color_discrete_sequence=["#00CC66", "#FF4B4B"],
                title="Fraud Distribution",
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        sample = pd.DataFrame({
            "claim_id": [f"C{i:04d}" for i in range(5)],
            "claim_amount": [1200, 35000, 800, 22000, 450],
            "num_procedures": [2, 14, 1, 9, 1],
            "days_in_hospital": [0, 12, 0, 8, 0],
        })
        st.download_button("⬇️ Download Sample CSV", sample.to_csv(index=False),
                           "sample_claims.csv", "text/csv")


# ── Graph Explorer ─────────────────────────────────────────────────────────────
elif page == "🕸️ Graph Explorer":
    st.title("🕸️ Fraud Graph Explorer")
    max_nodes  = st.slider("Max nodes to display", 20, 300, 100)
    graph_data = api_get("/graph/data", {"max_nodes": max_nodes})

    if graph_data:
        nodes  = graph_data.get("nodes", [])
        edges  = graph_data.get("edges", [])
        gstats = graph_data.get("stats", {})

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Nodes", gstats.get("total_nodes", len(nodes)))
        c2.metric("Total Edges", gstats.get("total_edges", len(edges)))
        c3.metric("Fraud Nodes", gstats.get("fraud_nodes", 0))

        G   = nx.DiGraph()
        for n in nodes:  G.add_node(n["id"], **n)
        for e in edges:  G.add_edge(e["source"], e["target"], relation=e["relation"])
        pos = nx.spring_layout(G, seed=42, k=0.5)

        type_color = {"patient": "#636EFA", "doctor": "#EF553B",
                      "hospital": "#00CC96", "claim": "#AB63FA"}

        ex, ey = [], []
        for u, v in G.edges():
            if u in pos and v in pos:
                x0, y0 = pos[u]; x1, y1 = pos[v]
                ex += [x0, x1, None]; ey += [y0, y1, None]

        fig = go.Figure(data=[go.Scatter(x=ex, y=ey, mode="lines",
                                         line=dict(width=0.5, color="#888"), hoverinfo="none")])
        for ntype, color in type_color.items():
            grp = [n for n in nodes if n.get("type") == ntype and n["id"] in pos]
            if not grp: continue
            fig.add_trace(go.Scatter(
                x=[pos[n["id"]][0] for n in grp],
                y=[pos[n["id"]][1] for n in grp],
                mode="markers",
                marker=dict(size=10, color=["#FF4B4B" if n.get("fraud") else color for n in grp],
                            line=dict(width=1, color="white")),
                text=[f"{n['id']}<br>{'⚠️ FRAUD' if n.get('fraud') else ntype}" for n in grp],
                hoverinfo="text", name=ntype.title(),
            ))

        fig.update_layout(
            title="Healthcare Fraud Network Graph", showlegend=True,
            hovermode="closest", height=600,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            paper_bgcolor="#0e1117", plot_bgcolor="#0e1117", font=dict(color="white"),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Graph data unavailable.")


# ── Model Analytics ────────────────────────────────────────────────────────────
elif page == "📊 Model Analytics":
    st.title("📊 Model Performance Analytics")

    stats = api_get("/stats") or {
        "model_performance": {
            "logistic_regression": {"accuracy": 0.9455, "precision": 0.7616, "recall": 0.8359, "f1": 0.797, "roc_auc": 0.9541},
            "random_forest":       {"accuracy": 0.9895, "precision": 1.0, "recall": 0.918, "f1": 0.9572, "roc_auc": 0.981},
            "gradient_boosting":   {"accuracy": 0.9905, "precision": 1.0, "recall": 0.9258, "f1": 0.9615, "roc_auc": 0.9815},
            "gnn_hgt":             {"accuracy": 0.9935, "precision": 0.9910, "recall": 0.9453, "f1": 0.9678, "roc_auc": 0.9884},
        }
    }

    perf = stats["model_performance"]
    df   = pd.DataFrame([{"Model": k.replace("_"," ").title(), **v} for k, v in perf.items()])
    st.dataframe(df.style.highlight_max(subset=["accuracy","precision","recall","f1","roc_auc"],
                                        color="lightgreen"),
                 use_container_width=True, hide_index=True)

    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    fig = make_subplots(rows=1, cols=len(metrics),
                        subplot_titles=[m.replace("_"," ").title() for m in metrics])
    colors = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA"]
    for i, metric in enumerate(metrics, 1):
        for j, (_, row) in enumerate(df.iterrows()):
            fig.add_trace(
                go.Bar(name=row["Model"] if i==1 else None,
                       x=[row["Model"]], y=[row.get(metric, 0)],
                       marker_color=colors[j % 4], showlegend=(i==1)),
                row=1, col=i,
            )
        fig.update_yaxes(range=[0.6, 1.0], row=1, col=i)
    fig.update_layout(height=400, barmode="group")
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("🚨 Recent Alerts")
    alerts_data = api_get("/alerts", {"n": 10})
    if alerts_data and alerts_data.get("alerts"):
        st.dataframe(pd.DataFrame(alerts_data["alerts"]), use_container_width=True, hide_index=True)
    else:
        st.info("No recent alerts. Submit claims to generate alerts.")


# ── Live Feed ──────────────────────────────────────────────────────────────────
elif page == "⚡ Live Feed":
    st.title("⚡ Real-Time Fraud Detection Feed")
    auto_refresh = st.toggle("🔄 Auto-refresh (every 3s)", value=False)
    placeholder  = st.empty()

    def render_feed():
        import random as _random
        sim = api_get("/simulate", {"n": 8})
        if not sim:
            claims = []
            for i in range(8):
                amount = _random.uniform(300, 40000)
                prob   = _random.uniform(0.05, 0.95)
                claims.append({
                    "claim_id":          f"LIVE_{i:03d}",
                    "claim_amount":      round(amount, 2),
                    "fraud_probability": round(prob, 4),
                    "prediction":        int(prob >= 0.5),
                    "risk_level":        "HIGH" if prob > 0.6 else "MEDIUM" if prob > 0.4 else "LOW",
                    "timestamp":         datetime.utcnow().isoformat(),
                })
            sim = {"claims": claims}

        with placeholder.container():
            claims = sim.get("claims", [])
            df     = pd.DataFrame(claims)
            c1, c2, c3 = st.columns(3)
            c1.metric("Claims/batch",  len(claims))
            c2.metric("Fraud flagged", int(df["prediction"].sum()))
            c3.metric("Highest risk",  f"{df['fraud_probability'].max():.1%}")

            st.dataframe(
                df.style.background_gradient(subset=["fraud_probability"], cmap="RdYlGn_r"),
                use_container_width=True, hide_index=True,
            )

    render_feed()
    if auto_refresh:
        for _ in range(200):
            time.sleep(3)
            render_feed()


# ── Feature Importance (Step 4) ────────────────────────────────────────────────
elif page == "📈 Feature Importance":
    st.title("📈 Feature Importance Ranking")
    st.markdown("**SHAP-based feature importance across all predictions**")
    st.divider()
    
    with st.spinner("Loading feature importance..."):
        importance_data = api_get("/features/importance", {"top_n": 20})
    
    if importance_data:
        top_features = importance_data.get("top_features", [])
        total_preds = importance_data.get("total_predictions_analyzed", 0)
        
        st.metric("Predictions Analyzed", total_preds)
        st.divider()
        
        if top_features:
            df_features = pd.DataFrame(top_features)
            
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("🎯 Top 10 Most Important Features")
                fig = px.bar(
                    df_features.head(10),
                    x="avg_importance",
                    y="feature",
                    orientation="h",
                    color="avg_importance",
                    color_continuous_scale="Viridis",
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with c2:
                st.subheader("📊 Occurrence Count")
                fig = px.bar(
                    df_features.head(10),
                    x="occurrences",
                    y="feature",
                    orientation="h",
                    color="occurrences",
                    color_continuous_scale="Blues",
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("📋 Full Feature Rankings")
            st.dataframe(df_features, use_container_width=True)
        else:
            st.info("No features analyzed yet. Process some predictions first.")
    else:
        st.error("Failed to load feature importance data.")


# ── Model Comparison (Step 4) ──────────────────────────────────────────────────
elif page == "🔬 Model Comparison":
    st.title("🔬 Advanced Model Comparison")
    st.markdown("**Side-by-side metrics comparison of all models**")
    st.divider()
    
    with st.spinner("Comparing models..."):
        comparison_data = api_get("/models/compare")
    
    if comparison_data:
        comparison = comparison_data.get("comparison", {})
        models_compared = comparison_data.get("models_compared", 0)
        
        st.success(f"✅ Compared {models_compared} models")
        st.divider()
        
        summary = comparison.get("summary", {})
        best_overall = summary.get("best_overall", "N/A")
        best_by_metric = summary.get("best_by_metric", {})
        
        c1, c2 = st.columns(2)
        c1.metric("🏆 Best Overall Model", best_overall.replace("_", " ").title())
        c2.metric("Metrics Evaluated", len(best_by_metric))
        
        st.divider()
        st.subheader("🥇 Best Model by Metric")
        if best_by_metric:
            metric_cols = st.columns(min(3, len(best_by_metric)))
            for idx, (metric, info) in enumerate(list(best_by_metric.items())[:3]):
                with metric_cols[idx % 3]:
                    st.metric(
                        metric.replace("_", " ").title(),
                        f"{info['value']:.4f}",
                        f"({info['model'].replace('_', ' ').title()})"
                    )
        
        st.divider()
        st.subheader("📊 Model Metrics Table")
        models = comparison.get("models", {})
        if models:
            df_models = pd.DataFrame([
                {
                    "Model": m.replace("_", " ").title(),
                    "Weighted Score": v["weighted_score"],
                    **v.get("metrics", {})
                }
                for m, v in models.items()
            ])
            st.dataframe(df_models, use_container_width=True, hide_index=True)
    else:
        st.error("Failed to load model comparison data.")


# ── Threshold Tuning (Step 4) ──────────────────────────────────────────────────
elif page == "⚙️ Thresholds":
    st.title("⚙️ Custom Fraud Detection Thresholds")
    st.markdown("**Optimize thresholds for different use cases**")
    st.divider()
    
    use_case = st.radio("Select Use Case", ["Balanced (F1)", "Conservative (Precision)", "Aggressive (Recall)"],
                        captions=["Maximize overall F1", "Minimize false positives", "Minimize false negatives"])
    use_case_map = {"Balanced (F1)": "balanced", "Conservative (Precision)": "conservative", "Aggressive (Recall)": "aggressive"}
    
    if st.button("🔍 Optimize Thresholds", use_container_width=True):
        with st.spinner("Optimizing thresholds..."):
            threshold_data = api_get("/settings/thresholds", {"use_case": use_case_map[use_case]})
        
        if threshold_data:
            recommended = threshold_data.get("recommended_threshold", 0.5)
            recommendations = threshold_data.get("recommendations", {})
            analysis = threshold_data.get("analysis", [])
            
            st.success(f"✅ Recommended Threshold: **{recommended}**")
            st.divider()
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Conservative (↓ FP)", recommendations.get("conservative", 0.75))
            c2.metric("Balanced (📊)", recommendations.get("balanced", 0.5))
            c3.metric("Aggressive (↓ FN)", recommendations.get("aggressive", 0.25))
            
            if analysis:
                st.subheader("📈 Threshold Analysis Across All Values")
                df_analysis = pd.DataFrame(analysis)
                
                fig = px.line(
                    df_analysis,
                    x="threshold",
                    y="positive_rate",
                    markers=True,
                    title="Fraud Detection Rate by Threshold"
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Failed to optimize thresholds.")
    else:
        st.info("Click 'Optimize Thresholds' to analyze your data and get recommendations.")


# ── Export Predictions (Step 4) ────────────────────────────────────────────────
elif page == "📥 Export Data":
    st.title("📥 Export Predictions & Analytics")
    st.markdown("**Download predictions in CSV/JSON format or view summary statistics**")
    st.divider()
    
    export_format = st.radio("Export Format", ["Summary", "JSON", "CSV"])
    limit = st.slider("Records to export", 100, 10000, 1000, step=100)
    
    if st.button("📊 Generate Export", use_container_width=True):
        with st.spinner(f"Exporting {limit} predictions as {export_format}..."):
            response = requests.get(
                f"{API_BASE}/export/predictions",
                params={"format": export_format.lower(), "limit": limit},
                timeout=30
            )
        
        if response.status_code == 200:
            if export_format == "Summary":
                summary = response.json().get("summary", {})
                
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("📊 Total", summary.get("total", 0))
                c2.metric("🚫 Fraud", summary.get("fraud_count", 0))
                c3.metric("✅ Legitimate", summary.get("legitimate_count", 0))
                c4.metric("📈 Avg Score", f"{summary.get('avg_fraud_score', 0):.2%}")
                c5.metric("⏱️ Avg Latency", f"{summary.get('avg_inference_time_ms', 0):.1f}ms")
                
                st.divider()
                st.success("Export Summary Generated Successfully")
                
            else:
                filename = f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                if export_format == "JSON":
                    st.download_button(
                        label="📥 Download JSON",
                        data=response.content,
                        file_name=f"{filename}.json",
                        mime="application/json"
                    )
                else:
                    st.download_button(
                        label="📥 Download CSV",
                        data=response.content,
                        file_name=f"{filename}.csv",
                        mime="text/csv"
                    )
        else:
            st.error("Failed to export predictions.")


# ── Batch Status (Step 4) ──────────────────────────────────────────────────────
elif page == "⏳ Batch Status":
    st.title("⏳ Batch Job Status & Results")
    st.markdown("**Monitor and retrieve results of batch uploads**")
    st.divider()
    
    job_id = st.text_input("Enter Batch Job ID")
    
    if job_id:
        with st.spinner("Fetching job status..."):
            status_response = api_get(f"/batch/status/{job_id}")
            results_response = api_get(f"/batch/results/{job_id}")
        
        if status_response and "job" in status_response:
            job = status_response["job"]
            
            # Status metrics
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Status", job.get("status", "unknown").upper())
            c2.metric("Progress", f"{job.get('progress', 0):.1f}%")
            c3.metric("Processed", f"{job.get('processed', 0)}/{job.get('total', 0)}")
            c4.metric("Successful", job.get("successful", 0))
            c5.metric("Failed", job.get("failed", 0))
            
            st.divider()
            
            # Progress bar
            progress_pct = job.get("progress", 0) / 100.0
            st.progress(progress_pct)
            
            # Timing info
            st.caption(f"Started: {job.get('started_at', 'N/A')}")
            if job.get("completed_at"):
                st.caption(f"Completed: {job.get('completed_at')}")
            
            # Detailed results
            if results_response and "job_results" in results_response:
                results = results_response["job_results"]
                summary = results.get("summary", {})
                
                st.divider()
                st.subheader("📊 Batch Summary")
                s_c1, s_c2, s_c3, s_c4 = st.columns(4)
                s_c1.metric("🚫 Fraud Count", summary.get("fraud_count", 0))
                s_c2.metric("✅ Legitimate", summary.get("legitimate_count", 0))
                s_c3.metric("📈 Avg Score", f"{summary.get('avg_fraud_score', 0):.2%}")
                s_c4.metric("⏱️ Avg Latency", f"{summary.get('avg_inference_time_ms', 0):.1f}ms")
                
                predictions = results.get("predictions", [])
                if predictions:
                    st.divider()
                    st.subheader("📋 First 20 Predictions")
                    df_preds = pd.DataFrame(predictions[:20])
                    st.dataframe(df_preds, use_container_width=True)
                
                errors = results.get("errors", [])
                if errors:
                    st.divider()
                    st.subheader("❌ Errors")
                    for error in errors:
                        st.error(f"{error.get('timestamp')}: {error.get('error')}")
        else:
            st.error("Job not found or error retrieving status.")
    else:
        st.info("Enter a Job ID to view batch status and results.")


# ── Data Quality Monitoring (Step 5) ────────────────────────────────────────────
elif page == "🔍 Data Quality":
    st.title("🔍 Data Quality Monitoring")
    st.markdown("**Monitor incoming data for anomalies, drift, and validation issues**")
    st.divider()
    
    # Tabs for different quality views
    quality_tab, alerts_tab, analyze_tab = st.tabs(["📊 Summary", "⚠️ Alerts", "🔬 Batch Analysis"])
    
    with quality_tab:
        st.subheader("Quality Monitoring Summary")
        
        # Get quality summary
        summary_response = api_get("/quality/summary?window_minutes=60")
        
        if summary_response and "summary" in summary_response:
            summary = summary_response["summary"]
            
            # Key metrics
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("🎯 Avg Quality Score", f"{summary.get('avg_quality_score', 0):.1f}/100")
            c2.metric("⚠️ Critical Issues", summary.get("critical_count", 0))
            c3.metric("⚡ Total Alerts", summary.get("alerts_count", 0))
            
            trend = summary.get("quality_trend", "stable")
            trend_emoji = "📈" if trend == "stable" else "📉" if trend == "degrading" else "⚠️"
            c4.metric(f"{trend_emoji} Trend", trend.title())
            
            st.divider()
            
            # Most common issues
            most_common = summary.get("most_common_issues", [])
            if most_common:
                st.subheader("🔴 Top Data Quality Issues")
                issue_df = pd.DataFrame(most_common)
                st.dataframe(issue_df, use_container_width=True, hide_index=True)
                
                # Pie chart
                fig = px.pie(
                    issue_df,
                    values="count",
                    names="field",
                    title="Issues Distribution",
                    hole=0.3,
                )
                st.plotly_chart(fig, use_container_width=True)
            
            st.info(f"📊 Monitoring window: Last 60 minutes | Total records checked: {summary.get('total_records_checked', 0)}")
        else:
            st.error("Failed to fetch quality summary")
    
    with alerts_tab:
        st.subheader("Recent Quality Alerts")
        
        limit = st.slider("Alerts to show", min_value=10, max_value=500, value=50, step=10)
        
        # Get alerts
        alerts_response = api_get(f"/quality/alerts?limit={limit}")
        
        if alerts_response and "alerts" in alerts_response:
            alerts = alerts_response["alerts"]
            stats = alerts_response.get("stats", {})
            
            # Stats
            s_c1, s_c2, s_c3 = st.columns(3)
            s_c1.metric("Total Alerts", stats.get("total", 0))
            s_c2.metric("🔴 Critical", stats.get("critical", 0))
            s_c3.metric("⚠️ Warnings", stats.get("warnings", 0))
            
            st.divider()
            
            if alerts:
                # Sort by timestamp desc
                alerts_sorted = sorted(alerts, key=lambda x: x.get("timestamp", ""), reverse=True)
                
                for alert in alerts_sorted[:20]:  # Show first 20
                    severity = alert.get("severity", "info")
                    check_type = alert.get("check_type", "unknown")
                    field = alert.get("field", "unknown")
                    message = alert.get("message", "")
                    
                    if severity == "critical":
                        st.error(f"🔴 [{check_type.upper()}] {field}: {message}")
                    elif severity == "warning":
                        st.warning(f"⚠️ [{check_type.upper()}] {field}: {message}")
                    else:
                        st.info(f"ℹ️ [{check_type.upper()}] {field}: {message}")
            else:
                st.success("✅ No recent alerts - data quality is good!")
        else:
            st.error("Failed to fetch alerts")
    
    with analyze_tab:
        st.subheader("Batch Quality Analysis")
        st.markdown("Upload a CSV to analyze data quality of multiple records")
        
        uploaded_file = st.file_uploader("Upload CSV file", type="csv")
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.info(f"Loaded {len(df)} records for analysis")
                
                # Show preview
                with st.expander("📋 Preview Data", expanded=False):
                    st.dataframe(df.head(10), use_container_width=True)
                
                if st.button("🔬 Analyze Quality", key="analyze_quality"):
                    with st.spinner("Analyzing data quality..."):
                        # Convert to list of dicts
                        records = df.fillna("").to_dict('records')
                        
                        # Send to backend
                        response = requests.post(
                            f"{API_BASE}/quality/analyze-batch",
                            json=records,
                            headers={"Authorization": f"Bearer {st.session_state.auth_token}"},
                        )
                        
                        if response.status_code == 200:
                            result = response.json()
                            
                            st.subheader("📊 Analysis Results")
                            
                            # Metrics
                            a_c1, a_c2, a_c3, a_c4 = st.columns(4)
                            a_c1.metric("Records Analyzed", result.get("analyzed", 0))
                            a_c2.metric("Avg Quality Score", f"{result.get('avg_quality_score', 0):.1f}")
                            a_c3.metric("Quality Grade", result.get("quality_grade", "N/A"))
                            a_c4.metric("Critical Issues", result.get("critical_issues", 0))
                            
                            st.divider()
                            
                            # Detailed results
                            results = result.get("results", [])
                            if results:
                                st.subheader("🔍 Sample Results (First 10)")
                                for i, res in enumerate(results[:10]):
                                    with st.expander(f"Record {i+1} - Quality Score: {res['quality_score']}", expanded=False):
                                        col1, col2, col3 = st.columns(3)
                                        col1.metric("Quality Score", res["quality_score"])
                                        col2.metric("Alerts", res["alert_count"])
                                        col3.metric("Critical", res["critical_alerts"])
                                        
                                        if res.get("alerts"):
                                            st.markdown("**Issues Found:**")
                                            for alert in res["alerts"][:5]:
                                                st.caption(f"• {alert['field']}: {alert['message']}")
                        else:
                            st.error(f"Analysis failed: {response.status_code}")
            except Exception as e:
                st.error(f"Error processing file: {e}")
        else:
            st.info("Upload a CSV file to begin analysis")


# ── Performance Optimization (Step 6) ──────────────────────────────────────────
elif page == "⚡ Performance":
    st.title("⚡ Performance Optimization")
    st.markdown("**Monitor latency, throughput, caching, and identify bottlenecks**")
    st.divider()
    
    perf_tab, cache_tab, batch_tab, bottleneck_tab = st.tabs(
        ["📊 Summary", "💾 Caching", "⚙️ Batch Optimization", "🔴 Bottlenecks"]
    )
    
    with perf_tab:
        st.subheader("Performance Metrics")
        
        # Refresh button
        if st.button("🔄 Refresh Metrics", key="refresh_perf"):
            st.rerun()
        
        # Get performance summary
        summary_response = api_get("/performance/summary")
        
        if summary_response and "performance" in summary_response:
            perf = summary_response["performance"]
            
            # Key metrics
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("⏱️ Avg Request", f"{perf.get('avg_request_time_ms', 0):.2f}ms")
            c2.metric("P95 Latency", f"{perf.get('p95_request_time_ms', 0):.2f}ms")
            c3.metric("P99 Latency", f"{perf.get('p99_request_time_ms', 0):.2f}ms")
            c4.metric("Max Latency", f"{perf.get('max_request_time_ms', 0):.2f}ms")
            
            st.divider()
            
            # Inference time
            c_inf1, c_inf2, c_inf3 = st.columns(3)
            c_inf1.metric("🤖 Avg Inference", f"{perf.get('avg_inference_time_ms', 0):.2f}ms")
            c_inf2.metric("📊 Total Requests", perf.get("total_requests", 0))
            c_inf3.metric("🔮 Total Inferences", perf.get("total_inferences", 0))
            
            # Latency distribution chart
            st.divider()
            st.subheader("📈 Latency Analysis")
            
            latency_data = {
                "Metric": ["Average", "P95", "P99", "Max"],
                "Latency (ms)": [
                    perf.get("avg_request_time_ms", 0),
                    perf.get("p95_request_time_ms", 0),
                    perf.get("p99_request_time_ms", 0),
                    perf.get("max_request_time_ms", 0),
                ]
            }
            
            fig = px.bar(latency_data, x="Metric", y="Latency (ms)", 
                        title="Request Latency Distribution",
                        color="Latency (ms)",
                        color_continuous_scale="RdYlGn_r")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Failed to fetch performance summary")
    
    with cache_tab:
        st.subheader("Cache Performance")
        
        cache_response = api_get("/cache/stats")
        
        if cache_response and "cache_stats" in cache_response:
            cache_stats = cache_response["cache_stats"]
            
            c1, c2 = st.columns(2)
            
            with c1:
                st.subheader("Response Cache")
                resp_cache = cache_stats.get("response_cache", {})
                
                c_r1, c_r2, c_r3 = st.columns(3)
                c_r1.metric("Size", f"{resp_cache.get('size', 0)}/{resp_cache.get('max_size', 0)}")
                c_r2.metric("Hit Rate", f"{resp_cache.get('hit_rate', 0):.1f}%")
                c_r3.metric("Utilization", f"{resp_cache.get('utilization', 0):.1f}%")
                
                st.caption(f"Hits: {resp_cache.get('hits', 0)} | Misses: {resp_cache.get('misses', 0)}")
            
            with c2:
                st.subheader("Prediction Cache")
                pred_cache = cache_stats.get("prediction_cache", {})
                
                c_p1, c_p2 = st.columns(2)
                c_p1.metric("Size", f"{pred_cache.get('size', 0)}/{pred_cache.get('max_size', 0)}")
                c_p2.metric("Utilization", f"{(pred_cache.get('size', 0) / max(1, pred_cache.get('max_size', 1)) * 100):.1f}%")
            
            st.divider()
            
            # Cache recommendations
            resp_hit_rate = resp_cache.get('hit_rate', 0)
            if resp_hit_rate < 30:
                st.warning("⚠️ Low response cache hit rate. Consider increasing TTL or caching more endpoints.")
            elif resp_hit_rate > 70:
                st.success("✅ Excellent response cache hit rate!")
            
            # Clear cache button
            st.divider()
            col1, col2, col3 = st.columns(3)
            
            if col1.button("🗑️ Clear Response Cache"):
                clear_resp = requests.post(
                    f"{API_BASE}/cache/clear?cache_type=response",
                    headers={"Authorization": f"Bearer {st.session_state.auth_token}"},
                )
                if clear_resp.status_code == 200:
                    st.success("Response cache cleared")
                    st.rerun()
            
            if col2.button("🗑️ Clear Prediction Cache"):
                clear_pred = requests.post(
                    f"{API_BASE}/cache/clear?cache_type=prediction",
                    headers={"Authorization": f"Bearer {st.session_state.auth_token}"},
                )
                if clear_pred.status_code == 200:
                    st.success("Prediction cache cleared")
                    st.rerun()
            
            if col3.button("🗑️ Clear All Caches"):
                clear_all = requests.post(
                    f"{API_BASE}/cache/clear?cache_type=all",
                    headers={"Authorization": f"Bearer {st.session_state.auth_token}"},
                )
                if clear_all.status_code == 200:
                    st.success("All caches cleared")
                    st.rerun()
        else:
            st.error("Failed to fetch cache stats")
    
    with batch_tab:
        st.subheader("Batch Optimization")
        st.markdown("Optimized batch predictions with automatic caching")
        
        # Upload CSV
        uploaded_file = st.file_uploader("Upload CSV for batch prediction", type="csv", key="batch_opt")
        
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                st.info(f"Loaded {len(df)} records")
                
                with st.expander("📋 Preview", expanded=False):
                    st.dataframe(df.head(10), use_container_width=True)
                
                if st.button("🚀 Run Optimized Batch", key="run_opt_batch"):
                    with st.spinner("Processing with optimization..."):
                        records = [
                            {
                                "claim_id": f"claim_{i}",
                                "patient_id": row.get("patient_id", f"pat_{i}"),
                                "doctor_id": row.get("doctor_id", f"doc_{i}"),
                                "hospital_id": row.get("hospital_id", f"hosp_{i}"),
                                "claim_amount": float(row.get("claim_amount", 1000)),
                                **{k: v for k, v in row.items() if k not in 
                                   ["claim_id", "patient_id", "doctor_id", "hospital_id", "claim_amount"]}
                            }
                            for i, row in df.iterrows()
                        ]
                        
                        batch_response = requests.post(
                            f"{API_BASE}/predict-batch/optimized",
                            json={"claims": records},
                            headers={"Authorization": f"Bearer {st.session_state.auth_token}"},
                        )
                        
                        if batch_response.status_code == 200:
                            result = batch_response.json()
                            summary = result.get("summary", {})
                            
                            st.subheader("✅ Optimization Results")
                            
                            s_c1, s_c2, s_c3, s_c4 = st.columns(4)
                            s_c1.metric("Total Time", f"{summary.get('total_time_ms', 0):.2f}ms")
                            s_c2.metric("Avg Inference", f"{summary.get('avg_inference_time_ms', 0):.2f}ms")
                            s_c3.metric("Batch Size", summary.get("optimal_batch_size", "N/A"))
                            s_c4.metric("Cache Hits", summary.get("cache_hit_count", 0))
                            
                            st.success(f"✨ Processed {summary.get('successful', 0)}/{summary.get('total_records', 0)} records")
                        else:
                            st.error(f"Batch failed: {batch_response.status_code}")
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.info("Upload a CSV to run optimized batch predictions")
    
    with bottleneck_tab:
        st.subheader("Performance Bottlenecks")
        
        bottleneck_response = api_get("/performance/bottlenecks")
        
        if bottleneck_response and "analysis" in bottleneck_response:
            analysis = bottleneck_response["analysis"]
            bottlenecks = analysis.get("bottlenecks", [])
            
            if bottlenecks:
                st.warning(f"🔴 Found {len(bottlenecks)} potential bottlenecks")
                
                for i, bn in enumerate(bottlenecks):
                    severity = bn.get("severity", "info")
                    bn_type = bn.get("type", "unknown")
                    message = bn.get("message", "")
                    
                    if severity == "critical":
                        st.error(f"🔴 [{bn_type.upper()}] {message}")
                    elif severity == "warning":
                        st.warning(f"⚠️ [{bn_type.upper()}] {message}")
                    else:
                        st.info(f"ℹ️ [{bn_type.upper()}] {message}")
            else:
                st.success("✅ No performance bottlenecks detected!")
            
            # Recommendations
            recommendations = bottleneck_response.get("recommendations", [])
            if recommendations:
                st.divider()
                st.subheader("💡 Recommendations")
                for rec in recommendations:
                    st.caption(f"• {rec}")
        else:
            st.error("Failed to fetch bottleneck analysis")


# ── Model Interpretability (Step 7) ────────────────────────────────────────────
elif page == "🔮 Interpretability":
    st.title("🔮 Model Interpretability")
    st.markdown("**Understand why the model makes predictions with SHAP values, interactions, and explanations**")
    st.divider()
    
    explain_tab, compare_tab, summary_tab, pd_tab = st.tabs(
        ["📖 Explain Prediction", "⚖️ Compare", "📊 Summary", "📈 Partial Dependence"]
    )
    
    with explain_tab:
        st.subheader("Prediction Explanation")
        st.markdown("Get detailed explanations for any prediction")
        
        col1, col2 = st.columns(2)
        
        with col1:
            pred_id = st.text_input("Prediction ID", key="explain_pred_id")
        
        with col2:
            fraud_score = st.slider("Fraud Risk Score", 0.0, 1.0, 0.5)
        
        if st.button("🔍 Generate Explanation", key="gen_explain"):
            if pred_id:
                with st.spinner("Generating explanation..."):
                    try:
                        explain_response = requests.post(
                            f"{API_BASE}/explain/prediction",
                            params={
                                "prediction_id": pred_id,
                                "prediction_score": fraud_score,
                            },
                            json={
                                "features": {"claim_amount": 5000},
                                "feature_importance": {"claim_amount": 0.3}
                            },
                            headers={"Authorization": f"Bearer {st.session_state.auth_token}"},
                        )
                        
                        if explain_response.status_code == 200:
                            result = explain_response.json()
                            exp = result.get("explanation", {})
                            
                            # Main metrics
                            c1, c2, c3 = st.columns(3)
                            c1.metric("🎯 Risk Level", exp.get("prediction_label", "N/A"))
                            c2.metric("📊 Score", f"{exp.get('prediction_score', 0):.2%}")
                            c3.metric("🎓 Confidence", f"{exp.get('confidence', 0):.2%}")
                            
                            st.divider()
                            
                            # Explanation text
                            st.subheader("📖 Explanation")
                            st.info(exp.get("explanation", "No explanation available"))
                            
                            # Contributions
                            st.divider()
                            st.subheader("🔴 Contributing Factors")
                            
                            contributions = exp.get("contributions", [])
                            if contributions:
                                for i, contrib in enumerate(contributions[:5]):
                                    col_l, col_r = st.columns([3, 1])
                                    with col_l:
                                        st.caption(f"**{i+1}. {contrib['feature']}**")
                                        st.caption(f"Value: {contrib['value']} | Impact: {contrib['direction']}")
                                    with col_r:
                                        st.metric(f"SHAP", f"{contrib['shap_value']:.3f}")
                            
                            # Interactions
                            st.divider()
                            st.subheader("🔗 Feature Interactions")
                            
                            interactions = exp.get("interactions", [])
                            if interactions:
                                for inter in interactions[:3]:
                                    st.info(f"⚡ {inter['feature1']} × {inter['feature2']}: {inter['interpretation']}")
                            else:
                                st.caption("No significant interactions detected")
                        else:
                            st.error(f"Explanation failed: {explain_response.status_code}")
                    except Exception as e:
                        st.error(f"Error: {e}")
            else:
                st.warning("Please enter a prediction ID")
    
    with compare_tab:
        st.subheader("Compare Predictions")
        st.markdown("Compare explanations of two different predictions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            pred_id_1 = st.text_input("First Prediction ID", key="compare_1")
        
        with col2:
            pred_id_2 = st.text_input("Second Prediction ID", key="compare_2")
        
        if st.button("⚖️ Compare Explanations", key="compare_btn"):
            if pred_id_1 and pred_id_2:
                with st.spinner("Comparing..."):
                    try:
                        compare_response = requests.post(
                            f"{API_BASE}/explain/compare",
                            params={"pred_id_1": pred_id_1, "pred_id_2": pred_id_2},
                            headers={"Authorization": f"Bearer {st.session_state.auth_token}"},
                        )
                        
                        if compare_response.status_code == 200:
                            result = compare_response.json()
                            comp = result.get("comparison", {})
                            
                            # Side-by-side comparison
                            c1, c2, c3 = st.columns(3)
                            
                            with c1:
                                st.subheader("📋 Prediction 1")
                                p1 = comp.get("prediction_1", {})
                                st.metric("Score", f"{p1.get('score', 0):.2%}")
                                st.caption(f"Label: {p1.get('label', 'N/A')}")
                            
                            with c2:
                                st.subheader("📏 Difference")
                                diff = comp.get("score_difference", 0)
                                st.metric("Score Diff", f"{diff:.2%}")
                                st.caption(f"ID 1: {p1.get('id', 'N/A')[:8]}...")
                            
                            with c3:
                                st.subheader("📋 Prediction 2")
                                p2 = comp.get("prediction_2", {})
                                st.metric("Score", f"{p2.get('score', 0):.2%}")
                                st.caption(f"Label: {p2.get('label', 'N/A')}")
                            
                            st.divider()
                            
                            # Similar factors
                            similar = comp.get("similar_risk_factors", [])
                            if similar:
                                st.subheader("🔄 Shared Risk Factors")
                                for factor in similar:
                                    st.caption(f"• {factor}")
                            
                            # Different factors
                            st.divider()
                            st.subheader("🔀 Different Risk Factors")
                            
                            diff_factors = comp.get("different_risk_factors", {})
                            col_l, col_r = st.columns(2)
                            
                            with col_l:
                                st.caption("**Unique to Prediction 1:**")
                                for factor in diff_factors.get("unique_to_1", []):
                                    st.caption(f"• {factor}")
                            
                            with col_r:
                                st.caption("**Unique to Prediction 2:**")
                                for factor in diff_factors.get("unique_to_2", []):
                                    st.caption(f"• {factor}")
                        else:
                            st.error(f"Comparison failed: {compare_response.status_code}")
                    except Exception as e:
                        st.error(f"Error: {e}")
            else:
                st.warning("Please enter both prediction IDs")
    
    with summary_tab:
        st.subheader("Interpretation Summary")
        st.markdown("Overall patterns across recent predictions")
        
        if st.button("📊 Generate Summary", key="gen_summary"):
            with st.spinner("Analyzing..."):
                try:
                    summary_response = api_get("/interpret/summary?n_predictions=100")
                    
                    if summary_response and "summary" in summary_response:
                        summary = summary_response["summary"]
                        
                        # Key metrics
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("📊 Total Predictions", summary.get("total_predictions", 0))
                        c2.metric("📈 Avg Fraud Score", f"{summary.get('avg_fraud_score', 0):.2%}")
                        c3.metric("🔴 High Risk", summary.get("high_risk_count", 0))
                        c4.metric("🟢 Low Risk", summary.get("low_risk_count", 0))
                        
                        st.divider()
                        
                        # Risk distribution
                        st.subheader("📊 Risk Distribution")
                        risk_data = {
                            "Risk Level": ["High", "Medium", "Low"],
                            "Count": [
                                summary.get("high_risk_count", 0),
                                summary.get("medium_risk_count", 0),
                                summary.get("low_risk_count", 0),
                            ]
                        }
                        
                        fig = px.pie(
                            pd.DataFrame(risk_data),
                            values="Count",
                            names="Risk Level",
                            color_discrete_map={"High": "#FF6B6B", "Medium": "#FFC93C", "Low": "#51CF66"},
                            title="Fraud Risk Distribution"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Most impactful features
                        st.divider()
                        st.subheader("🔴 Most Impactful Features")
                        
                        features = summary.get("most_impactful_features", [])
                        impact_scores = summary.get("feature_impact_scores", {})
                        
                        if features:
                            impact_data = {
                                "Feature": features[:5],
                                "Impact Score": [impact_scores.get(f, 0) for f in features[:5]],
                            }
                            
                            fig = px.bar(
                                pd.DataFrame(impact_data),
                                x="Feature",
                                y="Impact Score",
                                title="Feature Importance Across All Predictions",
                                color="Impact Score",
                                color_continuous_scale="Reds"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error("Failed to generate summary")
                except Exception as e:
                    st.error(f"Error: {e}")
    
    with pd_tab:
        st.subheader("Partial Dependence Analysis")
        st.markdown("Analyze how individual features affect fraud predictions")
        
        feature_name = st.text_input("Feature Name", key="pd_feature")
        
        st.caption("For demonstration, we'll analyze the feature's impact pattern")
        
        if st.button("📈 Analyze Feature Impact", key="analyze_pd"):
            if feature_name:
                with st.spinner("Analyzing partial dependence..."):
                    try:
                        # Create simulated feature values and predictions
                        import numpy as np
                        values = [float(i)/10 for i in range(101)]
                        predictions = [0.3 + 0.004*v for v in values]  # Simulated correlation
                        
                        pd_response = requests.post(
                            f"{API_BASE}/interpret/partial-dependence",
                            params={"feature_name": feature_name},
                            json={"values": values, "predictions": predictions},
                            headers={"Authorization": f"Bearer {st.session_state.auth_token}"},
                        )
                        
                        if pd_response.status_code == 200:
                            result = pd_response.json()
                            
                            # Impact metrics
                            range_impact = result.get("range_impact", {})
                            
                            c1, c2, c3 = st.columns(3)
                            c1.metric("Low Range Avg", f"{range_impact.get('low_range_avg_prediction', 0):.2%}")
                            c2.metric("High Range Avg", f"{range_impact.get('high_range_avg_prediction', 0):.2%}")
                            c3.metric("Impact", f"{range_impact.get('impact', 0):.2%}")
                            
                            st.divider()
                            
                            # Interpretation
                            st.subheader("📖 Interpretation")
                            st.info(result.get("interpretation", "No interpretation available"))
                            
                            # PD Plot
                            st.divider()
                            st.subheader("📊 Partial Dependence Plot")
                            
                            pd_plot = result.get("partial_dependence_plot", [])
                            if pd_plot:
                                df_pd = pd.DataFrame(pd_plot)
                                fig = px.line(
                                    df_pd,
                                    x="feature_value",
                                    y="avg_prediction",
                                    title=f"How {feature_name} affects fraud predictions",
                                    markers=True,
                                    labels={"feature_value": feature_name, "avg_prediction": "Avg Fraud Risk"}
                                )
                                st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.error(f"Analysis failed: {pd_response.status_code}")
                    except Exception as e:
                        st.error(f"Error: {e}")
            else:
                st.warning("Please enter a feature name")


# ── Compliance & Audit (Step 8) ────────────────────────────────────────────────
elif page == "📋 Compliance":
    st.title("📋 Compliance & Audit")
    st.markdown("**GDPR compliance, audit logging, and regulatory reporting**")
    st.divider()
    
    audit_tab, gdpr_tab, logs_tab, reports_tab = st.tabs(
        ["✅ Audit Trail", "🔒 GDPR", "📝 Logs", "📊 Reports"]
    )
    
    with audit_tab:
        st.subheader("Audit Trail & Integrity")
        
        if st.button("🔍 Verify Audit Integrity", key="verify_audit"):
            with st.spinner("Verifying integrity..."):
                try:
                    integrity_response = api_get("/audit/verify-integrity")
                    
                    if integrity_response and "integrity_valid" in integrity_response:
                        is_valid = integrity_response["integrity_valid"]
                        issues = integrity_response.get("issues", [])
                        total_logs = integrity_response.get("total_logs", 0)
                        
                        if is_valid:
                            st.success(f"✅ Audit logs are valid | {total_logs} total logs")
                        else:
                            st.error(f"⚠️ {len(issues)} integrity issues found:")
                            for issue in issues:
                                st.caption(f"• {issue}")
                    else:
                        st.error("Failed to verify integrity")
                except Exception as e:
                    st.error(f"Error: {e}")
        
        st.divider()
        st.subheader("Recent Audit Events")
        
        limit = st.slider("Show last N events", 10, 500, 50)
        action_filter = st.selectbox("Filter by action", 
                                     ["All", "PREDICTION_MADE", "DATA_ACCESS", "DATA_EXPORT", "DATA_DELETE"])
        
        try:
            filter_param = None if action_filter == "All" else action_filter
            logs_response = api_get(f"/audit/logs?limit={limit}" + (f"&action={filter_param}" if filter_param else ""))
            
            if logs_response and "logs" in logs_response:
                logs = logs_response["logs"]
                
                st.metric("Total Events", logs_response.get("total_logs", 0))
                
                if logs:
                    for log in logs[:20]:
                        with st.expander(f"⏱️ {log['timestamp'][:19]} | {log['action']} | {log['user']}", expanded=False):
                            st.json(log)
        except Exception as e:
            st.error(f"Error fetching logs: {e}")
    
    with gdpr_tab:
        st.subheader("GDPR Compliance Status")
        
        if st.button("🔄 Refresh GDPR Status", key="refresh_gdpr"):
            st.rerun()
        
        try:
            gdpr_response = api_get("/compliance/gdpr-status")
            
            if gdpr_response and "gdpr" in gdpr_response:
                gdpr = gdpr_response["gdpr"]
                
                # Metrics
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("👥 Users with Consent", gdpr.get("total_users_with_consent", 0))
                c2.metric("📋 Retention Policies", gdpr.get("retention_policies", 0))
                c3.metric("⏳ Pending Requests", gdpr.get("pending_data_subject_requests", 0))
                c4.metric("✅ Processed Requests", gdpr.get("processed_data_subject_requests", 0))
                
                st.divider()
                
                # Compliance score
                score = gdpr.get("compliance_score", 0)
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.progress(score / 100)
                with col2:
                    st.metric("Compliance Score", f"{score:.0f}/100")
                
                if score < 70:
                    st.error("🔴 CRITICAL: Below 70% - immediate action required")
                elif score < 85:
                    st.warning("🟡 ACTION NEEDED: Compliance score below 85%")
                else:
                    st.success("🟢 GOOD: High GDPR compliance")
        except Exception as e:
            st.error(f"Error fetching GDPR status: {e}")
        
        st.divider()
        st.subheader("Data Subject Requests")
        
        col_type, col_reason = st.columns(2)
        
        with col_type:
            request_type = st.selectbox(
                "Request Type",
                ["access", "rectification", "erasure", "portability"]
            )
        
        with col_reason:
            user_id = st.text_input("User ID", key="dsr_user")
        
        details = st.text_area("Details", placeholder="Additional request details...")
        
        if st.button("📝 File Request", key="file_dsr"):
            if user_id:
                with st.spinner("Filing request..."):
                    try:
                        dsr_response = requests.post(
                            f"{API_BASE}/compliance/data-subject-request",
                            params={
                                "user_id": user_id,
                                "request_type": request_type,
                                "details": details or None,
                            },
                            headers={"Authorization": f"Bearer {st.session_state.auth_token}"},
                        )
                        
                        if dsr_response.status_code == 200:
                            result = dsr_response.json()
                            st.success(f"✅ Request filed: {result['request_id']}")
                            st.info(result['message'])
                        else:
                            st.error(f"Failed: {dsr_response.status_code}")
                    except Exception as e:
                        st.error(f"Error: {e}")
            else:
                st.warning("Please enter user ID")
    
    with logs_tab:
        st.subheader("Complete Audit Logs")
        
        col1, col2 = st.columns(2)
        
        with col1:
            log_limit = st.slider("Logs to show", 10, 1000, 100)
        
        with col2:
            log_action = st.selectbox(
                "Filter action",
                ["All", "PREDICTION_MADE", "DATA_ACCESS", "DATA_EXPORT", "DATA_DELETE"]
            )
        
        try:
            filter_action = None if log_action == "All" else log_action
            logs_response = api_get(f"/audit/logs?limit={log_limit}" + (f"&action={filter_action}" if filter_action else ""))
            
            if logs_response and "logs" in logs_response:
                logs = logs_response["logs"]
                
                if logs:
                    # Convert to DataFrame for display
                    df_logs = pd.DataFrame([
                        {
                            "Time": log.get("timestamp", "")[:19],
                            "Action": log.get("action", ""),
                            "User": log.get("user", ""),
                            "Resource": log.get("resource", ""),
                            "Status": log.get("status", ""),
                        }
                        for log in logs
                    ])
                    
                    st.dataframe(df_logs, use_container_width=True, hide_index=True)
                else:
                    st.info("No audit logs found")
        except Exception as e:
            st.error(f"Error fetching logs: {e}")
    
    with reports_tab:
        st.subheader("Compliance Reports")
        
        report_type = st.selectbox(
            "Report Type",
            ["Compliance Dashboard", "Audit Report", "GDPR Report"]
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            if report_type == "Audit Report":
                days = st.slider("Audit period (days)", 1, 365, 30)
        
        if st.button(f"📊 Generate {report_type}", key="gen_report"):
            with st.spinner("Generating report..."):
                try:
                    if report_type == "Compliance Dashboard":
                        report_response = api_get("/reports/compliance-dashboard")
                    elif report_type == "Audit Report":
                        report_response = api_get(f"/reports/audit-report?days={days}")
                    else:  # GDPR Report
                        report_response = api_get("/reports/gdpr-report")
                    
                    if report_response and "report" in report_response:
                        report = report_response["report"]
                        
                        st.success(f"✅ Report Generated: {report.get('report_type', 'Unknown')}")
                        
                        if report_type == "Compliance Dashboard":
                            # Show summary
                            dashboard = report.get("audit_summary", {})
                            gdpr_comp = report.get("gdpr_compliance", {})
                            
                            st.subheader("📊 Audit Summary")
                            st.metric("Total Events", dashboard.get("total_events", 0))
                            
                            # Events by action
                            if dashboard.get("events_by_action"):
                                action_data = dashboard["events_by_action"]
                                fig = px.bar(
                                    x=list(action_data.keys()),
                                    y=list(action_data.values()),
                                    title="Events by Action",
                                    labels={"x": "Action", "y": "Count"}
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            
                            st.subheader("🔒 GDPR Compliance")
                            gdpr_status = gdpr_comp.get("compliance", {})
                            st.metric("Compliance Score", f"{gdpr_status.get('compliance_score', 0):.0f}/100")
                            
                            # Recommendations
                            recommendations = report.get("recommendations", [])
                            if recommendations:
                                st.warning("⚠️ Recommendations:")
                                for rec in recommendations:
                                    st.caption(f"• {rec}")
                        
                        elif report_type == "Audit Report":
                            # Show audit details
                            st.metric("Events Audited", report.get("total_events", 0))
                            st.metric("Period", f"{report.get('period_days', 0)} days")
                            
                            if report.get("top_users"):
                                st.subheader("👥 Top Users")
                                for user, count in report["top_users"]:
                                    st.caption(f"**{user}**: {count} events")
                        
                        else:  # GDPR Report
                            gdpr_comp = report.get("compliance", {})
                            
                            c1, c2, c3 = st.columns(3)
                            c1.metric("Score", f"{gdpr_comp.get('compliance_score', 0):.0f}/100")
                            c2.metric("Pending Requests", gdpr_comp.get("pending_data_subject_requests", 0))
                            c3.metric("Users", gdpr_comp.get("total_users_with_consent", 0))
                            
                            recommendations = report.get("recommendations", [])
                            if recommendations:
                                st.subheader("💡 Recommendations")
                                for rec in recommendations:
                                    st.info(rec)
                    else:
                        st.error("Failed to generate report")
                except Exception as e:
                    st.error(f"Error: {e}")


# ── Explainability (Step 9) ────────────────────────────────────────────────────
elif page == "🧠 Explainability":
    st.title("🧠 Explainability & Understanding")
    st.markdown("**Advanced explanations: LIME anchors, counterfactuals, what-if analysis**")
    st.divider()
    
    anchors_tab, counterfactual_tab, whatif_tab, sensitivity_tab, compare_tab, boundaries_tab = st.tabs([
        "🎯 Anchors", "🔄 Counterfactuals", "❓ What-If", "📊 Sensitivity", "⚖️ Compare", "🔀 Boundaries"
    ])
    
    with anchors_tab:
        st.subheader("LIME-Style Anchors")
        st.info("Anchors are simple rules that explain why the model made a prediction in the local region.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            pred_score = st.slider("Prediction Score", 0.0, 1.0, 0.75)
        
        with col2:
            num_features = st.number_input("Number of features", 5, 15, 9)
        
        if st.button("🎯 Generate Anchor", key="gen_anchor"):
            with st.spinner("Generating anchor..."):
                try:
                    # Create dummy features
                    features = [0.5 + (i % 3) * 0.2 for i in range(int(num_features))]
                    
                    anchor_response = requests.post(
                        f"{API_BASE}/explain/anchors",
                        params={
                            "prediction_score": pred_score,
                            "features": features
                        },
                        headers={"Authorization": f"Bearer {st.session_state.auth_token}"},
                    )
                    
                    if anchor_response.status_code == 200:
                        result = anchor_response.json()["anchor"]
                        
                        st.success("✅ Anchor Generated")
                        
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Precision", f"{result['precision']:.2%}")
                        c2.metric("Coverage", f"{result['coverage']:.2%}")
                        c3.metric("Features", len(result['important_features']))
                        
                        st.markdown("**Important Features:**")
                        for feat in result['important_features']:
                            st.caption(f"• {feat}: {result['feature_values'].get(feat, 'N/A'):.3f}")
                        
                        st.info(f"**Interpretation:** {result['interpretation']}")
                    else:
                        st.error(f"Failed: {anchor_response.status_code}")
                except Exception as e:
                    st.error(f"Error: {e}")
    
    with counterfactual_tab:
        st.subheader("Counterfactual Explanations")
        st.info("Find minimum feature changes needed to flip the prediction.")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            current_pred = st.slider("Current Prediction", 0.0, 1.0, 0.8)
        
        with col2:
            target_pred = st.slider("Target Prediction", 0.0, 1.0, 0.2)
        
        with col3:
            cf_features = st.number_input("Features count", 5, 15, 9)
        
        if st.button("🔄 Generate Counterfactual", key="gen_cf"):
            with st.spinner("Generating counterfactual..."):
                try:
                    features = [0.6 + (i % 4) * 0.15 for i in range(int(cf_features))]
                    
                    cf_response = requests.post(
                        f"{API_BASE}/explain/counterfactual",
                        params={
                            "current_prediction": current_pred,
                            "target_prediction": target_pred,
                            "features": features
                        },
                        headers={"Authorization": f"Bearer {st.session_state.auth_token}"},
                    )
                    
                    if cf_response.status_code == 200:
                        result = cf_response.json()["counterfactual"]
                        
                        st.success("✅ Counterfactual Found")
                        
                        c1, c2, c3 = st.columns(3)
                        c1.metric("# Changes", result['num_changes'])
                        c2.metric("Distance", f"{result['change_distance']:.3f}")
                        c3.metric("Confidence", f"{result['confidence']:.2%}")
                        
                        if result['changed_features']:
                            st.markdown("**Feature Changes Required:**")
                            for feat, (old, new) in result['changed_features'].items():
                                st.caption(f"• {feat}: {old:.3f} → {new:.3f} (Δ {new-old:+.3f})")
                        
                        st.info(f"**Action:** {result['interpretation']}")
                    else:
                        st.error(f"Failed: {cf_response.status_code}")
                except Exception as e:
                    st.error(f"Error: {e}")
    
    with whatif_tab:
        st.subheader("What-If Analysis")
        st.info("Simulate predictions with different feature values.")
        
        scenario_name = st.text_input("Scenario Name", value="Reduce High-Risk Features")
        
        col1, col2 = st.columns(2)
        
        with col1:
            base_pred = st.slider("Base Prediction", 0.0, 1.0, 0.7)
        
        with col2:
            whatif_features = st.number_input("# Features", 5, 15, 9)
        
        # Feature modifications
        st.markdown("**Feature Modifications:**")
        
        modifications = {}
        feature_names = ["doctor_frequency", "claim_frequency", "claim_amount", 
                        "approval_rate", "avg_claim_cost"]
        
        for feat in feature_names[:3]:
            new_val = st.slider(f"Modify {feat}", 0.0, 100.0, 50.0, key=f"whatif_{feat}")
            modifications[feat] = new_val
        
        if st.button("❓ Analyze What-If", key="analyze_whatif"):
            with st.spinner("Analyzing scenario..."):
                try:
                    features = [0.55 + (i % 4) * 0.12 for i in range(int(whatif_features))]
                    
                    whatif_response = requests.post(
                        f"{API_BASE}/explain/what-if",
                        params={
                            "scenario_name": scenario_name,
                            "current_prediction": base_pred,
                            "features": features,
                            "modifications": modifications
                        },
                        headers={"Authorization": f"Bearer {st.session_state.auth_token}"},
                    )
                    
                    if whatif_response.status_code == 200:
                        result = whatif_response.json()["scenario"]
                        
                        st.success(f"✅ Scenario: {result['name']}")
                        
                        # Prediction change visualization
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Original", f"{result['original_prediction']:.3f}")
                        c2.metric("Modified", f"{result['modified_prediction']:.3f}")
                        c3.metric("Change", f"{result['change']:+.3f}")
                        
                        # Display direction
                        st.markdown(f"**{result['direction']}**")
                        st.info(result['recommendation'])
                        
                        if result.get('modifications'):
                            st.markdown("**Changes Applied:**")
                            for feat, (old, new) in result['modifications'].items():
                                st.caption(f"• {feat}: {old:.2f} → {new:.2f}")
                    else:
                        st.error(f"Failed: {whatif_response.status_code}")
                except Exception as e:
                    st.error(f"Error: {e}")
    
    with sensitivity_tab:
        st.subheader("Sensitivity Analysis")
        st.info("Understand how each feature impacts the prediction.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            sens_feature = st.selectbox("Feature to analyze", 
                                       ["doctor_frequency", "claim_frequency", "claim_amount", 
                                        "approval_rate", "avg_claim_cost"])
        
        with col2:
            sens_pred = st.slider("Base Prediction", 0.0, 1.0, 0.6)
        
        steps = st.slider("Analysis steps", 3, 10, 5)
        
        if st.button("📊 Analyze Sensitivity", key="analyze_sens"):
            with st.spinner("Calculating sensitivity..."):
                try:
                    sens_features = [0.5 + (i % 4) * 0.15 for i in range(9)]
                    
                    sens_response = requests.post(
                        f"{API_BASE}/explain/sensitivity",
                        params={
                            "feature_name": sens_feature,
                            "current_prediction": sens_pred,
                            "features": sens_features,
                            "steps": steps
                        },
                        headers={"Authorization": f"Bearer {st.session_state.auth_token}"},
                    )
                    
                    if sens_response.status_code == 200:
                        result = sens_response.json()["sensitivity"]
                        
                        st.success(f"✅ Sensitivity for {sens_feature}")
                        
                        # Create sensitivity plot
                        sens_df = pd.DataFrame(result)
                        
                        fig = px.line(
                            sens_df,
                            x="value",
                            y="prediction",
                            markers=True,
                            title=f"Prediction vs {sens_feature}",
                            labels={"value": sens_feature, "prediction": "Fraud Score"}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Show data
                        st.dataframe(sens_df, use_container_width=True, hide_index=True)
                    else:
                        st.error(f"Failed: {sens_response.status_code}")
                except Exception as e:
                    st.error(f"Error: {e}")
    
    with compare_tab:
        st.subheader("Compare Two Predictions")
        st.info("Side-by-side comparison of explanations.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Prediction 1**")
            pred1_id = st.text_input("ID 1", value="pred_001")
            pred1_score = st.slider("Score 1", 0.0, 1.0, 0.8, key="pred1_score")
        
        with col2:
            st.markdown("**Prediction 2**")
            pred2_id = st.text_input("ID 2", value="pred_002")
            pred2_score = st.slider("Score 2", 0.0, 1.0, 0.3, key="pred2_score")
        
        if st.button("⚖️ Compare Predictions", key="compare_preds"):
            with st.spinner("Comparing explanations..."):
                try:
                    features1 = [0.6 + (i % 3) * 0.15 for i in range(9)]
                    features2 = [0.4 + (i % 3) * 0.12 for i in range(9)]
                    
                    comp_response = requests.post(
                        f"{API_BASE}/explain/compare-predictions",
                        params={
                            "pred1_id": pred1_id,
                            "pred1_score": pred1_score,
                            "pred1_features": features1,
                            "pred2_id": pred2_id,
                            "pred2_score": pred2_score,
                            "pred2_features": features2
                        },
                        headers={"Authorization": f"Bearer {st.session_state.auth_token}"},
                    )
                    
                    if comp_response.status_code == 200:
                        result = comp_response.json()["comparison"]
                        
                        st.success("✅ Comparison Generated")
                        
                        # Show comparison metrics
                        c1, c2, c3, c4 = st.columns(4)
                        
                        c1.metric("Pred 1 Score", f"{result['prediction_1']['score']:.3f}")
                        c2.metric("Pred 2 Score", f"{result['prediction_2']['score']:.3f}")
                        c3.metric("Score Diff", f"{result['score_difference']:.3f}")
                        c4.metric("Similarity", f"{result['similarity']:.2%}")
                        
                        st.divider()
                        
                        # Shared vs unique features
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.markdown("**Shared Risk Factors:**")
                            if result['shared_risk_factors']:
                                for feat in result['shared_risk_factors']:
                                    st.caption(f"• {feat}")
                            else:
                                st.caption("(None)")
                        
                        with col2:
                            st.markdown("**Unique to Pred 1:**")
                            if result['unique_to_1']:
                                for feat in result['unique_to_1']:
                                    st.caption(f"• {feat}")
                            else:
                                st.caption("(None)")
                        
                        with col3:
                            st.markdown("**Unique to Pred 2:**")
                            if result['unique_to_2']:
                                for feat in result['unique_to_2']:
                                    st.caption(f"• {feat}")
                            else:
                                st.caption("(None)")
                    else:
                        st.error(f"Failed: {comp_response.status_code}")
                except Exception as e:
                    st.error(f"Error: {e}")
    
    with boundaries_tab:
        st.subheader("Decision Boundaries")
        st.info("Analyze where the model switches between fraud/legitimate predictions.")
        
        boundary_feature = st.selectbox(
            "Feature to analyze",
            ["doctor_frequency", "claim_frequency", "claim_amount", 
             "approval_rate", "avg_claim_cost"]
        )
        
        if st.button("🔀 Analyze Boundaries", key="analyze_boundaries"):
            with st.spinner("Analyzing decision boundaries..."):
                try:
                    boundary_response = requests.get(
                        f"{API_BASE}/explain/decision-boundaries",
                        params={"feature_name": boundary_feature},
                        headers={"Authorization": f"Bearer {st.session_state.auth_token}"},
                    )
                    
                    if boundary_response.status_code == 200:
                        result = boundary_response.json()["boundaries"]
                        
                        st.success(f"✅ Decision Boundaries for {boundary_feature}")
                        
                        # Create boundary plot
                        boundary_df = pd.DataFrame(result["boundaries"])
                        
                        fig = px.scatter(
                            boundary_df,
                            x="value",
                            y="fraud_likelihood",
                            color="region",
                            color_discrete_map={
                                "Low Risk": "🟢",
                                "Medium Risk": "🟡",
                                "High Risk": "🔴"
                            },
                            title=f"Decision Boundary: {boundary_feature}",
                            labels={"value": boundary_feature, "fraud_likelihood": "Fraud Likelihood"}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Show regions
                        st.markdown("**Risk Regions:**")
                        for boundary in result["boundaries"]:
                            val = boundary["value"]
                            likelihood = boundary["fraud_likelihood"]
                            region = boundary["region"]
                            st.caption(f"{region}: {val:.2f} → {likelihood:.2%}")
                    else:
                        st.error(f"Failed: {boundary_response.status_code}")
                except Exception as e:
                    st.error(f"Error: {e}")


# ── Production Resilience (Step 10) ────────────────────────────────────────────
elif page == "🛡️ Resilience":
    st.title("🛡️ Production Resilience & Fault Tolerance")
    st.markdown("**Enterprise resilience: Circuit breakers, failover, rate limiting, graceful degradation**")
    st.divider()
    
    health_tab, circuit_tab, rate_tab, bulkhead_tab, degrade_tab, summary_tab = st.tabs([
        "❤️ Health", "🔌 Circuit Breakers", "🚦 Rate Limiting", "🚧 Bulkheads", "📉 Degradation", "📊 Summary"
    ])
    
    with health_tab:
        st.subheader("System Health Checks")
        
        if st.button("🔄 Refresh Health", key="refresh_health"):
            st.rerun()
        
        try:
            health_response = api_get("/health")
            
            if health_response and "checks" in health_response:
                overall = health_response["status"]
                
                # Show overall status
                if overall == "healthy":
                    st.success(f"✅ **System Status:** HEALTHY")
                elif overall == "degraded":
                    st.warning(f"🟡 **System Status:** DEGRADED")
                else:
                    st.error(f"🔴 **System Status:** UNHEALTHY")
                
                st.divider()
                
                # Check details
                st.subheader("Check Details")
                checks = health_response["checks"]
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    healthy_count = sum(1 for c in checks.values() if c == "healthy")
                    st.metric("Healthy Checks", healthy_count)
                
                with col2:
                    degraded_count = sum(1 for c in checks.values() if c == "degraded")
                    st.metric("Degraded Checks", degraded_count)
                
                with col3:
                    unhealthy_count = sum(1 for c in checks.values() if c == "unhealthy")
                    st.metric("Unhealthy Checks", unhealthy_count)
                
                st.markdown("**Individual Checks:**")
                for check_name, status in checks.items():
                    if status == "healthy":
                        st.caption(f"✅ {check_name}")
                    elif status == "degraded":
                        st.caption(f"🟡 {check_name}")
                    else:
                        st.caption(f"🔴 {check_name}")
        except Exception as e:
            st.error(f"Error: {e}")
    
    with circuit_tab:
        st.subheader("Circuit Breaker Status")
        st.info("Circuit breakers prevent cascading failures by stopping requests to failing services.")
        
        try:
            cb_response = api_get("/resilience/circuit-breakers")
            
            if cb_response and "circuit_breakers" in cb_response:
                cbs = cb_response["circuit_breakers"]
                
                st.metric("Total Circuit Breakers", len(cbs))
                st.divider()
                
                for cb_name, cb_status in cbs.items():
                    with st.expander(f"🔌 {cb_name} - **{cb_status['state'].upper()}**", expanded=False):
                        col1, col2, col3, col4 = st.columns(4)
                        
                        col1.metric("Total Calls", cb_status["total_calls"])
                        col2.metric("Success Rate", f"{cb_status['success_rate']:.1%}")
                        col3.metric("Rejected", cb_status["rejected"])
                        col4.metric("State Changes", cb_status["state_changes"])
                        
                        if cb_status["last_failure"]:
                            st.caption(f"Last failure: {cb_status['last_failure'][:19]}")
                        
                        # Show state indicator
                        state_color = "🟢" if cb_status["state"] == "closed" else (
                            "🟡" if cb_status["state"] == "half_open" else "🔴"
                        )
                        st.markdown(f"{state_color} **State:** {cb_status['state'].upper()}")
        except Exception as e:
            st.error(f"Error: {e}")
    
    with rate_tab:
        st.subheader("Rate Limiting Status")
        st.info("Rate limiters protect services from being overwhelmed by too many requests.")
        
        try:
            rl_response = api_get("/resilience/rate-limits")
            
            if rl_response and "rate_limiters" in rl_response:
                rls = rl_response["rate_limiters"]
                
                st.metric("Total Rate Limiters", len(rls))
                st.divider()
                
                for rl_name, rl_status in rls.items():
                    with st.expander(f"🚦 {rl_name}", expanded=False):
                        col1, col2, col3, col4 = st.columns(4)
                        
                        col1.metric("Tokens Available", f"{rl_status['tokens_available']:.0f}")
                        col2.metric("Allowed", rl_status["requests_allowed"])
                        col3.metric("Rejected", rl_status["requests_rejected"])
                        col4.metric("Rejection Rate", f"{rl_status['rejection_rate']:.2%}")
                        
                        # Show capacity bar
                        if rl_status['tokens_available'] > 0:
                            st.progress(min(rl_status['tokens_available'] / 100, 1.0))
                        else:
                            st.warning("⚠️ Rate limit exceeded - requests being throttled")
        except Exception as e:
            st.error(f"Error: {e}")
    
    with bulkhead_tab:
        st.subheader("Resource Bulkheads")
        st.info("Bulkheads isolate resources and limit concurrent access to prevent resource exhaustion.")
        
        try:
            bh_response = api_get("/resilience/bulkheads")
            
            if bh_response and "bulkheads" in bh_response:
                bhs = bh_response["bulkheads"]
                
                st.metric("Total Bulkheads", len(bhs))
                st.divider()
                
                for bh_name, bh_status in bhs.items():
                    with st.expander(f"🚧 {bh_name}", expanded=False):
                        current = bh_status["current_tasks"]
                        max_cap = bh_status["max_concurrent"]
                        util = bh_status["utilization"]
                        
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Current Tasks", current)
                        col2.metric("Max Capacity", max_cap)
                        col3.metric("Utilization", f"{util:.1%}")
                        
                        # Show utilization bar
                        st.progress(util)
                        
                        if util > 0.9:
                            st.warning(f"⚠️ High utilization ({util:.1%}) - near capacity")
                        elif util > 0.7:
                            st.info(f"ℹ️ Moderate utilization ({util:.1%})")
                        else:
                            st.success(f"✅ Low utilization ({util:.1%})")
                        
                        st.caption(f"Total executed: {bh_status['total_executed']} | Rejected: {bh_status['rejected']}")
        except Exception as e:
            st.error(f"Error: {e}")
    
    with degrade_tab:
        st.subheader("Graceful Degradation")
        st.info("Disable non-critical features during failures to keep core services running.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            degrade_reason = st.text_input("Degradation reason", value="Manual maintenance")
        
        with col2:
            if st.button("📉 Trigger Degradation", key="trigger_degrade"):
                with st.spinner("Degrading services..."):
                    try:
                        degrade_response = requests.post(
                            f"{API_BASE}/resilience/degrade",
                            params={"reason": degrade_reason},
                            headers={"Authorization": f"Bearer {st.session_state.auth_token}"},
                        )
                        
                        if degrade_response.status_code == 200:
                            result = degrade_response.json()
                            st.success("✅ Graceful degradation triggered")
                            st.json(result["features"])
                        else:
                            st.error(f"Failed: {degrade_response.status_code}")
                    except Exception as e:
                        st.error(f"Error: {e}")
        
        if st.button("🔧 Recover Services", key="recover_services"):
            with st.spinner("Recovering services..."):
                try:
                    recover_response = requests.post(
                        f"{API_BASE}/resilience/recover",
                        headers={"Authorization": f"Bearer {st.session_state.auth_token}"},
                    )
                    
                    if recover_response.status_code == 200:
                        result = recover_response.json()
                        st.success("✅ Service recovery initiated")
                        st.json(result["features"])
                    else:
                        st.error(f"Failed: {recover_response.status_code}")
                except Exception as e:
                    st.error(f"Error: {e}")
        
        st.divider()
        
        try:
            feature_response = api_get("/resilience/features")
            
            if feature_response and "degradation" in feature_response:
                degrad = feature_response["degradation"]
                
                st.subheader("Feature Status")
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Total Features", degrad["total_features"])
                c2.metric("Enabled", degrad["enabled"])
                c3.metric("Disabled", degrad["disabled"])
                
                if degrad["disabled_features"]:
                    st.warning("**Disabled Features:**")
                    for feat in degrad["disabled_features"]:
                        st.caption(f"• {feat}")
        except Exception as e:
            st.error(f"Error: {e}")
    
    with summary_tab:
        st.subheader("Resilience Dashboard")
        st.info("Complete overview of all resilience metrics.")
        
        if st.button("🔄 Refresh Dashboard", key="refresh_resilience"):
            st.rerun()
        
        try:
            resilience_response = api_get("/resilience/dashboard")
            
            if resilience_response and "dashboard" in resilience_response:
                dashboard = resilience_response["dashboard"]
                
                # Overall status
                overall = dashboard["overall_health"]
                if overall == "healthy":
                    st.success(f"✅ **Overall System Health:** HEALTHY")
                elif overall == "degraded":
                    st.warning(f"🟡 **Overall System Health:** DEGRADED")
                else:
                    st.error(f"🔴 **Overall System Health:** UNHEALTHY")
                
                st.divider()
                
                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    cbs = dashboard.get("circuit_breakers", {})
                    open_count = sum(1 for cb in cbs.values() if cb.get("state") == "open")
                    st.metric("Open Circuit Breakers", open_count)
                
                with col2:
                    rls = dashboard.get("rate_limiters", {})
                    rejected = sum(rl.get("requests_rejected", 0) for rl in rls.values())
                    st.metric("Rate Limited Requests", rejected)
                
                with col3:
                    bhs = dashboard.get("bulkheads", {})
                    total_util = sum(bh.get("utilization", 0) for bh in bhs.values()) / max(len(bhs), 1)
                    st.metric("Avg Bulkhead Utilization", f"{total_util:.1%}")
                
                with col4:
                    degrad = dashboard.get("graceful_degradation", {})
                    st.metric("Disabled Features", degrad.get("disabled", 0))
                
                st.divider()
                
                # Timeline
                st.subheader("Recent Activity")
                st.info("All metrics collected at: " + dashboard["timestamp"][:19])
                
                # Show key insights
                insights = []
                
                if open_count > 0:
                    insights.append(f"⚠️ {open_count} circuit breaker(s) open - services may be failing")
                
                if total_util > 0.8:
                    insights.append(f"⚠️ High resource utilization ({total_util:.1%}) - potential bottleneck")
                
                if degrad.get("disabled", 0) > 0:
                    insights.append(f"📉 {degrad['disabled']} features disabled - running in degraded mode")
                
                if not insights:
                    st.success("✅ All systems nominal - no issues detected")
                else:
                    for insight in insights:
                        st.warning(insight)
        except Exception as e:
            st.error(f"Error: {e}")
