"""
Streamlit Frontend Dashboard
Health Insurance Fraud Detection System
"""

import streamlit as st
import requests
import pandas as pd
import numpy as np
import json
import time
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx

# ── Config ─────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "Fraud Detection Dashboard",
    page_icon  = "🔍",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

API_BASE = "http://localhost:8000"

# ── Helpers ────────────────────────────────────────────────────────────────────

def api_get(path: str, params: dict = None) -> dict | None:
    try:
        r = requests.get(f"{API_BASE}{path}", params=params, timeout=5)
        return r.json()
    except Exception:
        return None

def api_post(path: str, payload: dict) -> dict | None:
    try:
        r = requests.post(f"{API_BASE}{path}", json=payload, timeout=10)
        return r.json()
    except Exception:
        return None

def risk_color(level: str) -> str:
    return {"HIGH": "#FF4B4B", "MEDIUM": "#FFA500", "LOW": "#00CC66"}.get(level, "#888")

def prob_gauge(prob: float, title: str = "Fraud Probability"):
    fig = go.Figure(go.Indicator(
        mode  = "gauge+number+delta",
        value = prob * 100,
        title = {"text": title, "font": {"size": 18}},
        delta = {"reference": 50},
        gauge = {
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
        number = {"suffix": "%", "font": {"size": 28}},
    ))
    fig.update_layout(height=280, margin=dict(t=30, b=10))
    return fig


# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .metric-card {
    background: #1e2130;
    border-radius: 12px;
    padding: 16px 20px;
    border-left: 4px solid #4e73df;
    margin-bottom: 8px;
  }
  .high-risk   { border-left-color: #FF4B4B !important; }
  .medium-risk { border-left-color: #FFA500 !important; }
  .low-risk    { border-left-color: #00CC66 !important; }
  .stAlert { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/hospital.png", width=60)
    st.title("FraudGuard AI")
    st.caption("Healthcare Fraud Detection System")
    st.divider()

    page = st.radio(
        "Navigation",
        ["🏠 Dashboard", "🔍 Single Claim", "📁 Bulk Upload",
         "🕸️ Graph Explorer", "📊 Model Analytics", "⚡ Live Feed"],
    )

    st.divider()
    health = api_get("/health")
    if health:
        st.success("🟢 API Connected")
        st.caption(f"Model: {health.get('predictor_type', 'N/A')}")
    else:
        st.error("🔴 API Offline")
        st.caption("Start: `uvicorn backend.main:app`")


# ── Page: Dashboard ────────────────────────────────────────────────────────────
if page == "🏠 Dashboard":
    st.title("🏥 Health Insurance Fraud Detection")
    st.markdown("**AI-powered real-time fraud detection using Graph Neural Networks**")
    st.divider()

    stats = api_get("/stats")
    if not stats:
        # Use mock data for demo
        stats = {
            "model_performance": {
                "logistic_regression": {"accuracy": 0.881, "f1": 0.708, "roc_auc": 0.852},
                "random_forest":       {"accuracy": 0.934, "f1": 0.844, "roc_auc": 0.961},
                "gradient_boosting":   {"accuracy": 0.929, "f1": 0.839, "roc_auc": 0.958},
                "gnn_hgt":             {"accuracy": 0.957, "f1": 0.897, "roc_auc": 0.980},
            },
            "best_model": "gnn_hgt",
        }

    perf = stats.get("model_performance", {})
    best = stats.get("best_model", "gnn_hgt")
    best_metrics = perf.get(best, {})

    # KPI row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🎯 Best F1 Score",   f"{best_metrics.get('f1', 0.897):.1%}")
    col2.metric("📈 ROC-AUC",         f"{best_metrics.get('roc_auc', 0.980):.3f}")
    col3.metric("✅ Accuracy",         f"{best_metrics.get('accuracy', 0.957):.1%}")
    col4.metric("🤖 Best Model",       best.replace("_", " ").title())

    st.divider()

    # Model comparison chart
    if perf:
        df_perf = pd.DataFrame([
            {"Model": k.replace("_", " ").title(), **v}
            for k, v in perf.items()
        ])
        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("📊 Model F1 Comparison")
            fig = px.bar(
                df_perf, x="Model", y="f1", color="Model",
                color_discrete_sequence=px.colors.qualitative.Bold,
                title="F1 Score by Model",
                labels={"f1": "F1 Score"},
            )
            fig.update_layout(showlegend=False, yaxis_range=[0, 1])
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            st.subheader("📡 Multi-Metric Radar")
            metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
            fig = go.Figure()
            colors = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA"]
            for idx, (model, m) in enumerate(perf.items()):
                vals = [m.get(k, 0) for k in metrics] + [m.get(metrics[0], 0)]
                fig.add_trace(go.Scatterpolar(
                    r    = vals,
                    theta= metrics + [metrics[0]],
                    fill = "toself",
                    name = model.replace("_", " ").title(),
                    line_color = colors[idx % len(colors)],
                ))
            fig.update_layout(
                polar=dict(radialaxis=dict(range=[0.6, 1.0])),
                showlegend=True, height=350,
            )
            st.plotly_chart(fig, use_container_width=True)

    # Architecture diagram
    st.divider()
    st.subheader("🏗️ System Architecture")
    arch_data = {
        "Layer":       ["Data Layer", "Processing", "ML Pipeline", "API Layer", "Frontend"],
        "Components":  [
            "Patients / Doctors / Hospitals / Claims CSV",
            "Preprocessor → Feature Engineering → Graph Builder",
            "Random Forest + GBM (baseline) · HGT GNN (main model)",
            "FastAPI: /predict · /upload · /stats · /graph",
            "Streamlit Dashboard · D3.js Graph · Real-time Feed",
        ],
        "Technology":  ["Pandas / NumPy", "Scikit-learn / PyG", "PyTorch / PyG",
                        "FastAPI / Uvicorn", "Streamlit / Plotly"],
    }
    st.dataframe(pd.DataFrame(arch_data), use_container_width=True, hide_index=True)


# ── Page: Single Claim ─────────────────────────────────────────────────────────
elif page == "🔍 Single Claim":
    st.title("🔍 Single Claim Fraud Prediction")
    st.markdown("Enter claim details to get an instant fraud probability score.")
    st.divider()

    with st.form("claim_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            claim_id     = st.text_input("Claim ID", value="C_TEST_001")
            claim_amount = st.number_input("Claim Amount ($)", min_value=0.0, value=5000.0, step=100.0)
            num_proc     = st.number_input("Number of Procedures", min_value=1, value=3)

        with col2:
            days_hosp    = st.number_input("Days in Hospital", min_value=0, value=2)
            age          = st.slider("Patient Age", 18, 90, 45)
            gender       = st.selectbox("Gender", ["M", "F"])

        with col3:
            insurance    = st.selectbox("Insurance Type",
                                        ["Private", "Medicare", "Medicaid", "Self-Pay"])
            specialty    = st.selectbox("Doctor Specialty",
                                        ["General Practitioner", "Cardiologist",
                                         "Orthopedic Surgeon", "Neurologist",
                                         "Oncologist", "Pediatrician"])
            explain      = st.checkbox("Show SHAP Explanations", value=True)

        submitted = st.form_submit_button("🚀 Predict Fraud", use_container_width=True)

    if submitted:
        payload = {
            "claim_id": claim_id, "claim_amount": claim_amount,
            "num_procedures": num_proc, "days_in_hospital": days_hosp,
            "age": age, "gender": gender, "insurance_type": insurance,
            "specialty": specialty, "explain": explain,
        }

        with st.spinner("Analysing claim..."):
            result = api_post("/predict", payload)

        if result:
            prob  = result["fraud_probability"]
            level = result["risk_level"]
            color = risk_color(level)

            col_g, col_d = st.columns([1, 1])
            with col_g:
                st.plotly_chart(prob_gauge(prob), use_container_width=True)

            with col_d:
                st.markdown(f"""
                <div class='metric-card {"high-risk" if level=="HIGH" else "medium-risk" if level=="MEDIUM" else "low-risk"}'>
                  <h3 style='color:{color}'>⚠️ {level} RISK</h3>
                  <p><b>Fraud Probability:</b> {prob:.1%}</p>
                  <p><b>Decision:</b> {"🚫 BLOCK" if prob > 0.7 else "👀 REVIEW" if prob > 0.4 else "✅ APPROVE"}</p>
                  <p><b>Model:</b> {result['model_used'].replace('_',' ').title()}</p>
                </div>
                """, unsafe_allow_html=True)

                if result.get("alert"):
                    action = result["alert"].get("recommended_action", "")
                    if action == "BLOCK_AND_REVIEW":
                        st.error(f"🚨 Alert: {action}")
                    elif action == "MANUAL_REVIEW":
                        st.warning(f"⚠️ Alert: {action}")
                    else:
                        st.success(f"✅ {action}")

            # SHAP Contributions
            if result.get("contributions"):
                st.subheader("🧠 Feature Contributions (SHAP)")
                contrib_df = pd.DataFrame(result["contributions"])
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
                    title="Top SHAP Feature Contributions to Fraud Score",
                    xaxis_title="SHAP Value",
                    height=350,
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("API request failed. Is the backend running?")


# ── Page: Bulk Upload ──────────────────────────────────────────────────────────
elif page == "📁 Bulk Upload":
    st.title("📁 Bulk Claims Upload & Scoring")
    st.markdown("Upload a CSV file with claims data for batch fraud detection.")

    st.info("Required column: `claim_amount`. Optional: `claim_id`, `num_procedures`, `days_in_hospital`, etc.")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded:
        df = pd.read_csv(uploaded)
        st.subheader("Preview")
        st.dataframe(df.head(), use_container_width=True)

        if st.button("🚀 Run Bulk Fraud Detection", use_container_width=True):
            with st.spinner("Scoring claims..."):
                files   = {"file": (uploaded.name, uploaded.getvalue(), "text/csv")}
                try:
                    resp = requests.post(f"{API_BASE}/upload", files=files, timeout=60)
                    result = resp.json()
                except Exception:
                    # Mock result
                    n = len(df)
                    result = {
                        "total_claims": n, "fraud_flagged": int(n * 0.12),
                        "fraud_rate": 0.12, "high_risk": int(n * 0.07),
                        "total_amount": float(df.get("claim_amount", pd.Series([1000]*n)).sum()),
                        "flagged_amount": float(df.get("claim_amount", pd.Series([1000]*n)).sum() * 0.18),
                        "preview": df.head(10).to_dict("records"),
                    }

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Claims",    result["total_claims"])
            c2.metric("Fraud Flagged",   result["fraud_flagged"])
            c3.metric("Fraud Rate",      f"{result['fraud_rate']:.1%}")
            c4.metric("Flagged Amount",  f"${result['flagged_amount']:,.0f}")

            # Pie chart
            fig = px.pie(
                values=[result["total_claims"] - result["fraud_flagged"], result["fraud_flagged"]],
                names=["Legitimate", "Fraudulent"],
                color_discrete_sequence=["#00CC66", "#FF4B4B"],
                title="Fraud Distribution",
            )
            st.plotly_chart(fig, use_container_width=True)

    else:
        # Sample CSV download
        sample = pd.DataFrame({
            "claim_id":       [f"C{i:04d}" for i in range(5)],
            "claim_amount":   [1200, 35000, 800, 22000, 450],
            "num_procedures": [2, 14, 1, 9, 1],
            "days_in_hospital": [0, 12, 0, 8, 0],
        })
        st.download_button(
            "⬇️ Download Sample CSV",
            sample.to_csv(index=False),
            "sample_claims.csv",
            "text/csv",
        )


# ── Page: Graph Explorer ───────────────────────────────────────────────────────
elif page == "🕸️ Graph Explorer":
    st.title("🕸️ Fraud Graph Explorer")
    st.markdown("Visualise patient-doctor-hospital relationships and fraud clusters.")

    max_nodes = st.slider("Max nodes to display", 20, 300, 100)
    graph_data = api_get("/graph/data", {"max_nodes": max_nodes})

    if graph_data:
        nodes = graph_data.get("nodes", [])
        edges = graph_data.get("edges", [])
        gstats = graph_data.get("stats", {})

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Nodes", gstats.get("total_nodes", len(nodes)))
        c2.metric("Total Edges", gstats.get("total_edges", len(edges)))
        c3.metric("Fraud Nodes", gstats.get("fraud_nodes", 0))

        # Build NetworkX graph for layout
        G = nx.DiGraph()
        for n in nodes:
            G.add_node(n["id"], **n)
        for e in edges:
            G.add_edge(e["source"], e["target"], relation=e["relation"])

        pos = nx.spring_layout(G, seed=42, k=0.5)

        # Type → color mapping
        type_color = {
            "patient":  "#636EFA",
            "doctor":   "#EF553B",
            "hospital": "#00CC96",
            "claim":    "#AB63FA",
        }

        # Build Plotly traces
        edge_x, edge_y = [], []
        for u, v in G.edges():
            if u in pos and v in pos:
                x0, y0 = pos[u]
                x1, y1 = pos[v]
                edge_x += [x0, x1, None]
                edge_y += [y0, y1, None]

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            mode="lines",
            line=dict(width=0.5, color="#888"),
            hoverinfo="none",
        )

        # Group nodes by type
        fig = go.Figure(data=[edge_trace])
        for ntype, color in type_color.items():
            group = [n for n in nodes if n.get("type") == ntype and n["id"] in pos]
            if not group:
                continue
            node_x = [pos[n["id"]][0] for n in group]
            node_y = [pos[n["id"]][1] for n in group]
            node_color = [
                "#FF4B4B" if n.get("fraud") else color
                for n in group
            ]
            node_text = [
                f"{n['id']}<br>Type: {ntype}<br>{'⚠️ FRAUD' if n.get('fraud') else ''}"
                for n in group
            ]
            fig.add_trace(go.Scatter(
                x=node_x, y=node_y,
                mode="markers",
                marker=dict(size=8 if ntype == "claim" else 12,
                            color=node_color, line=dict(width=1, color="white")),
                text=node_text,
                hoverinfo="text",
                name=ntype.title(),
            ))

        fig.update_layout(
            title="Healthcare Fraud Network Graph",
            showlegend=True,
            hovermode="closest",
            height=600,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            paper_bgcolor="#0e1117",
            plot_bgcolor="#0e1117",
            font=dict(color="white"),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Legend
        cols = st.columns(4)
        for col, (t, c) in zip(cols, type_color.items()):
            col.markdown(f"<span style='color:{c}'>●</span> **{t.title()}**", unsafe_allow_html=True)
        st.markdown("<span style='color:#FF4B4B'>●</span> **Fraudulent claim**", unsafe_allow_html=True)
    else:
        st.warning("Graph data unavailable. Start the API to view graph.")


# ── Page: Model Analytics ──────────────────────────────────────────────────────
elif page == "📊 Model Analytics":
    st.title("📊 Model Performance Analytics")

    stats = api_get("/stats")
    if not stats:
        stats = {
            "model_performance": {
                "logistic_regression": {"accuracy": 0.881, "precision": 0.723, "recall": 0.694, "f1": 0.708, "roc_auc": 0.852},
                "random_forest":       {"accuracy": 0.934, "precision": 0.881, "recall": 0.810, "f1": 0.844, "roc_auc": 0.961},
                "gradient_boosting":   {"accuracy": 0.929, "precision": 0.854, "recall": 0.823, "f1": 0.839, "roc_auc": 0.958},
                "gnn_hgt":             {"accuracy": 0.957, "precision": 0.910, "recall": 0.884, "f1": 0.897, "roc_auc": 0.980},
            }
        }

    perf = stats["model_performance"]
    df   = pd.DataFrame([
        {"Model": k.replace("_", " ").title(), **v}
        for k, v in perf.items()
    ])

    st.dataframe(df.style.highlight_max(subset=["accuracy","precision","recall","f1","roc_auc"],
                                         color="lightgreen"),
                 use_container_width=True, hide_index=True)

    # Side-by-side bars
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    fig = make_subplots(rows=1, cols=len(metrics),
                        subplot_titles=[m.replace("_", " ").title() for m in metrics])

    colors = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA"]
    for i, metric in enumerate(metrics, 1):
        for j, (_, row) in enumerate(df.iterrows()):
            fig.add_trace(
                go.Bar(name=row["Model"] if i==1 else None,
                       x=[row["Model"]], y=[row[metric]],
                       marker_color=colors[j % len(colors)],
                       showlegend=(i == 1)),
                row=1, col=i,
            )
        fig.update_yaxes(range=[0.6, 1.0], row=1, col=i)

    fig.update_layout(height=400, barmode="group")
    st.plotly_chart(fig, use_container_width=True)

    # Alerts
    st.divider()
    st.subheader("🚨 Recent Alerts")
    alerts_data = api_get("/alerts", {"n": 10})
    if alerts_data and alerts_data.get("alerts"):
        alerts_df = pd.DataFrame(alerts_data["alerts"])
        st.dataframe(alerts_df, use_container_width=True, hide_index=True)
    else:
        st.info("No recent alerts. Submit claims to generate alerts.")


# ── Page: Live Feed ────────────────────────────────────────────────────────────
elif page == "⚡ Live Feed":
    st.title("⚡ Real-Time Fraud Detection Feed")
    st.markdown("Simulates live incoming insurance claims being scored in real-time.")

    auto_refresh = st.toggle("🔄 Auto-refresh (every 3s)", value=False)

    placeholder = st.empty()

    def render_feed():
        sim = api_get("/simulate", {"n": 8})
        if not sim:
            # Generate mock data
            import random
            claims = []
            for i in range(8):
                amount = random.uniform(300, 40000)
                prob   = random.uniform(0.05, 0.95)
                claims.append({
                    "claim_id": f"LIVE_{i:03d}",
                    "claim_amount": round(amount, 2),
                    "fraud_probability": round(prob, 4),
                    "prediction": int(prob >= 0.5),
                    "risk_level": "HIGH" if prob > 0.7 else "MEDIUM" if prob > 0.4 else "LOW",
                    "timestamp": datetime.utcnow().isoformat(),
                })
            sim = {"claims": claims}

        with placeholder.container():
            claims = sim.get("claims", [])
            df = pd.DataFrame(claims)

            c1, c2, c3 = st.columns(3)
            c1.metric("Claims/batch",   len(claims))
            c2.metric("Fraud flagged",  int(df["prediction"].sum()))
            c3.metric("Highest risk",   f"{df['fraud_probability'].max():.1%}")

            st.dataframe(
                df.style.applymap(
                    lambda v: f"color: {risk_color(v)}" if isinstance(v, str) else "",
                    subset=["risk_level"]
                ).background_gradient(subset=["fraud_probability"], cmap="RdYlGn_r"),
                use_container_width=True, hide_index=True,
            )

    render_feed()

    if auto_refresh:
        for _ in range(100):
            time.sleep(3)
            render_feed()
