# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Fraud-Guard: Complete Full-Stack Deployment
# Runs both FastAPI Backend and Streamlit Frontend in ONE Container
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

FROM python:3.11-slim

LABEL maintainer="Fraud-Guard Team"
LABEL description="Healthcare Insurance Fraud Detection - Full Stack Deployment"

# ── System Dependencies ────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Copy Requirements & Install Dependencies ────────────────────────────────
COPY requirements.txt .

# Install PyTorch CPU + PyTorch Geometric
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir torch==2.2.0 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir torch-scatter torch-sparse torch-geometric \
        --find-links https://data.pyg.org/whl/torch-2.2.0+cpu.html && \
    pip install --no-cache-dir -r requirements.txt

# ── Copy Entire Project ────────────────────────────────────────────────────
COPY . .

# ── Create Required Directories ─────────────────────────────────────────────
RUN mkdir -p \
    data/raw \
    data/processed \
    models/baseline \
    models/gnn \
    logs \
    training/logs \
    backend/database

# ── Streamlit Configuration ────────────────────────────────────────────────
RUN mkdir -p ~/.streamlit && \
    echo "[server]" > ~/.streamlit/config.toml && \
    echo "headless = true" >> ~/.streamlit/config.toml && \
    echo "port = 8501" >> ~/.streamlit/config.toml && \
    echo "enableXsrfProtection = false" >> ~/.streamlit/config.toml && \
    echo "enableCORS = false" >> ~/.streamlit/config.toml && \
    echo "" >> ~/.streamlit/config.toml && \
    echo "[logger]" >> ~/.streamlit/config.toml && \
    echo "level = \"info\"" >> ~/.streamlit/config.toml

# ── Expose Ports ────────────────────────────────────────────────────────────
# 8000 = FastAPI Backend
# 8501 = Streamlit Frontend
EXPOSE 8000 8501

# ── Health Check ────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# ── Create Startup Script ───────────────────────────────────────────────────
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
echo "========================================"\n\
echo "🚀 Fraud-Guard: Full-Stack Startup"\n\
echo "========================================"\n\
echo ""\n\
echo "Starting FastAPI Backend on :8000..."\n\
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --log-level info &\n\
BACKEND_PID=$!\n\
echo "✓ Backend PID: $BACKEND_PID"\n\
\n\
# Wait for backend to be ready\n\
sleep 3\n\
\n\
echo "Starting Streamlit Frontend on :8501..."\n\
streamlit run frontend/app.py --server.port 8501 --server.address 0.0.0.0 --logger.level=info &\n\
FRONTEND_PID=$!\n\
echo "✓ Frontend PID: $FRONTEND_PID"\n\
\n\
echo ""\n\
echo "========================================"\n\
echo "✅ Fraud-Guard is Running!"\n\
echo "========================================"\n\
echo "📊 Frontend (Streamlit):  http://localhost:8501"\n\
echo "🔌 Backend API (FastAPI):  http://localhost:8000"\n\
echo "📖 API Docs (Swagger):    http://localhost:8000/docs"\n\
echo "📚 API Docs (ReDoc):      http://localhost:8000/redoc"\n\
echo "❤️ Health Check:          http://localhost:8000/health"\n\
echo ""\n\
echo "========================================"\n\
echo "Waiting for services..."\n\
echo "========================================"\n\
echo ""\n\
\n\
wait\n\
' > /app/start.sh && chmod +x /app/start.sh

# ── Run Both Services ───────────────────────────────────────────────────────
CMD ["/app/start.sh"]
