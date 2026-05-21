# 🚀 Deployment Guide — Healthcare Fraud Detection System

Complete step-by-step instructions to deploy the fraud detection system locally, on cloud, or production environment.

---

## Table of Contents

1. [Quick Start (Local)](#quick-start-local)
2. [Prerequisites](#prerequisites)
3. [Configuration](#configuration)
4. [Docker Deployment](#docker-deployment)
5. [Database Setup](#database-setup)
6. [Model Training](#model-training)
7. [Accessing the Application](#accessing-the-application)
8. [Monitoring & Health Checks](#monitoring--health-checks)
9. [Production Hardening](#production-hardening)
10. [Troubleshooting](#troubleshooting)
11. [Cloud Deployment (AWS/GCP/Azure)](#cloud-deployment)

---

## Quick Start (Local)

**TL;DR — Get running in 60 seconds:**

```bash
# 1. Clone and navigate
cd health-fraud-detection

# 2. Copy environment file
cp .env.example .env  # or create one with defaults

# 3. Start all services (API + Frontend + DB)
docker-compose -f docker/docker-compose.yml up -d

# 4. Wait for services to be ready
sleep 5

# 5. Access applications
# API:       http://localhost:8000
# Frontend:  http://localhost:8501
# Health:    http://localhost:8000/health
```

---

## Prerequisites

### **Required Software**

| Component | Version | Purpose |
|-----------|---------|---------|
| Docker | 20.10+ | Container runtime |
| Docker Compose | 2.0+ | Multi-container orchestration |
| Python | 3.11+ | Local development (optional) |
| Git | 2.30+ | Version control |
| curl/wget | Any | Health checks |

### **System Requirements**

| Resource | Minimum | Recommended | Production |
|----------|---------|-------------|-----------|
| CPU | 2 cores | 4 cores | 8+ cores |
| RAM | 4 GB | 8 GB | 16+ GB |
| Disk | 10 GB | 20 GB | 50+ GB |
| Network | 100 Mbps | 1 Gbps | 1+ Gbps |

### **Installation**

**macOS/Linux:**
```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Add user to docker group (avoid sudo)
sudo usermod -aG docker $USER
newgrp docker
```

**Windows (PowerShell as Admin):**
```powershell
# Using Windows Package Manager
winget install Docker.DockerDesktop

# Or WSL2 + Docker Desktop
# Download from: https://www.docker.com/products/docker-desktop
```

**Verify Installation:**
```bash
docker --version          # Docker 20.10+
docker-compose --version  # Docker Compose 2.0+
docker run hello-world    # Should print "Hello from Docker!"
```

---

## Configuration

### **1. Environment Variables**

Create `.env` file in project root:

```bash
# === Backend Configuration ===
API_HOST=0.0.0.0
API_PORT=8000
API_LOG_LEVEL=info
API_WORKERS=4

# === Database ===
POSTGRES_USER=fraud_admin
POSTGRES_PASSWORD=secure_password_here  # CHANGE THIS!
POSTGRES_DB=fraud_detection
POSTGRES_HOST=postgres
POSTGRES_PORT=5432

# === JWT & Security ===
JWT_SECRET_KEY=your-secret-key-change-this-in-production
JWT_ALGORITHM=HS256
JWT_EXPIRATION_HOURS=24
API_KEY=test_key_123  # For local testing

# === ML Models ===
MODEL_PATH=/app/models
DATA_PATH=/app/data
GNN_WEIGHTS_PATH=/app/models/gnn/best_model.pt

# === Features ===
ENABLE_COMPLIANCE=true
ENABLE_MONITORING=true
ENABLE_HEALTH_CHECKS=true
ENABLE_CIRCUIT_BREAKER=true

# === Rate Limiting ===
RATE_LIMIT_PREDICTIONS_PER_SEC=100
RATE_LIMIT_BULK_UPLOADS_PER_MIN=10

# === Logging ===
LOG_FORMAT=json
LOG_LEVEL=info
SENTRY_DSN=  # Optional: error tracking
```

### **2. Docker Network**

Services communicate via Docker network (already configured in docker-compose.yml):

```yaml
networks:
  fraud-network:
    driver: bridge
```

Service DNS names inside container:
- API: `http://api:8000`
- Frontend: `http://frontend:8501`
- Database: `postgres:5432`

---

## Docker Deployment

### **Option 1: One-Command Deployment**

```bash
# Start all services
cd docker
docker-compose up -d

# View logs in real-time
docker-compose logs -f api frontend postgres

# Stop all services
docker-compose down

# Stop and remove volumes (careful: deletes database!)
docker-compose down -v
```

### **Option 2: Step-by-Step Deployment**

#### **Step 1: Build Images**
```bash
# Build API image
docker build -f docker/Dockerfile.api -t fraud-api:latest .

# Build Frontend image
docker build -f docker/Dockerfile.frontend -t fraud-frontend:latest .

# Verify images
docker images | grep fraud-
```

#### **Step 2: Start Database**
```bash
# Start PostgreSQL only
docker-compose -f docker/docker-compose.yml up -d postgres

# Wait for database to be ready (30-60 seconds)
docker-compose -f docker/docker-compose.yml logs postgres | grep "database system is ready"

# Verify connection
docker-compose exec postgres psql -U fraud_admin -d fraud_detection -c "SELECT version();"
```

#### **Step 3: Start API**
```bash
# Start API service
docker-compose -f docker/docker-compose.yml up -d api

# View logs
docker-compose logs api

# Wait for startup (should see "Application startup complete")
sleep 5

# Test API health
curl http://localhost:8000/health
```

#### **Step 4: Start Frontend**
```bash
# Start Frontend service
docker-compose -f docker/docker-compose.yml up -d frontend

# View logs
docker-compose logs frontend

# Access at http://localhost:8501
```

### **Option 3: Development Mode (Auto-Reload)**

```bash
# Start with volume mounts for live reloading
docker-compose -f docker/docker-compose.yml \
  -f docker/docker-compose.override.yml \
  up -d

# Changes to .py files will auto-reload
```

---

## Database Setup

### **1. Initialize Database Schema**

```bash
# Run migrations (auto-runs on API startup)
docker-compose exec api alembic upgrade head

# Verify schema was created
docker-compose exec postgres psql -U fraud_admin -d fraud_detection -c "\dt"
```

Expected tables:
- `users` - API users
- `claims` - Insurance claims
- `audit_logs` - Compliance audit trail
- `predictions` - Model predictions with timestamps
- `performance_metrics` - System metrics

### **2. Load Sample Data**

```bash
# Generate synthetic data
docker-compose exec api python data/generate_synthetic.py \
  --n-patients 1000 \
  --n-doctors 100 \
  --n-hospitals 50

# Load into database
docker-compose exec api python -c "
from backend.database.session import SessionLocal
from src.data.preprocessor import preprocess_claims
import pandas as pd

df = pd.read_csv('data/processed/claims_engineered.csv')
# Insert into database (code to implement)
"

# Verify data
docker-compose exec postgres psql -U fraud_admin -d fraud_detection -c "SELECT COUNT(*) FROM claims;"
```

### **3. Backup & Restore**

```bash
# Backup database
docker-compose exec postgres pg_dump -U fraud_admin fraud_detection > backup.sql

# Restore from backup
docker-compose exec -T postgres psql -U fraud_admin fraud_detection < backup.sql

# Backup to file on host
docker-compose exec postgres pg_dump -U fraud_admin fraud_detection > ~/fraud_detection_backup.sql
```

---

## Model Training

### **Option 1: Run Inside Docker**

```bash
# Train baseline + GNN models (runs once, takes 10-30 min)
docker-compose --profile train -f docker/docker-compose.yml up pipeline

# View logs
docker-compose logs pipeline

# When done, models are saved to `models/` directory on host
```

### **Option 2: Pre-load Trained Models**

If you already have trained models:

```bash
# Copy model files to models/ directory
cp your_models/* models/

# Verify models exist
ls -la models/baseline/*.pkl
ls -la models/gnn/*.pt

# Restart API to load models
docker-compose restart api
```

### **Option 3: Train Locally (for development)**

```bash
# Install dependencies locally
pip install -r requirements.txt

# Train models
python scripts/train_gnn.py \
  --epochs 50 \
  --batch-size 64 \
  --device cpu

# Or run full pipeline
python scripts/run_pipeline.py
```

---

## Accessing the Application

### **1. FastAPI Backend**

**Swagger UI:** http://localhost:8000/docs
- Interactive API documentation
- Test all endpoints from browser
- See request/response schemas

**ReDoc UI:** http://localhost:8000/redoc
- Alternative API documentation (read-only)

**Example API Calls:**

```bash
# Get authentication token
TOKEN=$(curl -s -X POST http://localhost:8000/token \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "api_key=test_key_123" | jq -r '.access_token')

# Predict single claim
curl -X POST http://localhost:8000/predict \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "claim_id": "CLM-001",
    "patient_id": "PAT-001",
    "doctor_id": "DOC-001",
    "hospital_id": "HOS-001",
    "amount": 5000,
    "service_type": "Surgery"
  }'

# Get system health
curl http://localhost:8000/health | jq

# Get statistics
curl http://localhost:8000/stats | jq
```

### **2. Streamlit Frontend**

**Access:** http://localhost:8501

**Navigation Tabs:**
- 🏠 Dashboard - Fraud metrics overview
- 🔍 Single Claim - Manual prediction
- 📁 Bulk Upload - Batch processing
- 📊 Model Analytics - Model comparison
- 🔍 Data Quality - Data drift detection
- ⚡ Performance - System performance
- 🛡️ Resilience - Health checks & failover

**Usage:**
1. Enter your API key (default: `test_key_123`)
2. Submit claims or upload CSV files
3. View predictions and explanations

### **3. PostgreSQL Database**

**Connect via CLI:**
```bash
docker-compose exec postgres psql -U fraud_admin -d fraud_detection

# Common queries
\dt                    # List tables
SELECT COUNT(*) FROM claims;
SELECT * FROM audit_logs ORDER BY created_at DESC LIMIT 10;
\q                      # Exit
```

**Connect via GUI (e.g., DBeaver):**
- Host: `localhost`
- Port: `5432`
- Database: `fraud_detection`
- User: `fraud_admin`
- Password: (from `.env`)

---

## Monitoring & Health Checks

### **1. Service Health Status**

```bash
# API Health
curl http://localhost:8000/health | jq

# Expected response:
# {
#   "status": "healthy",
#   "uptime_seconds": 3600,
#   "database": "connected",
#   "models_loaded": true,
#   "cache": "operational",
#   "timestamp": "2024-01-15T10:30:00Z"
# }
```

### **2. Docker Container Status**

```bash
# List all containers
docker-compose ps

# View resource usage
docker stats

# View container logs
docker-compose logs api          # Last 100 lines
docker-compose logs api -f       # Follow in real-time
docker-compose logs api --tail=50  # Last 50 lines
```

### **3. Prometheus Metrics**

```bash
# Access metrics endpoint
curl http://localhost:8000/metrics

# Example metrics:
# predictions_total{model="random_forest"} 1523
# prediction_latency_seconds{model="gnn"} 0.045
# fraud_detection_rate 0.185
# api_requests_total{endpoint="/predict"} 5432
```

### **4. System Monitoring**

```bash
# Monitor CPU/Memory
watch docker stats

# Check disk usage
docker system df

# View network traffic
docker network inspect fraud-network
```

---

## Production Hardening

### **1. Environment Setup**

Create `.env.prod`:
```bash
# SECURITY: Use strong passwords!
POSTGRES_PASSWORD=<generate-random-64-char-password>
JWT_SECRET_KEY=<generate-random-64-char-key>
API_KEY=<generate-api-key>

# Disable debug mode
API_LOG_LEVEL=warning
DEBUG=false

# Enable monitoring & security
ENABLE_MONITORING=true
ENABLE_HEALTH_CHECKS=true
ENABLE_CIRCUIT_BREAKER=true
ENABLE_RATE_LIMITING=true
ENABLE_COMPLIANCE=true

# Performance
API_WORKERS=8
RATE_LIMIT_PREDICTIONS_PER_SEC=1000
RATE_LIMIT_BULK_UPLOADS_PER_MIN=100

# Resilience
MAX_RETRIES=3
CIRCUIT_BREAKER_FAILURE_THRESHOLD=5
CIRCUIT_BREAKER_TIMEOUT_SECONDS=60
```

### **2. Docker Compose Production Config**

Create `docker/docker-compose.prod.yml`:

```yaml
version: "3.9"

services:
  api:
    build:
      context: ..
      dockerfile: docker/Dockerfile.api
    restart: always
    ports:
      - "8000:8000"
    environment:
      - LOG_LEVEL=warning
      - API_WORKERS=8
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 10s
    deploy:
      resources:
        limits:
          cpus: "4"
          memory: 4GB
        reservations:
          cpus: "2"
          memory: 2GB
    networks:
      - fraud-network

  frontend:
    build:
      context: ..
      dockerfile: docker/Dockerfile.frontend
    restart: always
    ports:
      - "8501:8501"
    deploy:
      resources:
        limits:
          cpus: "2"
          memory: 2GB

  postgres:
    image: postgres:15-alpine
    restart: always
    volumes:
      - postgres_data:/var/lib/postgresql/data
    environment:
      - POSTGRES_PASSWORD=<from-.env>
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U fraud_admin"]
      interval: 10s
      timeout: 5s
      retries: 5

volumes:
  postgres_data:
    driver: local

networks:
  fraud-network:
    driver: bridge
```

**Deploy with production config:**
```bash
docker-compose -f docker/docker-compose.yml \
               -f docker/docker-compose.prod.yml \
               up -d
```

### **3. SSL/TLS & HTTPS**

```bash
# Generate self-signed certificate (development)
openssl req -x509 -newkey rsa:4096 -nodes \
  -out cert.pem -keyout key.pem -days 365

# Or use Let's Encrypt (production)
certbot certonly --standalone -d your-domain.com
```

Update docker-compose.yml to mount certificates.

### **4. Backup Strategy**

```bash
#!/bin/bash
# backup.sh - Run daily via cron

BACKUP_DIR="/mnt/backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Backup database
docker-compose exec -T postgres pg_dump -U fraud_admin fraud_detection | \
  gzip > "$BACKUP_DIR/fraud_db_$TIMESTAMP.sql.gz"

# Backup models
tar -czf "$BACKUP_DIR/models_$TIMESTAMP.tar.gz" models/

# Keep only last 30 days
find "$BACKUP_DIR" -mtime +30 -exec rm {} \;

echo "Backup completed: $BACKUP_DIR"
```

Add to crontab:
```bash
crontab -e
# Add: 0 2 * * * /path/to/backup.sh  (runs daily at 2 AM)
```

---

## Troubleshooting

### **Issue: API won't start**

```bash
# Check logs
docker-compose logs api

# Common causes:
# 1. Port 8000 already in use
lsof -i :8000  # Kill process: kill -9 <PID>

# 2. Models not found
ls -la models/
# Ensure models exist or run training

# 3. Database not accessible
docker-compose logs postgres
```

### **Issue: Frontend can't connect to API**

```bash
# Check network connectivity
docker-compose exec frontend curl http://api:8000/health

# Check if API is running
docker-compose ps

# Restart both services
docker-compose restart api frontend
```

### **Issue: High memory usage**

```bash
# Check container memory
docker stats

# Reduce workers or model batch size
# Edit .env: API_WORKERS=2

# Restart
docker-compose restart api
```

### **Issue: Database connection timeout**

```bash
# Check PostgreSQL health
docker-compose exec postgres pg_isready

# View Postgres logs
docker-compose logs postgres

# Increase timeout in connection string (if using SQLAlchemy)
pool_recycle=3600
pool_pre_ping=true
```

### **Issue: Models not loading**

```bash
# Verify model files exist
ls -la models/baseline/
ls -la models/gnn/

# Check model file sizes (should not be 0 bytes)
du -h models/*

# If missing, run training
docker-compose --profile train up

# Or copy pre-trained models to models/ directory
```

### **Issue: Out of disk space**

```bash
# Check disk usage
docker system df

# Remove unused images
docker system prune

# Remove dangling volumes
docker volume prune

# Clear logs
docker-compose logs --help | grep -i tail
# Reduce log retention in docker-compose.yml
```

---

## Cloud Deployment

### **AWS ECS (Elastic Container Service)**

```bash
# 1. Push images to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com

docker tag fraud-api:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/fraud-api:latest
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/fraud-api:latest

# 2. Create ECS task definition (see DEPLOYMENT.md in docs/)

# 3. Run on ECS
aws ecs create-service \
  --cluster fraud-detection \
  --service-name fraud-api \
  --task-definition fraud-api:1 \
  --desired-count 2
```

### **Google Cloud Run**

```bash
# 1. Authenticate
gcloud auth login

# 2. Build and push
gcloud builds submit --tag gcr.io/$PROJECT/fraud-api

# 3. Deploy
gcloud run deploy fraud-api \
  --image gcr.io/$PROJECT/fraud-api \
  --platform managed \
  --memory 2Gi \
  --cpu 2 \
  --allow-unauthenticated
```

### **Azure Container Instances**

```bash
# 1. Push to ACR
az acr build --registry <registry-name> --image fraud-api:latest .

# 2. Deploy
az container create \
  --resource-group myResourceGroup \
  --name fraud-api \
  --image <registry-name>.azurecr.io/fraud-api:latest \
  --cpu 2 --memory 2
```

---

## Verification Checklist

After deployment, verify everything works:

- [ ] API is running: `curl http://localhost:8000/health`
- [ ] Frontend is accessible: `http://localhost:8501`
- [ ] Database connected: `docker-compose exec postgres pg_isready`
- [ ] Models loaded: Check API logs for "Model loaded successfully"
- [ ] Authentication works: `curl -X POST http://localhost:8000/token`
- [ ] Can make predictions: Submit claim via frontend or API
- [ ] Metrics being collected: `curl http://localhost:8000/metrics`
- [ ] No errors in logs: `docker-compose logs --grep ERROR`

---

## Performance Tuning

### **API Optimization**

```bash
# Increase workers
API_WORKERS=8  # More workers = more parallelism

# Increase timeout for bulk uploads
UPLOAD_TIMEOUT=300  # seconds

# Enable caching
ENABLE_CACHE=true
CACHE_TTL=3600
```

### **Database Optimization**

```bash
# Create indexes on frequently queried columns
docker-compose exec postgres psql -U fraud_admin fraud_detection -c "
CREATE INDEX idx_claims_patient_id ON claims(patient_id);
CREATE INDEX idx_claims_created_at ON claims(created_at);
CREATE INDEX idx_predictions_fraud_score ON predictions(fraud_score DESC);
"
```

### **Frontend Performance**

```bash
# Streamlit caching
@st.cache_data
def get_stats():
    return api_get("/stats")

# Lazy loading for large tables
@st.cache_resource
def get_large_dataset():
    pass
```

---

## Next Steps

1. **Deploy & Test**: Follow Quick Start section
2. **Load Data**: Use sample data or make real API calls
3. **Monitor**: Watch metrics at `/metrics` endpoint
4. **Configure**: Adjust `.env` for your environment
5. **Backup**: Set up automated database backups
6. **Scale**: Add load balancer for multi-instance deployment

---

## Support & Documentation

- **API Docs:** http://localhost:8000/docs
- **GitHub Issues:** Report bugs and feature requests
- **README.md:** System overview and features
- **FIXES_APPLIED.md:** Known issues and fixes

---

**🎉 You're ready to deploy! Start with Quick Start section above.**
