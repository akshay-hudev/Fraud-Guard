# Cloud Deployment Guide

## Option A — Render (Simplest, Free Tier)

### 1. Push to GitHub
```bash
git init
git add .
git commit -m "feat: initial fraud detection system"
git remote add origin https://github.com/YOUR_USERNAME/health-fraud-detection.git
git push -u origin main
```

### 2. Deploy API on Render
1. Go to https://render.com → New → Web Service
2. Connect your GitHub repository
3. Configure:
   - **Environment**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
   - **Instance Type**: Free (512 MB RAM — enough for RF model)
4. Add environment variable: `PYTHONPATH=/opt/render/project/src`
5. Click **Deploy**

### 3. Deploy Frontend on Render
1. New → Web Service → same repo
2. **Start Command**: `streamlit run frontend/app.py --server.port $PORT --server.address 0.0.0.0`
3. Set env var: `API_BASE=https://your-api-service.onrender.com`

---

## Option B — AWS (Production Grade)

### Architecture
```
Route53 → CloudFront → ALB
                         ├── ECS Fargate (API)   ← ECR image
                         └── ECS Fargate (UI)    ← ECR image
                         S3 (models + data)
                         CloudWatch (logs + metrics)
```

### 1. Build & Push to ECR
```bash
# Authenticate
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin \
  123456789.dkr.ecr.us-east-1.amazonaws.com

# Build
docker build -f docker/Dockerfile.api -t fraud-api .
docker build -f docker/Dockerfile.frontend -t fraud-frontend .

# Tag & push
docker tag fraud-api:latest 123456789.dkr.ecr.us-east-1.amazonaws.com/fraud-api:latest
docker push 123456789.dkr.ecr.us-east-1.amazonaws.com/fraud-api:latest
```

### 2. ECS Task Definition (API)
```json
{
  "family": "fraud-api",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "512",
  "memory": "1024",
  "containerDefinitions": [{
    "name": "fraud-api",
    "image": "123456789.dkr.ecr.us-east-1.amazonaws.com/fraud-api:latest",
    "portMappings": [{"containerPort": 8000}],
    "environment": [{"name": "PYTHONPATH", "value": "/app"}],
    "logConfiguration": {
      "logDriver": "awslogs",
      "options": {
        "awslogs-group": "/ecs/fraud-api",
        "awslogs-region": "us-east-1",
        "awslogs-stream-prefix": "ecs"
      }
    }
  }]
}
```

### 3. Store models on S3
```bash
# Upload trained models
aws s3 sync models/ s3://your-bucket/fraud-models/
aws s3 sync data/processed/ s3://your-bucket/fraud-data/

# Add to startup script: download from S3 before server starts
# CMD: aws s3 sync s3://your-bucket/fraud-models/ models/ && uvicorn backend.main:app ...
```

---

## Option C — Docker Compose (Self-hosted VPS)

```bash
# On your server (Ubuntu 22.04)
apt install docker.io docker-compose-v2 -y

git clone https://github.com/YOUR/health-fraud-detection.git
cd health-fraud-detection

# Train models first
docker compose -f docker/docker-compose.yml --profile train up pipeline

# Start full stack
docker compose -f docker/docker-compose.yml up -d

# Check
docker compose ps
curl http://localhost:8000/health
```

### Nginx reverse proxy config
```nginx
server {
    listen 80;
    server_name yourdomain.com;

    location /api/ {
        proxy_pass http://localhost:8000/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    location / {
        proxy_pass http://localhost:8501/;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

---

## Environment Variables Reference

| Variable | Description | Default |
|---|---|---|
| `PYTHONPATH` | Python module path | `/app` |
| `API_BASE` | Backend URL for frontend | `http://localhost:8000` |
| `LOG_LEVEL` | Logging level | `info` |
| `MODEL_DIR` | Model storage path | `models/` |
| `DATA_DIR` | Data storage path | `data/` |
