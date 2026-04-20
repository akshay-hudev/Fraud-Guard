# 🚀 Fraud-Guard: HF Spaces Deployment Guide

Complete step-by-step guide to deploy your entire project to Hugging Face Spaces.

---

## ✅ Pre-Deployment Checklist

- [ ] All models trained and saved in `training/models/baseline/`
- [ ] `requirements.txt` updated with all dependencies
- [ ] GitHub account with your project pushed
- [ ] Hugging Face account created
- [ ] Project structure verified

---

## 📋 Step 1: Verify Project Structure

Your project should have:

```
health-fraud-detection/
├── Dockerfile                    ✅ Created
├── .dockerignore                ✅ Created
├── requirements.txt             ✅ Already exists
├── backend/
│   ├── main.py
│   ├── predictor.py
│   ├── explainable_ai.py
│   ├── production_hardening.py
│   └── ... (all 16 files)
├── frontend/
│   └── app.py                  ✅ No changes needed!
├── training/
│   ├── src/
│   ├── data/
│   └── models/baseline/
│       ├── random_forest.pkl
│       ├── gradient_boosting.pkl
│       ├── logistic_regression.pkl
│       └── scaler.pkl
├── models/ (symlink or copy)
│   └── baseline/ (models for runtime)
└── README.md
```

**Verify:** Run in terminal:
```powershell
Test-Path training/models/baseline/random_forest.pkl
Test-Path training/models/baseline/gradient_boosting.pkl
```

Both should return `True`.

---

## 🔄 Step 2: Push to GitHub

Your project needs to be on GitHub for HF Spaces to pull it.

```powershell
# Navigate to project
cd C:\Users\Lenovo\OneDrive\Desktop\Akshay\Programming\ML_Assignment\Fraud_Detection\health-fraud-detection

# Initialize git (if not done)
git init
git add .
git commit -m "Prepare for HF Spaces deployment"

# Add remote (replace with your repo)
git remote add origin https://github.com/YOUR_USERNAME/health-fraud-detection.git
git branch -M main
git push -u origin main
```

**Verify:** Check GitHub - all files should be there.

---

## 🎯 Step 3: Create HF Space

### Option A: Web Interface (Easiest)

1. Go to **https://huggingface.co/spaces**
2. Click **"Create new Space"**
3. Configure:
   - **Owner**: Your HF username
   - **Space name**: `health-fraud-detection`
   - **License**: `openrail`
   - **Space SDK**: `Docker` ← **IMPORTANT!**
   - **Space hardware**: `CPU Basic` (Free)
4. Click **"Create Space"**

### Option B: Command Line

```powershell
# Install HF CLI
pip install huggingface-hub

# Login
huggingface-cli login

# Create space
huggingface-cli repo create health-fraud-detection --type space --space-sdk docker
```

---

## 📤 Step 4: Upload Files to HF Space

### Option A: Git Push (Recommended - Auto-sync)

```powershell
# Add HF remote
git remote add huggingface https://huggingface.co/spaces/YOUR_USERNAME/health-fraud-detection

# Push to HF
git push huggingface main --force
```

Your Space will automatically:
1. Detect the `Dockerfile`
2. Build the image (~5-10 minutes)
3. Start the container
4. Run both services

### Option B: Web UI Upload

1. Go to your Space on HF
2. Click **"Files"** tab
3. Upload all files:
   - Check "yes, upload as is"
   - Select entire folder OR upload critical files:
     - Dockerfile
     - requirements.txt
     - backend/ folder
     - frontend/ folder
     - training/models/baseline/ folder

---

## ⏳ Step 5: Wait for Build

Once files are uploaded, HF Spaces starts building:

1. Click **"Logs"** tab to monitor progress
2. Watch for messages:
   ```
   Building...
   Step 1/X: FROM python:3.11-slim
   Step 2/X: RUN apt-get update...
   ...
   Successfully built xxxxx
   Starting application...
   ```

**Build time:** 5-15 minutes (first time only)

---

## ✅ Step 6: Verify Deployment

Once build completes, you'll see the Streamlit interface.

### Check Backend is Running

1. Click **"App"** tab
2. Open your browser's Developer Console (F12)
3. Look for network requests to `/predict` - should succeed ✅

### Test Full Functionality

In the Streamlit dashboard:

- [ ] **Dashboard** - KPIs load
- [ ] **Single Claim** - Predictions work
- [ ] **Batch Upload** - CSV scoring works
- [ ] **Model Analytics** - Charts render
- [ ] **Graph Explorer** - Visualization loads
- [ ] **Live Feed** - Streaming predictions work

### Direct API Test

Visit these URLs in your browser (replace with your Space URL):

```
https://huggingface.co/spaces/YOUR_USERNAME/health-fraud-detection
```

Then test endpoint in browser console:

```javascript
fetch('/health')
    .then(r => r.json())
    .then(d => console.log(d))
```

Should return: `{"status": "healthy", ...}`

---

## 🔧 Step 7: Configure Environment (Optional)

If you need to set environment variables:

1. Go to Space **Settings** → **Configuration**
2. Add environment variables:
   ```
   LOG_LEVEL=info
   API_BASE=http://localhost:8000
   PYTHONUNBUFFERED=1
   ```

---

## 🎉 Step 8: Share Your Public URL

Your project is now **LIVE AND PUBLIC**!

Share the URL:
```
https://huggingface.co/spaces/YOUR_USERNAME/health-fraud-detection
```

### Share Examples:
- Add to GitHub README
- LinkedIn profile
- Portfolio website
- Resume
- Email to professors/friends
- Social media

---

## 📊 What's Running

| Service | Port | Status |
|---------|------|--------|
| Streamlit Frontend | 8501 (→ 7860) | ✅ Public |
| FastAPI Backend | 8000 (internal) | ✅ Running |
| JWT Auth | :8000/token | ✅ Working |
| SHAP Explanations | via /predict | ✅ Real-time |
| Model Registry | Loaded | ✅ 3 Models |

---

## 🐛 Troubleshooting

### Build Fails
- Check **Logs** tab for error
- Common fixes:
  - Ensure `requirements.txt` has all deps
  - Models must exist in `training/models/baseline/`
  - Run locally first: `docker build -t fraud-guard .`

### App starts but crashes
- Check **Logs** tab → "Container" tab
- Look for import errors or missing files
- Ensure `API_BASE = "http://localhost:8000"` in frontend/app.py

### Models not found
- Verify models exist before deployment:
  ```powershell
  ls training/models/baseline/
  ```
- Upload to HF Space or include in git

### Slow startup
- First request is slow (models load)
- Subsequent requests are faster
- Normal for ML apps

### Port already in use
- HF handles this automatically
- Remaps to free port (usually 7860)

---

## 🚀 Advanced: Persist Data Between Restarts

HF Spaces will restart periodically. To persist data:

1. **SQLite Database**: Stored in `/app/backend/fraud_detection.db`
   - Persists as long as Space is active
   - Resets on Space restart

2. **Add HuggingFace Datasets** (optional, advanced):
   ```python
   from huggingface_hub import HfApi
   
   api = HfApi()
   api.upload_folder(
       folder_path="logs/",
       repo_id="your-username/fraud-logs",
       repo_type="dataset"
   )
   ```

---

## 📝 Important Notes

1. **Free tier limitations:**
   - Space sleeps after 48 hours of no requests
   - CPU-only hardware (~2 vCPU)
   - No persistent storage beyond Space lifetime

2. **Performance:**
   - First request: ~3-5 seconds (model load)
   - Subsequent: ~500-1000ms per prediction
   - Batch predictions: faster

3. **Access:**
   - Public by default (anyone can access)
   - No authentication needed from users
   - API rate limiting is active (100 req/min)

4. **Updates:**
   - Push changes to GitHub
   - HF auto-rebuilds Space
   - Old Space restarts with new code

---

## ✨ Success Indicators

Your deployment is successful when:

- ✅ Space shows "Running" status
- ✅ Streamlit dashboard loads
- ✅ Predictions work (in <2 seconds)
- ✅ SHAP explanations render
- ✅ All pages accessible
- ✅ API endpoints respond (check Logs)
- ✅ No errors in browser console

---

## 📞 Need Help?

If deployment fails:

1. Check **Logs** → "Container" for errors
2. Verify `Dockerfile` syntax: `docker build -t fraud-guard .` (locally)
3. Ensure models are in git repo
4. Review `requirements.txt` for typos

---

## 🎓 Next Steps (Optional)

After successful deployment:

1. Add Space badge to GitHub README:
   ```markdown
   [![Fraud-Guard](https://img.shields.io/badge/🤗-View_on_HuggingFace-yellow)](https://huggingface.co/spaces/YOUR_USERNAME/health-fraud-detection)
   ```

2. Create demo video showing the app

3. Share on social media

4. Get feedback from users

---

**Congratulations! 🎉 Your Fraud-Guard project is now LIVE online!**
