# Deployment Guide

This application consists of two parts:
1. **Backend** (FastAPI) - Needs Python runtime
2. **Frontend** (Static HTML/JS) - Can be deployed to Netlify

## Option 1: Deploy Backend to Railway (Recommended)

1. **Sign up/Login to Railway**: https://railway.app
2. **Create New Project**: Click "New Project" → "Deploy from GitHub repo"
3. **Select your repository**: Choose `garlic-fe`
4. **Configure Environment Variables**:
   - `GARLIC_MODEL_WEIGHTS` (optional): Path to your trained model weights
   - `GARLIC_CONFIDENCE_THRESHOLD` (optional): Default 0.5
5. **Deploy**: Railway will automatically detect the Python app and deploy
6. **Get your backend URL**: Railway will provide a URL like `https://your-app.railway.app`
7. **Update frontend**: Edit `frontend/static/index.html` and replace `https://your-backend-url.railway.app` with your actual Railway URL

## Option 2: Deploy Backend to Render

1. **Sign up/Login to Render**: https://render.com
2. **Create New Web Service**: Connect your GitHub repo
3. **Configure**:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
4. **Add Environment Variables** (same as Railway)
5. **Deploy**: Render will build and deploy automatically
6. **Update frontend**: Replace backend URL in `frontend/static/index.html`

## Deploy Frontend to Netlify

1. **Sign up/Login to Netlify**: https://netlify.com
2. **Create New Site**: 
   - Connect to GitHub
   - Select your `garlic-fe` repository
   - **Build settings**:
     - Base directory: (leave empty)
     - Build command: `mkdir -p frontend/static && cp frontend/static/index.html frontend/static/ 2>/dev/null || echo "Static files ready"`
     - Publish directory: `frontend/static`
3. **Update API URL**: 
   - Go to Site settings → Environment variables
   - Add `REACT_APP_API_URL` (or update the JavaScript in `index.html` directly)
4. **Deploy**: Netlify will build and deploy automatically

## Alternative: Deploy Everything to Railway

Railway can also serve static files. You can deploy both backend and frontend together:

1. Follow Railway deployment steps above
2. Add a static file server route in `backend/main.py`:
   ```python
   from fastapi.staticfiles import StaticFiles
   app.mount("/", StaticFiles(directory="frontend/static", html=True), name="static")
   ```
3. Update `frontend/static/index.html` to use relative API paths: `/api/...`

## Environment Variables

### Backend (Railway/Render):
- `GARLIC_MODEL_WEIGHTS`: Path to model weights file (optional)
- `GARLIC_CONFIDENCE_THRESHOLD`: Detection confidence threshold (default: 0.5)
- `PORT`: Automatically set by hosting platform

### Frontend (Netlify):
- Update `API_BASE_URL` in `frontend/static/index.html` JavaScript section

## Testing Locally

1. **Start backend**:
   ```bash
   uvicorn backend.main:app --reload
   ```

2. **Serve frontend**:
   ```bash
   cd frontend/static
   python -m http.server 8080
   ```
   Or use any static file server.

3. **Open browser**: http://localhost:8080

## Notes

- The backend needs access to your model weights file. Upload them to the hosting platform or use a cloud storage service (S3, etc.)
- For production, consider adding:
  - CORS configuration
  - Rate limiting
  - Authentication
  - Error logging/monitoring

