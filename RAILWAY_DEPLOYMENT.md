# Railway Backend Deployment Guide

## Step-by-Step Instructions

### 1. Sign up for Railway
- Go to https://railway.app
- Sign up with your GitHub account (recommended for easy repo connection)

### 2. Create New Project
1. Click **"New Project"** button
2. Select **"Deploy from GitHub repo"**
3. Authorize Railway to access your GitHub if prompted
4. Select your repository: `hckw/garlic` (or your repo name)
5. Select the branch: `main`

### 3. Configure the Service
Railway will auto-detect it's a Python app. You may need to configure:

**Settings → Build:**
- Railway will use `nixpacks.toml` automatically
- Or manually set:
  - **Build Command**: `pip install -r requirements-backend.txt`
  - **Start Command**: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`

### 4. Set Environment Variables
Go to your service → **Variables** tab and add:

- `GARLIC_MODEL_WEIGHTS` (optional): Path to your trained model weights file
- `GARLIC_CONFIDENCE_THRESHOLD` (optional): Default is `0.5`
- `ALLOWED_ORIGINS` (optional): Comma-separated list of allowed CORS origins
  - Example: `https://your-streamlit-app.streamlit.app,https://your-netlify-site.netlify.app`

### 5. Deploy
- Railway will automatically start building and deploying
- Watch the logs to see the build progress
- Once deployed, Railway will provide a URL like: `https://your-app.railway.app`

### 6. Get Your Backend URL
- Go to your service → **Settings** → **Domains**
- Copy the generated Railway domain (e.g., `https://garlic-production.up.railway.app`)
- This is your backend API URL

### 7. Update Streamlit Frontend
1. Go to Streamlit Cloud → Your app → **Settings** → **Secrets**
2. Add/Update: `GARLIC_API_URL` = `https://your-railway-url.railway.app`
3. Save and redeploy

## Troubleshooting

### Build Fails
- Check Railway logs for errors
- Ensure `requirements-backend.txt` is in the repo
- Verify Python version compatibility

### API Not Responding
- Check that the service is running (green status)
- Verify the start command is correct
- Check logs for runtime errors

### CORS Errors
- Add your Streamlit/Netlify domain to `ALLOWED_ORIGINS` environment variable
- Or set `ENVIRONMENT=development` to allow all origins (not recommended for production)

### Model Weights Not Found
- If you have trained model weights, upload them to Railway
- Or use a cloud storage service (S3, etc.) and update the path
- The app will use pretrained weights if `GARLIC_MODEL_WEIGHTS` is not set

## Railway Free Tier Limits
- 500 hours of usage per month
- $5 credit (enough for small projects)
- Automatic sleep after inactivity (wakes on request)

## Next Steps
After deployment:
1. Test your backend: Visit `https://your-railway-url.railway.app/health`
2. Update Streamlit Cloud with the backend URL
3. Test the full workflow end-to-end

