# Movie Recommendation System

This project uses MovieLens CSV files and optional Kaggle movie metadata to build a local movie recommender.

## Files

- `movie.py` - recommender implementation
- `requirements.txt` - Python dependencies
- `archive/ml-latest-small/` - MovieLens dataset files (`movies.csv`, `ratings.csv`, `tags.csv`, `links.csv`)

## Features

- Content-based recommendations from user interest text
- Collaborative filtering using `surprise.SVD`, `surprise.NMF`, `surprise.KNNBasic`
- Fallback recommendations based on top-rated movies
- Optional Kaggle metadata enrichment when `movies_metadata.csv` is available
- Hybrid recommendation support combining content and collaborative scores

## Setup

Create and activate a Python environment, then install dependencies:

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

Run the recommender from the project folder:

```bash
python movie.py --interest "adventure fantasy" --top-n 10
```

For collaborative recommendations by user ID:

```bash
python movie.py --user-id 1 --top-n 10 --model SVD
```

If you have Kaggle metadata available, include it:

```bash
python movie.py --interest "science fiction" --top-n 10 --kaggle-metadata path\to\movies_metadata.csv
```

If interest is weak or not provided, the fallback top-rated movies are returned.

## Notes

- `scikit-surprise` is required for collaborative filtering.
- The recommender uses the local archive dataset by default.

## Render Deployment

This repository includes a `render.yaml` configuration for deploying the FastAPI backend and React frontend as separate services on Render's free tier.

### Render environment variables

The configurations are defined in the `render.yaml` blueprint. If deploying manually or verifying settings, use these values (also provided in `render.env.example`):

- **Backend Service (`cine-match-ai-backend`)**:
  - `DATA_DIR=archive/ml-latest-small`
  - `MODELS_DIR=trained_models`
  - `BACKEND_CORS_ORIGINS=https://cine-match-ai.onrender.com` (change this if you use a custom domain)
- **Frontend Static Site (`cine-match-ai`)**:
  - `VITE_API_BASE_URL=https://cine-match-ai-backend.onrender.com/api`

### Production build & startup commands

- **Backend Web Service**:
  - **Build Command**: `pip install -r requirements.txt && python -m app.models.train_collaborative && python -m app.models.train_content` (automatically installs requirements, trains SVD/NMF/KNN models, and generates the TF-IDF vocabulary on the build server).
  - **Start Command**: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
- **Frontend Static Site**:
  - **Build Command**: `cd frontend && npm install && npm run build`
  - **Publish Path**: `frontend/dist`

### Deployment steps

1. Push your repository to GitHub.
2. Log in to [Render](https://dashboard.render.com/).
3. Click **New** -> **Blueprint**.
4. Connect your GitHub repository.
5. Render will automatically read the `render.yaml` configuration. Review the plan and click **Apply**.
6. Once deployed, the frontend will be served at `https://cine-match-ai.onrender.com` and the API at `https://cine-match-ai-backend.onrender.com`.

### Free Render optimizations

- **Low-Memory Similarity calculations**: The backend utilizes an optimized content recommender that calculates movie similarity on-the-fly. This keeps the application memory footprint under **100MB** during runtime, completely avoiding Out-of-Memory (OOM) crashes on the 512MB free tier instance.
- **Build-time Training**: Pre-training models on Render's build server ensures the deployment is fully automated and self-contained, without having to commit or push large model pickle files to Git.
