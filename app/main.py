from __future__ import annotations

import os
from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.router import router

BASE_CORS_ORIGINS = os.environ.get(
    "BACKEND_CORS_ORIGINS",
    "https://cine-match-ai-frontend.onrender.com,http://localhost:5173,http://localhost:3000,http://127.0.0.1:3000",
)
ORIGINS: List[str] = [origin.strip() for origin in BASE_CORS_ORIGINS.split(",") if origin.strip()]

app = FastAPI(
    title="Movie Recommendation API",
    version="1.0",
    description="A FastAPI backend for MovieLens movie browsing and recommendations.",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")


@app.get("/", summary="API root", description="Basic API health check and endpoint information.")
def root() -> dict[str, str]:
    return {"message": "Movie Recommendation API is running", "documentation": "/docs"}
