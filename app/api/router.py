from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from app.popularity import calculate_popularity_scores, most_rated_movies, top_rated_movies, trending_movies
from app.preprocessing.load_data import load_movielens_data
from app.preprocessing.metadata import load_enriched_movielens_movies
from app.recommender import ContentRecommender, HybridRecommender, InterestRecommender
from app.utils.wikipedia import fill_missing_with_wikipedia
from app.api.schemas import (
    GenreSchema,
    MovieSchema,
    PopularResponse,
    PopularMovieSchema,
    RecommendationMovieSchema,
    RecommendationResponse,
    RecommendHybridRequest,
    RecommendInterestRequest,
    RecommendMovieRequest,
    RecommendUserRequest,
)


DEFAULT_DATA_DIR = Path(os.environ.get("DATA_DIR", Path(__file__).resolve().parents[2] / "archive" / "ml-latest-small"))
DEFAULT_MODEL_DIR = Path(os.environ.get("MODELS_DIR", Path(__file__).resolve().parents[2] / "trained_models"))
MODEL_FILES: dict[str, str] = {"SVD": "svd.pkl", "NMF": "nmf.pkl", "KNNBasic": "knn.pkl"}


router = APIRouter(prefix="", tags=["movies"])


def _load_movies() -> pd.DataFrame:
    movies = load_enriched_movielens_movies(DEFAULT_DATA_DIR).copy()
    movies["title"] = movies["title"].fillna("").astype(str)
    movies["genres"] = movies["genres"].fillna("").astype(str)
    if "overview" not in movies.columns:
        movies["overview"] = ""
    else:
        movies["overview"] = movies["overview"].fillna("").astype(str)
    return movies


def _movie_to_dict(row: pd.Series) -> Dict[str, Any]:
    return {
        "movieId": int(row["movieId"]),
        "title": str(row["title"]),
        "genres": str(row.get("genres", "") or ""),
        "overview": str(row.get("overview", "") or ""),
        "poster_path": row.get("poster_path") if pd.notna(row.get("poster_path")) else None,
        "poster_url": row.get("poster_url") if pd.notna(row.get("poster_url")) else None,
        "runtime": float(row["runtime"]) if pd.notna(row.get("runtime")) else None,
        "release_date": str(row.get("release_date", "") or ""),
        "vote_average": float(row["vote_average"]) if pd.notna(row.get("vote_average")) else None,
        "imdb_id": str(row.get("imdb_id", "") or ""),
    }


_MOVIES_DF = _load_movies()
_RATINGS_DF = load_movielens_data(DEFAULT_DATA_DIR)["ratings"].copy()


@router.get(
    "/movies",
    response_model=List[MovieSchema],
    summary="List available movies",
    description="Return the available MovieLens movie titles and genres.",
)
def get_movies(limit: int = Query(100, ge=1, le=500, description="Maximum number of movies to return.")) -> List[Dict[str, Any]]:
    return [_movie_to_dict(row) for _, row in _MOVIES_DF.head(limit).iterrows()]


@router.get(
    "/genres",
    response_model=List[GenreSchema],
    summary="List available genres",
    description="Return all unique genres from the MovieLens dataset.",
)
def get_genres() -> List[Dict[str, str]]:
    unique_genres = sorted(
        {genre for genres in _MOVIES_DF["genres"].fillna("") for genre in genres.split("|") if genre}
    )
    return [{"name": genre} for genre in unique_genres]


@router.get(
    "/popular",
    response_model=PopularResponse,
    summary="Get popular movie rankings",
    description="Return top rated, trending, and most rated movies.",
)
def get_popular(
    top_n: int = Query(10, ge=1, le=50, description="Number of movies in each popular category."),
    recent_percentile: float = Query(0.80, ge=0.0, le=1.0, description="Percentile cutoff for trending movies."),
    top_percentile: float = Query(0.90, ge=0.0, le=1.0, description="Percentile cutoff for weighted rating threshold."),
) -> Dict[str, List[Dict[str, Any]]]:
    popularity = calculate_popularity_scores(DEFAULT_DATA_DIR, top_percentile=top_percentile)
    return {
        "top_rated": [fill_missing_with_wikipedia(movie) for movie in top_rated_movies(popularity, top_n=top_n)],
        "trending": [
            fill_missing_with_wikipedia(movie)
            for movie in trending_movies(popularity, _RATINGS_DF, top_n=top_n, recent_percentile=recent_percentile)
        ],
        "most_rated": [fill_missing_with_wikipedia(movie) for movie in most_rated_movies(popularity, top_n=top_n)],
    }


@router.post(
    "/recommend/movie",
    response_model=RecommendationResponse,
    summary="Recommend movies based on a movie title",
    description="Return movie recommendations using fuzzy title matching and content similarity.",
)
def recommend_movie(request: RecommendMovieRequest) -> RecommendationResponse:
    try:
        recommender = ContentRecommender(data_dir=DEFAULT_DATA_DIR, models_dir=DEFAULT_MODEL_DIR)
        movies = recommender.recommend_similar(request.title, top_n=request.top_n, min_match_score=request.min_match_score)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    return {
        "recommendations": [
            fill_missing_with_wikipedia(
                {
                    "title": rec.title,
                    "genres": rec.genres,
                    "overview": rec.overview,
                    "poster_path": rec.poster_path,
                    "poster_url": rec.poster_path,
                    "similarity_score": rec.similarity_score,
                }
            )
            for rec in movies
        ]
    }


@router.post(
    "/recommend/interest",
    response_model=RecommendationResponse,
    summary="Recommend movies based on user interest",
    description="Return movie recommendations by matching user interest text against movie content.",
)
def recommend_interest(request: RecommendInterestRequest) -> RecommendationResponse:
    try:
        recommender = InterestRecommender(data_dir=DEFAULT_DATA_DIR)
        movies = recommender.recommend(request.interest, top_n=request.top_n)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    return {"recommendations": [
        fill_missing_with_wikipedia(
            {
                "title": movie["title"],
                "genres": movie["genres"],
                "overview": movie.get("overview", ""),
                "poster_path": movie.get("poster_path"),
                "poster_url": movie.get("poster_url") or movie.get("poster_path"),
                "runtime": movie.get("runtime"),
                "release_date": movie.get("release_date"),
                "vote_average": movie.get("vote_average"),
                "imdb_id": movie.get("imdb_id"),
                "similarity_score": movie.get("similarity_score"),
            }
        )
        for movie in movies
    ]}


def _load_collaborative_model(model_name: str) -> Any:
    if model_name not in MODEL_FILES:
        raise HTTPException(status_code=400, detail=f"Unsupported model_name '{model_name}'. Choose from {list(MODEL_FILES)}.")
    model_path = DEFAULT_MODEL_DIR / MODEL_FILES[model_name]
    if not model_path.exists():
        raise HTTPException(
            status_code=503,
            detail=f"Collaborative model artifact not found: {model_path}. Train it with app/models/train_collaborative.py.",
        )
    return joblib.load(model_path)


@router.post(
    "/recommend/user",
    response_model=RecommendationResponse,
    summary="Recommend movies for a specific user",
    description="Return top movies for a user using a trained collaborative filtering model.",
)
def recommend_user(request: RecommendUserRequest) -> RecommendationResponse:
    algo = _load_collaborative_model(request.model_name)
    watched = set(_RATINGS_DF.loc[_RATINGS_DF["userId"] == request.user_id, "movieId"].astype(int).tolist())
    candidates = _MOVIES_DF.loc[~_MOVIES_DF["movieId"].isin(watched)].copy()

    predictions: List[Dict[str, Any]] = []
    for _, row in candidates.iterrows():
        movie_id = int(row["movieId"])
        try:
            prediction = algo.predict(request.user_id, movie_id)
            score = float(prediction.est)
        except Exception:
            continue
        predictions.append(
            {
                "title": str(row["title"]),
                "genres": str(row.get("genres", "") or ""),
                "overview": str(row.get("overview", "") or ""),
                "poster_path": row.get("poster_path") if pd.notna(row.get("poster_path")) else None,
                "poster_url": row.get("poster_url") if pd.notna(row.get("poster_url")) else None,
                "runtime": float(row["runtime"]) if pd.notna(row.get("runtime")) else None,
                "release_date": str(row.get("release_date", "") or ""),
                "vote_average": float(row["vote_average"]) if pd.notna(row.get("vote_average")) else None,
                "imdb_id": str(row.get("imdb_id", "") or ""),
                "predicted_rating": score,
            }
        )

    top_predictions = sorted(predictions, key=lambda item: item["predicted_rating"], reverse=True)[: request.top_n]
    return {"recommendations": [fill_missing_with_wikipedia(movie) for movie in top_predictions]}


@router.post(
    "/recommend/hybrid",
    response_model=RecommendationResponse,
    summary="Recommend hybrid movies using content, collaborative, and popularity signals",
    description="Return hybrid movie recommendations for a user and interest query.",
)
def recommend_hybrid(request: RecommendHybridRequest) -> RecommendationResponse:
    try:
        recommender = HybridRecommender(
            data_dir=DEFAULT_DATA_DIR,
            models_dir=DEFAULT_MODEL_DIR,
            model_name=request.model_name,
        )
        movies = recommender.recommend(request.interest, request.user_id, top_n=request.top_n)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return {"recommendations": [fill_missing_with_wikipedia(movie) for movie in movies]}
