from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from app.preprocessing.load_data import load_movielens_data
from app.preprocessing.metadata import load_enriched_movielens_movies

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "archive" / "ml-latest-small"


def _load_movie_and_rating_data(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    data = load_movielens_data(data_dir)
    movies = load_enriched_movielens_movies(data_dir).copy()
    ratings = data["ratings"].copy()

    movies["title"] = movies["title"].fillna("").astype(str)
    movies["genres"] = movies["genres"].fillna("").astype(str)
    movies["overview"] = movies["overview"].fillna("").astype(str) if "overview" in movies.columns else ""

    if "timestamp" in ratings.columns:
        ratings["timestamp"] = pd.to_datetime(ratings["timestamp"], unit="s", errors="coerce")
    return movies, ratings


def _compute_rating_statistics(ratings: pd.DataFrame) -> pd.DataFrame:
    stats = (
        ratings.groupby("movieId", dropna=False)["rating"]
        .agg(average_rating="mean", rating_count="count")
        .reset_index()
    )
    stats["average_rating"] = stats["average_rating"].astype(float)
    stats["rating_count"] = stats["rating_count"].astype(int)
    return stats


def _imdb_weighted_rating(
    average_rating: pd.Series,
    rating_count: pd.Series,
    m: float,
    c: float,
) -> pd.Series:
    return (rating_count / (rating_count + m)) * average_rating + (m / (rating_count + m)) * c


def _compute_trending_scores(
    ratings: pd.DataFrame,
    recent_percentile: float = 0.80,
) -> pd.DataFrame:
    if "timestamp" not in ratings.columns or ratings["timestamp"].isna().all():
        return pd.DataFrame(columns=["movieId", "trending_score", "recent_rating_count"])

    recent_threshold = ratings["timestamp"].quantile(recent_percentile)
    recent = ratings[ratings["timestamp"] >= recent_threshold]

    trending = (
        recent.groupby("movieId", dropna=False)["rating"]
        .agg(recent_rating_mean="mean", recent_rating_count="count")
        .reset_index()
    )
    trending["trending_score"] = trending["recent_rating_mean"] * np.log1p(trending["recent_rating_count"])
    return trending[["movieId", "trending_score", "recent_rating_count"]]


def calculate_popularity_scores(
    data_dir: Path,
    top_percentile: float = 0.90,
    min_votes: Optional[int] = None,
) -> pd.DataFrame:
    movies, ratings = _load_movie_and_rating_data(data_dir)
    rating_stats = _compute_rating_statistics(ratings)
    merged = movies.merge(rating_stats, on="movieId", how="left")

    c = merged["average_rating"].mean()
    m = merged["rating_count"].quantile(top_percentile)
    if min_votes is not None:
        m = float(max(m, min_votes))

    merged["weighted_rating"] = _imdb_weighted_rating(
        merged["average_rating"].fillna(0.0),
        merged["rating_count"].fillna(0).astype(float),
        m,
        c,
    )

    merged["rating_count"] = merged["rating_count"].fillna(0).astype(int)
    merged["average_rating"] = merged["average_rating"].fillna(0.0)
    return merged


def _prepare_film_records(frame: pd.DataFrame, columns: List[str]) -> List[Dict[str, object]]:
    return frame[columns].fillna("N/A").to_dict(orient="records")


def top_rated_movies(popularity: pd.DataFrame, top_n: int = 10) -> List[Dict[str, object]]:
    selected = (
        popularity.sort_values(["weighted_rating", "rating_count"], ascending=[False, False])
        .head(top_n)
    )
    return _prepare_film_records(
        selected,
        ["movieId", "title", "genres", "overview", "poster_path", "poster_url", "runtime", "release_date", "vote_average", "imdb_id", "weighted_rating", "average_rating", "rating_count"],
    )


def trending_movies(
    popularity: pd.DataFrame,
    ratings: pd.DataFrame,
    top_n: int = 10,
    recent_percentile: float = 0.80,
) -> List[Dict[str, object]]:
    trending_scores = _compute_trending_scores(ratings, recent_percentile=recent_percentile)
    if trending_scores.empty:
        return top_rated_movies(popularity, top_n)

    merged = popularity.merge(trending_scores, on="movieId", how="left")
    merged["trending_score"] = merged["trending_score"].fillna(0.0)
    return _prepare_film_records(
        merged.sort_values(["trending_score", "rating_count"], ascending=[False, False]).head(top_n),
        ["movieId", "title", "genres", "overview", "poster_path", "poster_url", "runtime", "release_date", "vote_average", "imdb_id", "trending_score", "average_rating", "rating_count"],
    )


def most_rated_movies(popularity: pd.DataFrame, top_n: int = 10) -> List[Dict[str, object]]:
    selected = popularity.sort_values("rating_count", ascending=False).head(top_n)
    return _prepare_film_records(
        selected,
        ["movieId", "title", "genres", "overview", "poster_path", "poster_url", "runtime", "release_date", "vote_average", "imdb_id", "average_rating", "rating_count"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Calculate movie popularity using IMDb weighted rating.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Directory containing MovieLens CSV files.")
    parser.add_argument("--top-n", type=int, default=10, help="Number of movies to return for each category.")
    parser.add_argument("--recent-percentile", type=float, default=0.80, help="Percentile cutoff for trending movies.")
    parser.add_argument("--top-percentile", type=float, default=0.90, help="Percentile cutoff for weighted rating threshold.")
    args = parser.parse_args()

    movies, ratings = _load_movie_and_rating_data(args.data_dir)
    popularity = calculate_popularity_scores(args.data_dir, top_percentile=args.top_percentile)

    print("Top Rated Movies:")
    for row in top_rated_movies(popularity, top_n=args.top_n):
        print(f"- {row['title']} | weighted_rating={row['weighted_rating']:.3f} | avg={row['average_rating']:.3f} | ratings={row['rating_count']}")

    print("\nTrending Movies:")
    for row in trending_movies(popularity, ratings, top_n=args.top_n, recent_percentile=args.recent_percentile):
        print(f"- {row['title']} | trending_score={row['trending_score']:.3f} | ratings={row['rating_count']}")

    print("\nMost Rated Movies:")
    for row in most_rated_movies(popularity, top_n=args.top_n):
        print(f"- {row['title']} | avg={row['average_rating']:.3f} | ratings={row['rating_count']}")


if __name__ == "__main__":
    main()
