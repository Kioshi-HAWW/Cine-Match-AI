from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import KNNBasic, NMF, SVD

from app.preprocessing.feature_engineering import create_content_column
from app.preprocessing.load_data import load_movielens_data
from app.preprocessing.metadata import load_enriched_movielens_movies

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "archive" / "ml-latest-small"
DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[2] / "trained_models"
DEFAULT_MODEL_NAME = "SVD"
WEIGHTS = {
    "content": 0.45,
    "collaborative": 0.45,
    "popularity": 0.10,
}
MODEL_FILES: dict[str, str] = {
    "SVD": "svd.pkl",
    "NMF": "nmf.pkl",
    "KNNBasic": "knn.pkl",
}


class HybridRecommender:
    """Hybrid recommender combining content, collaborative, and popularity signals."""

    def __init__(
        self,
        data_dir: Optional[Union[str, Path]] = None,
        models_dir: Optional[Union[str, Path]] = None,
        model_name: str = DEFAULT_MODEL_NAME,
        max_features: int = 12000,
    ):
        self.data_dir = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
        self.models_dir = Path(models_dir) if models_dir is not None else DEFAULT_MODEL_DIR
        self.model_name = model_name
        self.max_features = max_features

        self._ensure_valid_model_name()
        self.ratings = self._load_ratings()
        self.movies = self._build_movie_metadata()
        self.popularity_score = self._build_popularity_score()
        self.vectorizer, self.content_matrix = self._build_content_model()
        self.collab_algo = self._load_collaborative_model()

    def _ensure_valid_model_name(self) -> None:
        if self.model_name not in MODEL_FILES:
            raise ValueError(f"Unsupported model_name '{self.model_name}'. Choose from {list(MODEL_FILES)}.")

    def _load_ratings(self) -> pd.DataFrame:
        ratings = load_movielens_data(self.data_dir)["ratings"].copy()
        if ratings.empty:
            raise ValueError(f"No ratings data found in {self.data_dir}")
        return ratings

    def _load_optional_metadata(self, metadata_path: Optional[Union[str, Path]]) -> pd.DataFrame:
        if metadata_path is None:
            return pd.DataFrame()

        metadata = pd.read_csv(Path(metadata_path), low_memory=False)
        if "tmdbId" not in metadata.columns and "tmdb_id" in metadata.columns:
            metadata["tmdbId"] = pd.to_numeric(metadata["tmdb_id"], errors="coerce")
        else:
            metadata["tmdbId"] = pd.to_numeric(metadata.get("tmdbId", pd.Series(dtype="object")), errors="coerce")

        metadata["overview"] = metadata.get("overview", "").fillna("").astype(str)
        metadata["genres"] = metadata.get("genres", "").fillna("").astype(str)
        return metadata

    def _build_movie_metadata(self) -> pd.DataFrame:
        movies = load_enriched_movielens_movies(self.data_dir).copy()
        movies["title"] = movies["title"].fillna("").astype(str)
        movies["genres"] = movies["genres"].fillna("").astype(str)
        movies["overview"] = movies.get("overview", "").fillna("").astype(str)
        movies["content"] = create_content_column(movies[["movieId", "title", "genres", "overview"]])["content"].fillna("")
        return movies.reset_index(drop=True)

    def _build_content_model(self) -> Tuple[TfidfVectorizer, np.ndarray]:
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=self.max_features)
        content_matrix = vectorizer.fit_transform(self.movies["content"].fillna(""))
        return vectorizer, content_matrix

    def _build_popularity_score(self) -> np.ndarray:
        rating_stats = (
            self.ratings.groupby("movieId")["rating"]
            .agg(average_rating="mean", rating_count="count")
            .reset_index()
        )
        rating_stats["popularity"] = rating_stats["average_rating"] * np.log1p(rating_stats["rating_count"])
        merged = self.movies[["movieId"]].merge(rating_stats[["movieId", "popularity"]], on="movieId", how="left")
        popularity = merged["popularity"].fillna(0.0).to_numpy(dtype=float)
        return self._normalize(popularity)

    def _load_collaborative_model(self) -> Any:
        model_path = self.models_dir / MODEL_FILES[self.model_name]
        if not model_path.exists():
            raise FileNotFoundError(
                f"Collaborative model not found: {model_path}. Train models with app/models/train_collaborative.py."\
            )
        return joblib.load(model_path)

    @staticmethod
    def _normalize(values: np.ndarray) -> np.ndarray:
        if values.size == 0:
            return values
        min_val = float(np.nanmin(values))
        max_val = float(np.nanmax(values))
        if np.isclose(max_val, min_val):
            return np.zeros_like(values)
        normalized = (values - min_val) / (max_val - min_val)
        return np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)

    def _predict_collaborative_scores(self, user_id: int) -> np.ndarray:
        movie_ids = self.movies["movieId"].astype(int).tolist()
        scores = []
        for movie_id in movie_ids:
            prediction = self.collab_algo.predict(user_id, movie_id)
            scores.append(float(prediction.est))
        return self._normalize(np.array(scores, dtype=float))

    def recommend(
        self,
        interest: str,
        user_id: int,
        top_n: int = 10,
    ) -> List[Dict[str, Any]]:
        interest_text = str(interest or "").strip()
        if not interest_text:
            raise ValueError("Interest text must be provided.")

        query = self.vectorizer.transform([interest_text])
        content_scores = cosine_similarity(query, self.content_matrix).flatten()
        content_scores = self._normalize(content_scores)
        collaborative_scores = self._predict_collaborative_scores(user_id)
        popularity_scores = self.popularity_score

        combined_scores = (
            WEIGHTS["content"] * content_scores
            + WEIGHTS["collaborative"] * collaborative_scores
            + WEIGHTS["popularity"] * popularity_scores
        )

        recommendations = self.movies.copy()
        recommendations["content_score"] = content_scores
        recommendations["collaborative_score"] = collaborative_scores
        recommendations["popularity_score"] = popularity_scores
        recommendations["combined_score"] = combined_scores

        recommendations = recommendations.sort_values("combined_score", ascending=False)
        recommendations = recommendations.drop_duplicates(subset=["movieId"])
        top_recommendations = recommendations.head(top_n)

        return [
            {
                "title": row["title"],
                "genres": row["genres"],
                "overview": row["overview"],
                "poster_path": row.get("poster_path") if pd.notna(row.get("poster_path")) else None,
                "poster_url": row.get("poster_url") if pd.notna(row.get("poster_url")) else None,
                "runtime": float(row["runtime"]) if pd.notna(row.get("runtime")) else None,
                "release_date": str(row.get("release_date", "") or ""),
                "vote_average": float(row["vote_average"]) if pd.notna(row.get("vote_average")) else None,
                "imdb_id": str(row.get("imdb_id", "") or ""),
                "content_score": float(row["content_score"]),
                "collaborative_score": float(row["collaborative_score"]),
                "popularity_score": float(row["popularity_score"]),
                "combined_score": float(row["combined_score"]),
            }
            for _, row in top_recommendations.iterrows()
        ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid recommendation combining content, collaborative, and popularity signals.")
    parser.add_argument("interest", type=str, help="User interest text for recommendation.")
    parser.add_argument("--user-id", type=int, required=True, help="MovieLens user ID for collaborative scoring.")
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME, choices=list(MODEL_FILES), help="Collaborative model to use.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Directory containing MovieLens CSV files.")
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODEL_DIR, help="Directory containing pre-trained collaborative models.")
    parser.add_argument("--top-n", type=int, default=10, help="Number of recommendations to return.")
    args = parser.parse_args()

    recommender = HybridRecommender(
        data_dir=args.data_dir,
        models_dir=args.models_dir,
        model_name=args.model_name,
    )
    recommendations = recommender.recommend(args.interest, user_id=args.user_id, top_n=args.top_n)

    print(f"Top {len(recommendations)} hybrid recommendations for user {args.user_id} and interest '{args.interest}':")
    for rank, rec in enumerate(recommendations, start=1):
        print(
            f"{rank}. {rec['title']} | score={rec['combined_score']:.4f} "
            f"| genres={rec['genres'] or 'N/A'} "
            f"| content={rec['content_score']:.4f} "
            f"| collab={rec['collaborative_score']:.4f} "
            f"| pop={rec['popularity_score']:.4f}"
        )
        print(f"    overview={rec['overview'][:200].strip() + ('...' if len(rec['overview']) > 200 else '')}")


if __name__ == "__main__":
    main()
