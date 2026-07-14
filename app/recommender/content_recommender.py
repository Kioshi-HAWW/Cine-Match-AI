from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import joblib
import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.preprocessing.feature_engineering import create_content_column
from app.preprocessing.load_data import load_movielens_data
from app.preprocessing.metadata import load_enriched_movielens_movies, normalize_json_text

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "archive" / "ml-latest-small"
DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[2] / "trained_models"
DEFAULT_MIN_MATCH_SCORE = 50


@dataclass(frozen=True)
class MovieRecommendation:
    title: str
    genres: str
    overview: str
    poster_path: Optional[str]
    similarity_score: float


class ContentRecommender:
    """Content-based movie recommender with robust fuzzy title matching."""

    def __init__(
        self,
        data_dir: Optional[Union[str, Path]] = None,
        metadata_path: Optional[Union[str, Path]] = None,
        models_dir: Optional[Union[str, Path]] = None,
        min_title_match_score: int = DEFAULT_MIN_MATCH_SCORE,
    ):
        self.data_dir = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
        self.models_dir = Path(models_dir) if models_dir is not None else DEFAULT_MODEL_DIR
        self.min_title_match_score = min_title_match_score

        self._validate_data_dir()
        self.movielens_data = load_movielens_data(self.data_dir)
        self.metadata_path = Path(metadata_path) if metadata_path is not None else None
        self.metadata = self._load_optional_metadata(metadata_path)
        self.movie_meta = self._build_movie_metadata()
        self.movie_titles = self.movie_meta["title"].astype(str).tolist()

        self.vectorizer, self.similarity_matrix = self._build_model()

    def _validate_data_dir(self) -> None:
        if not self.data_dir.exists():
            raise FileNotFoundError(f"MovieLens data directory not found: {self.data_dir}")

    def _load_optional_metadata(self, metadata_path: Optional[Union[str, Path]]) -> pd.DataFrame:
        if metadata_path is None:
            return pd.DataFrame()

        path = Path(metadata_path)
        if not path.exists():
            raise FileNotFoundError(f"Metadata file not found: {path}")

        metadata = pd.read_csv(path, low_memory=False)
        metadata["tmdbId"] = pd.to_numeric(
            metadata.get("tmdbId", metadata.get("tmdb_id", pd.Series(dtype="object"))),
            errors="coerce",
        )
        metadata["poster_path"] = metadata.get("poster_path", "").fillna("").astype(str)
        metadata["overview"] = metadata.get("overview", "").fillna("").astype(str)
        metadata["keywords"] = metadata.get("keywords", "").fillna("").astype(str)
        metadata["genres"] = metadata.get("genres", "").fillna("").astype(str)
        metadata["director"] = metadata.get("director", "").fillna("").astype(str)
        metadata["cast"] = metadata.get("cast", "").fillna("").astype(str)

        metadata["keywords"] = metadata["keywords"].apply(self._normalize_json_text)
        metadata["genres"] = metadata["genres"].apply(self._normalize_json_text)
        metadata["cast"] = metadata["cast"].apply(self._normalize_json_text)
        metadata["director"] = metadata["director"].apply(self._normalize_json_text)

        return metadata

    @staticmethod
    def _normalize_json_text(value: Any) -> str:
        return normalize_json_text(value).lower()

    def _build_movie_metadata(self) -> pd.DataFrame:
        movies = load_enriched_movielens_movies(self.data_dir, self.metadata_path if hasattr(self, "metadata_path") else None)
        for column in ["overview", "poster_path", "keywords", "director", "cast"]:
            if column not in movies.columns:
                movies[column] = ""
        movies["poster_path"] = movies["poster_path"].where(movies["poster_path"].notna(), None)
        movies["content"] = create_content_column(movies).get("content", "").fillna("")
        return movies.reset_index(drop=True)

    def _build_model(self) -> tuple[TfidfVectorizer, Optional[np.ndarray]]:
        tfidf_path = self.models_dir / "tfidf.pkl"
        similarity_path = self.models_dir / "similarity.pkl"
        corpus = self.movie_meta["content"].fillna("")

        if tfidf_path.exists():
            vectorizer = joblib.load(tfidf_path)
            tfidf_matrix = vectorizer.transform(corpus)
        else:
            vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=12000)
            tfidf_matrix = vectorizer.fit_transform(corpus)

        self.tfidf_matrix = tfidf_matrix

        similarity_matrix = None
        if similarity_path.exists():
            try:
                loaded_matrix = joblib.load(similarity_path)
                if isinstance(loaded_matrix, np.ndarray) and loaded_matrix.shape == (tfidf_matrix.shape[0], tfidf_matrix.shape[0]):
                    similarity_matrix = loaded_matrix
            except Exception:
                pass

        return vectorizer, similarity_matrix

    def _find_best_title_match(self, title: str) -> tuple[int, str, float]:
        normalized_title = str(title or "").strip()
        if not normalized_title:
            raise ValueError("Movie title must be a non-empty string.")

        match = process.extractOne(
            normalized_title,
            self.movie_titles,
            scorer=fuzz.token_set_ratio,
            processor=str.lower,
        )
        if match is None:
            raise ValueError(f"No matching movie found for title: {title}")

        matched_title, score, index = match
        if score < self.min_title_match_score:
            raise ValueError(
                f"Movie title '{title}' did not match any known movie titles with sufficient confidence."
            )

        return int(index), matched_title, float(score)

    def recommend_similar(
        self,
        title: str,
        top_n: int = 10,
        min_match_score: Optional[int] = None,
    ) -> List[MovieRecommendation]:
        if min_match_score is not None:
            self.min_title_match_score = min_match_score

        matched_index, _, _ = self._find_best_title_match(title)
        
        if self.similarity_matrix is not None:
            similarity_row = self.similarity_matrix[matched_index]
        else:
            # Calculate cosine similarity on-the-fly for only this specific movie
            similarity_row = cosine_similarity(self.tfidf_matrix[matched_index], self.tfidf_matrix).flatten()

        candidate_indices = np.argsort(similarity_row)[::-1]
        recommendations: List[MovieRecommendation] = []
        for index in candidate_indices:
            if index == matched_index:
                continue
            if len(recommendations) >= top_n:
                break

            row = self.movie_meta.iloc[index]
            recommendations.append(
                MovieRecommendation(
                    title=str(row["title"]),
                    genres=str(row.get("genres", "") or ""),
                    overview=str(row.get("overview", "") or ""),
                    poster_path=row.get("poster_path") if pd.notna(row.get("poster_path")) else None,
                    similarity_score=float(similarity_row[index]),
                )
            )

        return recommendations


def main() -> None:
    parser = argparse.ArgumentParser(description="Recommend similar movies by title.")
    parser.add_argument("title", type=str, help="Movie title to search for.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Directory containing MovieLens CSV files.")
    parser.add_argument("--metadata-path", type=Path, default=None, help="Optional Kaggle metadata CSV file for poster paths.")
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODEL_DIR, help="Directory containing pre-trained TF-IDF and similarity artifacts.")
    parser.add_argument("--top-n", type=int, default=10, help="Number of similar movies to return.")
    parser.add_argument("--min-match-score", type=int, default=DEFAULT_MIN_MATCH_SCORE, help="Minimum fuzzy title match score.")
    args = parser.parse_args()

    recommender = ContentRecommender(
        data_dir=args.data_dir,
        metadata_path=args.metadata_path,
        models_dir=args.models_dir,
        min_title_match_score=args.min_match_score,
    )
    recommendations = recommender.recommend_similar(args.title, top_n=args.top_n)

    print(f"Found {len(recommendations)} similar movies for '{args.title}':")
    for rank, recommendation in enumerate(recommendations, start=1):
        print(
            f"{rank}. {recommendation.title} | genres={recommendation.genres or 'N/A'} "
            f"| score={recommendation.similarity_score:.4f} "
            f"| poster={recommendation.poster_path or 'N/A'}"
        )


if __name__ == "__main__":
    main()
