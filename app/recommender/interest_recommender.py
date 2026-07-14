from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.preprocessing.feature_engineering import create_content_column
from app.preprocessing.metadata import load_enriched_movielens_movies, normalize_json_text

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "archive" / "ml-latest-small"


class InterestRecommender:
    """Recommend movies based on a user interest query."""

    def __init__(
        self,
        data_dir: Optional[Union[str, Path]] = None,
        metadata_path: Optional[Union[str, Path]] = None,
        max_features: int = 12000,
    ):
        self.data_dir = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
        self.metadata_path = Path(metadata_path) if metadata_path is not None else None
        self.movies = self._build_movie_metadata()
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=max_features)
        self.content_matrix = self.vectorizer.fit_transform(self.movies["content"].fillna(""))

    def _normalize_json_text(self, value: Any) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, str):
            trimmed = value.strip()
            if trimmed.startswith("["):
                try:
                    decoded = json.loads(trimmed)
                except json.JSONDecodeError:
                    return trimmed.lower()
                tokens: List[str] = []
                for item in decoded:
                    if isinstance(item, dict):
                        name = item.get("name") or item.get("title") or item.get("keyword")
                        if name:
                            tokens.append(str(name).strip().lower())
                    elif isinstance(item, str):
                        tokens.append(item.strip().lower())
                return " ".join(sorted({token for token in tokens if token}))
            return trimmed.lower()
        return str(value).strip().lower()

    def _build_movie_metadata(self) -> pd.DataFrame:
        movies = load_enriched_movielens_movies(self.data_dir, self.metadata_path)
        movies["overview"] = movies.get("overview", "").fillna("").astype(str)
        movies["genres"] = movies.get("genres", "").fillna("").astype(str)
        movies["poster_path"] = movies.get("poster_path", None)
        movies["poster_url"] = movies.get("poster_url", movies["poster_path"])
        movies["content"] = create_content_column(movies[["movieId", "title", "genres", "overview"]])["content"].fillna("")
        return movies.reset_index(drop=True)

    def recommend(self, interest: str, top_n: int = 10) -> List[Dict[str, Any]]:
        interest_text = str(interest or "").strip()
        if not interest_text:
            return []

        query_vector = self.vectorizer.transform([interest_text])
        scores = cosine_similarity(query_vector, self.content_matrix).flatten()
        top_indices = np.argsort(scores)[::-1][:top_n]

        recommendations: List[Dict[str, Any]] = []
        for index in top_indices:
            recommendations.append(
                {
                    "title": str(self.movies.at[index, "title"]),
                    "genres": str(self.movies.at[index, "genres"]),
                    "overview": str(self.movies.at[index, "overview"]),
                    "poster_path": self.movies.at[index, "poster_path"] if pd.notna(self.movies.at[index, "poster_path"]) else None,
                    "poster_url": self.movies.at[index, "poster_url"] if pd.notna(self.movies.at[index, "poster_url"]) else None,
                    "runtime": float(self.movies.at[index, "runtime"]) if "runtime" in self.movies.columns and pd.notna(self.movies.at[index, "runtime"]) else None,
                    "release_date": str(self.movies.at[index, "release_date"]) if "release_date" in self.movies.columns else "",
                    "vote_average": float(self.movies.at[index, "vote_average"]) if "vote_average" in self.movies.columns and pd.notna(self.movies.at[index, "vote_average"]) else None,
                    "imdb_id": str(self.movies.at[index, "imdb_id"]) if "imdb_id" in self.movies.columns else "",
                    "similarity_score": float(scores[index]),
                }
            )

        return recommendations


def main() -> None:
    parser = argparse.ArgumentParser(description="Recommend movies based on a user interest query.")
    parser.add_argument("interest", type=str, help="User interest text for recommendation.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Directory containing MovieLens CSV files.")
    parser.add_argument("--metadata-path", type=Path, default=None, help="Optional metadata CSV file for enriched overview and genres.")
    parser.add_argument("--top-n", type=int, default=10, help="Number of recommendations to return.")
    args = parser.parse_args()

    recommender = InterestRecommender(
        data_dir=args.data_dir,
        metadata_path=args.metadata_path,
    )
    results = recommender.recommend(args.interest, top_n=args.top_n)

    print(f"Top {len(results)} movies matching: {args.interest}")
    for rank, movie in enumerate(results, start=1):
        print(
            f"{rank}. {movie['title']} | score={movie['similarity_score']:.4f} "
            f"| genres={movie['genres'] or 'N/A'} "
            f"| overview={movie['overview'][:180].strip() + ('...' if len(movie['overview']) > 180 else '')}"
        )


if __name__ == "__main__":
    main()
