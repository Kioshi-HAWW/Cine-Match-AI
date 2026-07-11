"""Movie recommender using MovieLens CSV data."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

try:
    from surprise import Dataset, Reader, SVD, NMF, KNNBasic
    SURPRISE_AVAILABLE = True
except ImportError:  # pragma: no cover
    SURPRISE_AVAILABLE = False


class MovieRecommender:
    """A movie recommender built on local MovieLens CSV files and optional Kaggle metadata."""

    DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "archive" / "ml-latest-small"

    def __init__(
        self,
        data_dir: Optional[Union[str, Path]] = None,
        kaggle_metadata_path: Optional[Union[str, Path]] = None,
    ):
        self.data_dir = Path(data_dir) if data_dir is not None else self.DEFAULT_DATA_DIR
        self.movies = self._load_csv(
            "movies.csv",
            {"movieId": int, "title": str, "genres": str},
        )
        self.ratings = self._load_csv(
            "ratings.csv",
            {"userId": int, "movieId": int, "rating": float, "timestamp": int},
        )
        self.tags = self._load_csv(
            "tags.csv",
            {"userId": int, "movieId": int, "tag": str, "timestamp": int},
            optional=True,
        )
        self.links = self._load_csv(
            "links.csv",
            {"movieId": int, "imdbId": str, "tmdbId": str},
            optional=True,
        )

        self.kaggle_metadata_path = self._resolve_kaggle_metadata_path(kaggle_metadata_path)
        self.kaggle_metadata = self._load_kaggle_metadata()
        self.movie_meta = self._build_movie_metadata()
        self.content_vectorizers, self.content_matrices = self._build_content_model()
        self.global_rank = self._build_global_rankings()

        if SURPRISE_AVAILABLE:
            self.surprise_dataset, self.trainset = self._build_surprise_dataset()
            self.collab_models = self._build_collab_models()
        else:
            self.surprise_dataset = None
            self.trainset = None
            self.collab_models = {}

    def _load_csv(self, file_name: str, dtypes: Dict[str, Any], optional: bool = False) -> pd.DataFrame:
        path = self.data_dir / file_name
        if optional and not path.exists():
            return pd.DataFrame(columns=list(dtypes.keys()))

        if not path.exists():
            raise FileNotFoundError(f"Required dataset file not found: {path}")

        return pd.read_csv(path, dtype=dtypes)

    def _resolve_kaggle_metadata_path(self, kaggle_metadata_path: Optional[Union[str, Path]]) -> Optional[Path]:
        if kaggle_metadata_path is not None:
            candidate = Path(kaggle_metadata_path)
            return candidate if candidate.exists() else None

        candidates = [
            self.data_dir / "movies_metadata.csv",
            self.data_dir / "the-movies-dataset" / "movies_metadata.csv",
            self.data_dir / "kaggle" / "movies_metadata.csv",
        ]

        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def _load_kaggle_metadata(self) -> pd.DataFrame:
        if self.kaggle_metadata_path is None:
            return pd.DataFrame()

        base = pd.read_csv(self.kaggle_metadata_path, low_memory=False)
        if "tmdbId" not in base.columns and "tmdb_id" in base.columns:
            base["tmdbId"] = pd.to_numeric(base["tmdb_id"], errors="coerce")
        elif "tmdbId" in base.columns:
            base["tmdbId"] = pd.to_numeric(base["tmdbId"], errors="coerce")

        if "imdbId" not in base.columns and "imdb_id" in base.columns:
            base["imdbId"] = base["imdb_id"].astype(str).str.replace(r"^tt", "", regex=True)

        base["overview"] = base.get("overview", "").fillna("").astype(str)
        base["tagline"] = base.get("tagline", "").fillna("").astype(str)
        base["keywords_text"] = self._normalize_json_field(base.get("keywords", pd.Series(dtype="object")))
        base["kaggle_genres"] = self._normalize_json_field(base.get("genres", pd.Series(dtype="object")))

        links = self.links.copy()
        links["tmdbId"] = pd.to_numeric(links["tmdbId"], errors="coerce")
        if "tmdbId" in base.columns and not base["tmdbId"].isna().all():
            merged = links.merge(base, on="tmdbId", how="left", suffixes=("", "_kaggle"))
        elif "imdbId" in base.columns:
            merged = links.merge(base, on="imdbId", how="left", suffixes=("", "_kaggle"))
        else:
            merged = pd.DataFrame()

        if merged.empty or "movieId" not in merged.columns:
            return pd.DataFrame()

        return merged[["movieId", "overview", "tagline", "keywords_text", "kaggle_genres"]].copy()

    def _normalize_json_field(self, column: pd.Series) -> pd.Series:
        def normalize_value(value: Any) -> str:
            if pd.isna(value):
                return ""
            if isinstance(value, str) and value.strip().startswith("["):
                try:
                    import json

                    items = json.loads(value)
                    if isinstance(items, list):
                        names = []
                        for item in items:
                            if isinstance(item, dict):
                                name = item.get("name") or item.get("keyword") or item.get("title")
                                if name:
                                    names.append(str(name).strip().lower())
                            else:
                                names.append(str(item).strip().lower())
                        return " ".join(sorted({name for name in names if name}))
                except json.JSONDecodeError:
                    pass
            if isinstance(value, str):
                return value.strip().replace("|", " ").lower()
            return str(value)

        return pd.Series([normalize_value(v) for v in column], index=column.index)

    def _build_movie_metadata(self) -> pd.DataFrame:
        movies = self.movies.copy()
        movies["genre_text"] = movies["genres"].fillna("").str.replace(r"\|", " ", regex=True)

        if not self.tags.empty:
            tags = (
                self.tags.groupby("movieId")["tag"]
                .apply(
                    lambda values: " ".join(
                        sorted({str(v).strip().lower() for v in values if pd.notna(v) and str(v).strip()})
                    )
                )
                .reset_index()
            )
        else:
            tags = pd.DataFrame({"movieId": [], "tag": []})

        movies = movies.merge(tags.rename(columns={"tag": "tags_text"}), on="movieId", how="left")
        movies["tags_text"] = movies["tags_text"].fillna("")

        if not self.kaggle_metadata.empty:
            movies = movies.merge(self.kaggle_metadata, on="movieId", how="left")
            movies["overview"] = movies["overview"].fillna("")
            movies["tagline"] = movies["tagline"].fillna("")
            movies["keywords_text"] = movies["keywords_text"].fillna("")
            movies["kaggle_genres"] = movies["kaggle_genres"].fillna("")
        else:
            movies["overview"] = ""
            movies["tagline"] = ""
            movies["keywords_text"] = ""
            movies["kaggle_genres"] = ""

        movies["content_text"] = (
            movies["title"].fillna("")
            + " "
            + movies["genre_text"]
            + " "
            + movies["overview"]
            + " "
            + movies["tagline"]
            + " "
            + movies["keywords_text"]
            + " "
            + movies["kaggle_genres"]
            + " "
            + movies["tags_text"]
        ).str.strip()
        movies["content_text"] = movies["content_text"].replace(r"\s+", " ", regex=True)
        return movies

        if not self.tags.empty:
            tags = (
                self.tags.groupby("movieId")["tag"]
                .apply(
                    lambda values: " ".join(
                        sorted({str(v).strip().lower() for v in values if pd.notna(v) and str(v).strip()})
                    )
                )
                .reset_index()
            )
        else:
            tags = pd.DataFrame({"movieId": [], "tag": []})

        movies = movies.merge(tags.rename(columns={"tag": "tags_text"}), on="movieId", how="left")
        movies["tags_text"] = movies["tags_text"].fillna("")
        movies["content_text"] = (
            movies["title"].fillna("") + " " + movies["genre_text"] + " " + movies["tags_text"]
        ).str.strip()
        movies["content_text"] = movies["content_text"].replace(r"\s+", " ", regex=True)
        return movies

    def _build_content_model(self) -> tuple[Dict[str, Any], Dict[str, Any]]:
        raw_content = self.movie_meta["content_text"].fillna("")
        tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=12000)
        count = CountVectorizer(stop_words="english", ngram_range=(1, 2), max_features=12000)
        return {
            "tfidf": tfidf,
            "count": count,
        }, {
            "tfidf": tfidf.fit_transform(raw_content),
            "count": count.fit_transform(raw_content),
        }

    def _build_global_rankings(self) -> pd.DataFrame:
        ranking = (
            self.ratings.groupby("movieId")["rating"]
            .agg(average_rating="mean", rating_count="count")
            .reset_index()
        )
        ranking = ranking.merge(self.movies[["movieId", "title", "genres"]], on="movieId", how="left")
        ranking["popularity_score"] = ranking["average_rating"] * np.log1p(ranking["rating_count"])
        return ranking.sort_values(["average_rating", "rating_count"], ascending=False).reset_index(drop=True)

    def _build_surprise_dataset(self):
        reader = Reader(rating_scale=(0.5, 5.0))
        data = Dataset.load_from_df(self.ratings[["userId", "movieId", "rating"]], reader)
        trainset = data.build_full_trainset()
        return data, trainset

    def _build_collab_models(self) -> Dict[str, Any]:
        models: Dict[str, Any] = {
            "SVD": SVD(n_factors=50, random_state=42),
            "NMF": NMF(n_factors=20, random_state=42),
            "KNNBasic": KNNBasic(sim_options={"name": "cosine", "user_based": True}, verbose=False),
        }
        for algo in models.values():
            algo.fit(self.trainset)
        return models

    def _build_recommendation_records(self, movie_ids: List[int], scores: List[float]) -> List[Dict[str, Any]]:
        frame = pd.DataFrame({"movieId": movie_ids, "score": scores})
        frame = frame.merge(self.movies[["movieId", "title", "genres"]], on="movieId", how="left")
        return frame.drop_duplicates(subset=["movieId"]).sort_values("score", ascending=False).to_dict(orient="records")

    def recommend_by_interest(self, user_interest: str, top_n: int = 10, vectorizer: str = "tfidf") -> List[Dict[str, Any]]:
        user_interest = str(user_interest or "").strip()
        if len(user_interest) < 3:
            return []

        if vectorizer not in self.content_matrices:
            raise ValueError("vectorizer must be 'tfidf' or 'count'.")

        query = self.content_vectorizers[vectorizer].transform([user_interest])
        scores = linear_kernel(query, self.content_matrices[vectorizer]).flatten()
        if scores.max() <= 0:
            return []

        top_indices = np.argsort(scores)[::-1][: top_n * 2]
        selected = [int(idx) for idx in top_indices if scores[idx] > 0][:top_n]
        return self._build_recommendation_records(self.movie_meta.iloc[selected]["movieId"].tolist(), [float(scores[idx]) for idx in selected])

    def recommend_by_user(self, user_id: int, top_n: int = 10, model_name: str = "SVD") -> List[Dict[str, Any]]:
        if not SURPRISE_AVAILABLE:
            raise RuntimeError("Collaborative filtering requires scikit-surprise.")

        if model_name not in self.collab_models:
            raise ValueError(f"Unsupported model_name '{model_name}'. Choose from {list(self.collab_models)}.")

        if user_id not in set(self.ratings["userId"]):
            return []

        seen = set(self.ratings[self.ratings["userId"] == user_id]["movieId"])
        unseen_movie_ids = [int(mid) for mid in self.movies["movieId"].tolist() if mid not in seen]

        predictions = [self.collab_models[model_name].predict(user_id, movie_id) for movie_id in unseen_movie_ids]
        predictions.sort(key=lambda item: item.est, reverse=True)
        best = predictions[:top_n]
        return self._build_recommendation_records([int(pred.iid) for pred in best], [float(pred.est) for pred in best])

    def recommend_fallback(self, top_n: int = 10) -> List[Dict[str, Any]]:
        top = self.global_rank.head(top_n).copy()
        top["score"] = top["average_rating"]
        return top[["movieId", "title", "genres", "score", "rating_count"]].to_dict(orient="records")

    def recommend(self, user_interest: Optional[str] = None, user_id: Optional[int] = None, top_n: int = 10, model_name: str = "SVD") -> Dict[str, Any]:
        if user_interest and len(str(user_interest).strip()) >= 3:
            content_recs = self.recommend_by_interest(user_interest, top_n)
            if content_recs:
                return {
                    "type": "content",
                    "source": "interest",
                    "query": user_interest,
                    "recommendations": content_recs,
                }

        if user_id is not None and SURPRISE_AVAILABLE:
            collab_recs = self.recommend_by_user(user_id, top_n, model_name)
            if collab_recs:
                return {
                    "type": "collaborative",
                    "source": "user_id",
                    "user_id": user_id,
                    "model": model_name,
                    "recommendations": collab_recs,
                }

        return {
            "type": "fallback",
            "source": "top_rated",
            "recommendations": self.recommend_fallback(top_n),
        }

    def recommend_hybrid(self, user_interest: str, user_id: int, top_n: int = 10, model_name: str = "SVD", alpha: float = 0.5) -> Dict[str, Any]:
        if not user_interest or not user_id or not SURPRISE_AVAILABLE:
            return self.recommend(user_interest, user_id, top_n, model_name)

        content_recs = self.recommend_by_interest(user_interest, top_n * 3)
        collab_recs = self.recommend_by_user(user_id, top_n * 3, model_name)
        if not content_recs or not collab_recs:
            return self.recommend(user_interest, user_id, top_n, model_name)

        content_df = pd.DataFrame(content_recs)[["movieId", "score"]].rename(columns={"score": "content_score"})
        collab_df = pd.DataFrame(collab_recs)[["movieId", "score"]].rename(columns={"score": "collab_score"})
        merged = content_df.merge(collab_df, on="movieId", how="inner")
        if merged.empty:
            return self.recommend(user_interest, user_id, top_n, model_name)

        merged["content_score_norm"] = (merged["content_score"] - merged["content_score"].min()) / (merged["content_score"].ptp() + 1e-9)
        merged["collab_score_norm"] = (merged["collab_score"] - merged["collab_score"].min()) / (merged["collab_score"].ptp() + 1e-9)
        merged["score"] = alpha * merged["content_score_norm"] + (1 - alpha) * merged["collab_score_norm"]
        merged = merged.sort_values("score", ascending=False).head(top_n)

        return {
            "type": "hybrid",
            "user_id": user_id,
            "query": user_interest,
            "model": model_name,
            "alpha": alpha,
            "recommendations": self._build_recommendation_records(merged["movieId"].tolist(), merged["score"].tolist()),
        }


def _print_recommendations(title: str, records: list[Dict[str, Any]]) -> None:
    print(title)
    for record in records:
        print(f"- {record['title']} ({record.get('genres','N/A')}) score={record['score']:.3f}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the MovieLens recommender with optional metadata.")
    parser.add_argument("--interest", type=str, default="", help="User interest text for content-based recommendation.")
    parser.add_argument("--user-id", type=int, default=None, help="MovieLens user ID for collaborative recommendation.")
    parser.add_argument("--top-n", type=int, default=10, help="Number of recommendations to return.")
    parser.add_argument("--model", type=str, default="SVD", choices=["SVD", "NMF", "KNNBasic"], help="Collaborative model to use.")
    parser.add_argument("--kaggle-metadata", type=str, default=None, help="Optional Kaggle movies_metadata.csv path.")
    args = parser.parse_args()

    recommender = MovieRecommender(kaggle_metadata_path=args.kaggle_metadata)

    if args.interest and len(args.interest.strip()) >= 3:
        recs = recommender.recommend_by_interest(args.interest, top_n=args.top_n)
        _print_recommendations(f"Content-based recommendations for '{args.interest}':", recs)
        return

    if args.user_id is not None and SURPRISE_AVAILABLE:
        recs = recommender.recommend_by_user(args.user_id, top_n=args.top_n, model_name=args.model)
        _print_recommendations(f"Collaborative recommendations for user {args.user_id} using {args.model}:", recs)
        return

    fallback = recommender.recommend_fallback(top_n=args.top_n)
    _print_recommendations("Fallback top-rated recommendations:", fallback)


if __name__ == "__main__":
    main()
