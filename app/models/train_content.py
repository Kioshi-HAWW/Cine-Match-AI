from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.preprocessing.feature_engineering import create_content_column
from app.preprocessing.load_data import load_movielens_data

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "trained_models"


def prepare_content_dataframe(movies: pd.DataFrame) -> pd.DataFrame:
    """Generate a movie dataframe with a cleaned content column."""
    movies = movies.copy()
    movies = create_content_column(movies)
    if "content" not in movies.columns:
        raise ValueError("The movies dataframe must contain a 'content' column after feature engineering.")
    movies["content"] = movies["content"].fillna("")
    return movies


def build_tfidf_matrix(content_series: pd.Series, max_features: int = 12000) -> Tuple[TfidfVectorizer, object]:
    """Fit a TF-IDF vectorizer on text content and return the transformer and matrix."""
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=max_features)
    matrix = vectorizer.fit_transform(content_series)
    return vectorizer, matrix


def build_similarity_matrix(tfidf_matrix: object) -> object:
    """Compute cosine similarity matrix from a TF-IDF matrix."""
    return cosine_similarity(tfidf_matrix, tfidf_matrix)


def save_model(object_to_save: object, path: Path) -> Path:
    """Persist a Python object to disk with joblib."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(object_to_save, path)
    return path


def train_content_model(
    data_dir: Path,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    save_similarity: bool = False,
) -> Tuple[Path, Optional[Path]]:
    """Train a content-based recommendation model and save artifacts."""
    data = load_movielens_data(data_dir)
    movies = prepare_content_dataframe(data["movies"])

    vectorizer, tfidf_matrix = build_tfidf_matrix(movies["content"])
    tfidf_path = output_dir / "tfidf.pkl"
    save_model(vectorizer, tfidf_path)

    similarity_path = None
    if save_similarity:
        similarity_matrix = build_similarity_matrix(tfidf_matrix)
        similarity_path = output_dir / "similarity.pkl"
        save_model(similarity_matrix, similarity_path)

    return tfidf_path, similarity_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the content-based TF-IDF model and similarity matrix.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "archive" / "ml-latest-small",
        help="Root path to MovieLens datasets.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save trained models.",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=12000,
        help="Maximum number of TF-IDF features.",
    )
    parser.add_argument(
        "--save-similarity",
        action="store_true",
        help="Whether to compute and save the dense 759MB similarity matrix (skip for low memory/Render free tier).",
    )
    args = parser.parse_args()

    tfidf_path, similarity_path = train_content_model(
        args.data_dir, args.output_dir, save_similarity=args.save_similarity
    )
    print(f"Saved TF-IDF vectorizer to: {tfidf_path}")
    if similarity_path:
        print(f"Saved similarity matrix to: {similarity_path}")
    else:
        print("Skipped saving similarity matrix (run with --save-similarity to generate).")


if __name__ == "__main__":
    main()
