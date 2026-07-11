from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import pandas as pd
from surprise import Dataset, KNNBasic, NMF, Reader, SVD, accuracy
from surprise.model_selection import train_test_split

from app.preprocessing.load_data import load_movielens_data

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "trained_models"
MODEL_FILES: dict[str, str] = {
    "SVD": "svd.pkl",
    "NMF": "nmf.pkl",
    "KNNBasic": "knn.pkl",
}


def load_ratings(data_dir: Path) -> pd.DataFrame:
    """Load MovieLens ratings into a DataFrame."""
    ratings = load_movielens_data(data_dir)["ratings"].copy()
    if ratings.empty:
        raise ValueError(f"No ratings data found in {data_dir}")
    return ratings


def build_surprise_dataset(ratings: pd.DataFrame) -> Tuple[Dataset, Reader]:
    """Build a Surprise dataset from ratings data."""
    reader = Reader(rating_scale=(ratings["rating"].min(), ratings["rating"].max()))
    return Dataset.load_from_df(ratings[["userId", "movieId", "rating"]], reader), reader


def get_collaborative_models() -> dict[str, Any]:
    """Return a mapping of collaborative filtering algorithms to train."""
    return {
        "SVD": SVD(n_factors=50, random_state=42),
        "NMF": NMF(n_factors=20, random_state=42),
        "KNNBasic": KNNBasic(sim_options={"name": "cosine", "user_based": True}, verbose=False),
    }


def train_model(algo_name: str, algo: Any, trainset: Any) -> Any:
    """Train a single Surprise algorithm on the given trainset."""
    algo.fit(trainset)
    return algo


def save_model(algo: Any, model_path: Path) -> Path:
    """Save a trained model to disk using joblib."""
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(algo, model_path)
    return model_path


def evaluate_model(algo: Any, testset: Any) -> Tuple[float, float]:
    """Evaluate a trained Surprise model on the provided testset and return RMSE and MAE."""
    predictions = algo.test(testset)
    rmse = accuracy.rmse(predictions, verbose=False)
    mae = accuracy.mae(predictions, verbose=False)
    return rmse, mae


def train_and_evaluate_collaborative_models(
    data_dir: Path,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, tuple[float, float]]:
    """Train the collaborative models and evaluate them using RMSE and MAE."""
    ratings = load_ratings(data_dir)
    dataset, _ = build_surprise_dataset(ratings)
    trainset, testset = train_test_split(dataset, test_size=test_size, random_state=random_state)

    results: dict[str, tuple[float, float]] = {}
    for model_name, algo in get_collaborative_models().items():
        trained_algo = train_model(model_name, algo, trainset)
        model_path = output_dir / MODEL_FILES[model_name]
        save_model(trained_algo, model_path)
        results[model_name] = evaluate_model(trained_algo, testset)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate collaborative recommendation models using MovieLens ratings.")
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
        help="Directory to save trained model artifacts.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of ratings held out for evaluation.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random state for the train/test split.",
    )
    args = parser.parse_args()

    results = train_and_evaluate_collaborative_models(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    print("Saved trained collaborative models to:")
    for model_name, filename in MODEL_FILES.items():
        print(f"- {model_name}: {args.output_dir / filename}")

    print("\nEvaluation results:")
    for model_name, (rmse, mae) in results.items():
        print(f"- {model_name}: RMSE={rmse:.4f}, MAE={mae:.4f}")


if __name__ == "__main__":
    main()
