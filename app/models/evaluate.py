from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from surprise import Dataset, Reader, accuracy
from surprise.model_selection import train_test_split

from app.preprocessing.load_data import load_movielens_data

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "trained_models"
MODEL_FILES = {
    "SVD": "svd.pkl",
    "NMF": "nmf.pkl",
    "KNNBasic": "knn.pkl",
}


def evaluate_models(data_dir: Path, model_dir: Path = DEFAULT_OUTPUT_DIR, test_size: float = 0.2, random_state: int = 42) -> dict[str, tuple[float, float]]:
    ratings = load_movielens_data(data_dir)["ratings"].copy()
    reader = Reader(rating_scale=(ratings["rating"].min(), ratings["rating"].max()))
    dataset = Dataset.load_from_df(ratings[["userId", "movieId", "rating"]], reader)
    _, testset = train_test_split(dataset, test_size=test_size, random_state=random_state)

    results: dict[str, tuple[float, float]] = {}
    for model_name, filename in MODEL_FILES.items():
        model_path = model_dir / filename
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found for {model_name}: {model_path}")

        algo = joblib.load(model_path)
        predictions = algo.test(testset)
        rmse = accuracy.rmse(predictions, verbose=False)
        mae = accuracy.mae(predictions, verbose=False)
        results[model_name] = (rmse, mae)

    return results


def choose_best_model(metrics: dict[str, tuple[float, float]]) -> str:
    return min(metrics.items(), key=lambda item: item[1][0])[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained collaborative filtering models and choose the best model.")
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parents[2] / "archive" / "ml-latest-small", help="Root path to MovieLens datasets.")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory containing trained model pickle files.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Proportion of ratings held out for evaluation.")
    parser.add_argument("--random-state", type=int, default=42, help="Random state for train/test split.")
    args = parser.parse_args()

    results = evaluate_models(args.data_dir, args.model_dir, args.test_size, args.random_state)
    best_model = choose_best_model(results)

    print("Evaluation results:")
    for model_name, (rmse, mae) in results.items():
        print(f"- {model_name}: RMSE={rmse:.4f}, MAE={mae:.4f}")
    print(f"Best collaborative model by RMSE: {best_model}")


if __name__ == "__main__":
    main()
