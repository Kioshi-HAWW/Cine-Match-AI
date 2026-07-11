from __future__ import annotations

import argparse
from pathlib import Path

import joblib
from surprise import Dataset, KNNBasic, Reader

from app.preprocessing.load_data import load_movielens_data

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "trained_models"


def train_knn_model(data_dir: Path, output_dir: Path = DEFAULT_OUTPUT_DIR) -> Path:
    ratings = load_movielens_data(data_dir)["ratings"].copy()
    reader = Reader(rating_scale=(ratings["rating"].min(), ratings["rating"].max()))
    dataset = Dataset.load_from_df(ratings[["userId", "movieId", "rating"]], reader)
    trainset = dataset.build_full_trainset()

    algo = KNNBasic(sim_options={"name": "cosine", "user_based": True}, verbose=False)
    algo.fit(trainset)

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "knn.pkl"
    joblib.dump(algo, model_path)
    return model_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the KNNBasic collaborative filtering model.")
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parents[2] / "archive" / "ml-latest-small", help="Root path to MovieLens datasets.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory to save the trained model.")
    args = parser.parse_args()

    model_path = train_knn_model(args.data_dir, args.output_dir)
    print(f"Saved KNN model to: {model_path}")


if __name__ == "__main__":
    main()
