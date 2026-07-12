from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional

import pandas as pd

MOVIELENS_FILE_COLUMNS: Dict[str, Iterable[str]] = {
    "movies": ["movieId", "title", "genres"],
    "ratings": ["userId", "movieId", "rating"],
    "tags": ["userId", "movieId", "tag"],
    "links": ["movieId", "imdbId", "tmdbId"],
}

TMDB_FILE_COLUMNS: Dict[str, Iterable[str]] = {
    "movies_metadata": ["id", "title", "overview", "genres"],
    "credits": ["id", "cast", "crew"],
    "keywords": ["id", "keywords"],
}


class DatasetLoadingError(Exception):
    """Base exception for dataset loading failures."""


class MissingDatasetError(DatasetLoadingError):
    """Raised when a required dataset file does not exist."""


class InvalidDatasetSchemaError(DatasetLoadingError):
    """Raised when a dataset file is missing required columns."""


def get_data_root(data_dir: Optional[Path] = None) -> Path:
    """Return the root path for dataset files.

    Args:
        data_dir: Optional override for the root dataset directory.

    Returns:
        Path to the dataset root.
    """
    if data_dir is not None:
        return Path(data_dir)
    return Path(__file__).resolve().parents[2] / "data"


def _load_csv(path: Path, required_columns: Iterable[str], required: bool = True, low_memory: bool = False) -> Optional[pd.DataFrame]:
    """Load a CSV file and validate required columns.

    Args:
        path: Full path to the CSV file.
        required_columns: Column names expected in the CSV.
        required: If True, missing files raise an exception.
        low_memory: Whether pandas should use the low_memory option.

    Returns:
        Loaded DataFrame, or None when a non-required file is absent.

    Raises:
        MissingDatasetError: When a required file is missing.
        InvalidDatasetSchemaError: When required columns are absent.
    """
    if not path.exists():
        if required:
            raise MissingDatasetError(f"Required dataset file not found: {path}")
        return None

    df = pd.read_csv(path, low_memory=low_memory)
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise InvalidDatasetSchemaError(
            f"CSV file '{path.name}' is missing required columns: {', '.join(missing_columns)}"
        )

    return df


def load_movielens_data(data_dir: Optional[Path] = None) -> Dict[str, pd.DataFrame]:
    """Load MovieLens dataset CSV files.

    Args:
        data_dir: Root directory containing the MovieLens dataset folder or CSVs.

    Returns:
        Dictionary containing DataFrames for movies, ratings, tags, and links.
    """
    # Resolve relative paths against the project root (two levels above this file)
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if data_dir is not None:
        resolved = Path(data_dir)
        if not resolved.is_absolute():
            resolved = PROJECT_ROOT / resolved
    else:
        resolved = PROJECT_ROOT / "archive" / "ml-latest-small"

    root = resolved
    movielens_dir = root / "movielens"
    if not movielens_dir.exists():
        movielens_dir = root

    # movies and ratings are required; tags and links are optional
    required_files = {"movies": MOVIELENS_FILE_COLUMNS["movies"], "ratings": MOVIELENS_FILE_COLUMNS["ratings"]}
    optional_files = {"tags": MOVIELENS_FILE_COLUMNS["tags"], "links": MOVIELENS_FILE_COLUMNS["links"]}

    result: Dict[str, pd.DataFrame] = {}
    for name, columns in required_files.items():
        df = _load_csv(movielens_dir / f"{name}.csv", columns, required=True)
        if df is not None:
            result[name] = df
    for name, columns in optional_files.items():
        df = _load_csv(movielens_dir / f"{name}.csv", columns, required=False)
        if df is not None:
            result[name] = df

    return result


def load_tmdb_data(data_dir: Optional[Path] = None) -> Dict[str, pd.DataFrame]:
    """Load TMDb dataset CSV files.

    Args:
        data_dir: Root directory containing the TMDb dataset folder or CSVs.

    Returns:
        Dictionary containing DataFrames for movies_metadata, credits, and keywords.
    """
    root = get_data_root(data_dir)
    tmdb_dir = root / "tmdb"
    if not tmdb_dir.exists():
        tmdb_dir = root

    return {
        name: df
        for name, df in (
            (
                name,
                _load_csv(tmdb_dir / f"{name}.csv", columns, required=False, low_memory=(name == "movies_metadata")),
            )
            for name, columns in TMDB_FILE_COLUMNS.items()
        )
        if df is not None
    }


def load_all_data(data_dir: Optional[Path] = None) -> Dict[str, pd.DataFrame]:
    """Load both MovieLens and TMDb datasets.

    Args:
        data_dir: Root directory for data.

    Returns:
        Dictionary containing all loaded DataFrames.
    """
    data = load_movielens_data(data_dir)
    data.update(load_tmdb_data(data_dir))
    return data


__all__ = [
    "DatasetLoadingError",
    "MissingDatasetError",
    "InvalidDatasetSchemaError",
    "get_data_root",
    "load_movielens_data",
    "load_tmdb_data",
    "load_all_data",
]
