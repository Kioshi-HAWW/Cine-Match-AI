from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional, Union

import pandas as pd

from app.preprocessing.load_data import load_movielens_data, load_tmdb_data

TMDB_IMAGE_BASE_URL = os.environ.get("TMDB_IMAGE_BASE_URL", "https://image.tmdb.org/t/p/w500")


def normalize_json_text(value: Any) -> str:
    """Convert TMDb JSON-like lists into readable text tokens."""
    if pd.isna(value):
        return ""
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("["):
            try:
                decoded = json.loads(text.replace("'", '"'))
            except json.JSONDecodeError:
                return text
            tokens: list[str] = []
            for item in decoded:
                if isinstance(item, dict):
                    name = item.get("name") or item.get("title") or item.get("keyword")
                    if name:
                        tokens.append(str(name).strip())
                elif isinstance(item, str):
                    tokens.append(item.strip())
            return " ".join(dict.fromkeys(token for token in tokens if token))
        return text
    return str(value).strip()


def build_poster_url(poster_path: Any) -> Optional[str]:
    """Return a usable poster URL from a TMDb poster path or URL."""
    if pd.isna(poster_path):
        return None
    poster = str(poster_path).strip()
    if not poster:
        return None
    if poster.startswith(("http://", "https://")):
        return poster
    if not poster.startswith("/"):
        poster = f"/{poster}"
    return f"{TMDB_IMAGE_BASE_URL.rstrip('/')}{poster}"


def _metadata_from_path(metadata_path: Union[str, Path]) -> pd.DataFrame:
    metadata = pd.read_csv(Path(metadata_path), low_memory=False)
    if "tmdbId" not in metadata.columns:
        source = "tmdb_id" if "tmdb_id" in metadata.columns else "id"
        metadata["tmdbId"] = pd.to_numeric(metadata.get(source, pd.Series(dtype="object")), errors="coerce")
    return metadata


def load_movie_metadata(data_dir: Optional[Union[str, Path]] = None, metadata_path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """Load optional TMDb/IMDb-style metadata with normalized columns."""
    if metadata_path is not None:
        metadata = _metadata_from_path(metadata_path)
    else:
        tmdb_data = load_tmdb_data(Path(data_dir) if data_dir is not None else None)
        metadata = tmdb_data.get("movies_metadata", pd.DataFrame())
        if metadata.empty:
            return metadata
        metadata["tmdbId"] = pd.to_numeric(metadata.get("id", pd.Series(dtype="object")), errors="coerce")

    for column in ["overview", "genres", "poster_path", "runtime", "release_date", "vote_average", "imdb_id"]:
        if column not in metadata.columns:
            metadata[column] = None

    metadata["tmdbId"] = pd.to_numeric(metadata["tmdbId"], errors="coerce")
    metadata["overview"] = metadata["overview"].fillna("").astype(str)
    metadata["genres"] = metadata["genres"].apply(normalize_json_text).fillna("").astype(str)
    metadata["poster_path"] = metadata["poster_path"].apply(build_poster_url)
    metadata["runtime"] = pd.to_numeric(metadata["runtime"], errors="coerce")
    metadata["release_date"] = metadata["release_date"].fillna("").astype(str)
    metadata["vote_average"] = pd.to_numeric(metadata["vote_average"], errors="coerce")
    metadata["imdb_id"] = metadata["imdb_id"].fillna("").astype(str)

    return metadata.dropna(subset=["tmdbId"]).drop_duplicates("tmdbId")


def load_enriched_movielens_movies(data_dir: Optional[Union[str, Path]] = None, metadata_path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """Merge MovieLens movies with optional TMDb metadata through links.csv tmdbId."""
    data = load_movielens_data(Path(data_dir) if data_dir is not None else None)
    movies = data["movies"].copy()
    movies["title"] = movies["title"].fillna("").astype(str)
    movies["genres"] = movies["genres"].fillna("").astype(str)

    defaults = {
        "overview": "",
        "poster_path": None,
        "poster_url": None,
        "runtime": None,
        "release_date": "",
        "vote_average": None,
        "imdb_id": "",
        "tmdbId": None,
    }

    metadata = load_movie_metadata(data_dir, metadata_path)
    links = data.get("links", pd.DataFrame()).copy()
    if metadata.empty or links.empty or "tmdbId" not in links.columns:
        for column, value in defaults.items():
            if column not in movies.columns:
                movies[column] = value
        return movies

    links["tmdbId"] = pd.to_numeric(links["tmdbId"], errors="coerce")
    movies = movies.merge(links[["movieId", "tmdbId", "imdbId"]], on="movieId", how="left")
    movies = movies.merge(
        metadata[["tmdbId", "overview", "genres", "poster_path", "runtime", "release_date", "vote_average", "imdb_id"]],
        on="tmdbId",
        how="left",
        suffixes=("", "_meta"),
    )
    movies["overview"] = movies["overview"].fillna("").astype(str)
    movies["poster_path"] = movies["poster_path"].where(movies["poster_path"].notna(), None)
    movies["poster_url"] = movies["poster_path"]
    movies["runtime"] = movies["runtime"].where(movies["runtime"].notna(), None)
    movies["release_date"] = movies["release_date"].fillna("").astype(str)
    movies["vote_average"] = movies["vote_average"].where(movies["vote_average"].notna(), None)
    movies["imdb_id"] = movies["imdb_id"].where(movies["imdb_id"].astype(str).str.len() > 0, movies.get("imdbId", ""))
    movies["genres"] = movies["genres_meta"].where(movies["genres_meta"].fillna("").astype(str).str.len() > 0, movies["genres"])
    movies.drop(columns=[col for col in movies.columns if col.endswith("_meta")], inplace=True)
    return movies
