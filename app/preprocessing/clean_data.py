from __future__ import annotations

import pandas as pd


def _normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def clean_movies(movies: pd.DataFrame) -> pd.DataFrame:
    movies = movies.copy()
    movies = movies.drop_duplicates(subset=["movieId"])
    movies["movieId"] = movies["movieId"].astype(int)
    movies["title"] = movies["title"].apply(_normalize_text)
    movies["genres"] = movies["genres"].fillna("(no genres listed)").astype(str).str.strip().str.lower()
    movies["genres"] = movies["genres"].replace("", "(no genres listed)")
    return movies


def clean_ratings(ratings: pd.DataFrame) -> pd.DataFrame:
    ratings = ratings.copy()
    ratings = ratings.dropna(subset=["userId", "movieId", "rating"])
    ratings = ratings.drop_duplicates()
    ratings["userId"] = ratings["userId"].astype(int)
    ratings["movieId"] = ratings["movieId"].astype(int)
    ratings["rating"] = ratings["rating"].astype(float)
    return ratings


def clean_tags(tags: pd.DataFrame) -> pd.DataFrame:
    tags = tags.copy()
    tags = tags.dropna(subset=["userId", "movieId", "tag"])
    tags = tags.drop_duplicates(subset=["userId", "movieId", "tag"])
    tags["tag"] = tags["tag"].apply(_normalize_text)
    tags["userId"] = tags["userId"].astype(int)
    tags["movieId"] = tags["movieId"].astype(int)
    return tags


def clean_tmdb_metadata(metadata: pd.DataFrame) -> pd.DataFrame:
    metadata = metadata.copy()
    metadata = metadata.drop_duplicates(subset=["id"])
    metadata["id"] = metadata["id"].apply(lambda value: int(value) if pd.notna(value) and str(value).isdigit() else None)
    metadata["title"] = metadata["title"].apply(_normalize_text)
    metadata["overview"] = metadata["overview"].fillna("").astype(str).str.strip().str.lower()
    metadata["genres"] = metadata["genres"].fillna("[]")
    return metadata


def clean_credits(credits: pd.DataFrame) -> pd.DataFrame:
    credits = credits.copy()
    credits = credits.drop_duplicates(subset=["id"])
    credits["id"] = credits["id"].apply(lambda value: int(value) if pd.notna(value) and str(value).isdigit() else None)
    credits["cast"] = credits["cast"].fillna("[]").astype(str)
    credits["crew"] = credits["crew"].fillna("[]").astype(str)
    return credits


def clean_keywords(keywords: pd.DataFrame) -> pd.DataFrame:
    keywords = keywords.copy()
    keywords = keywords.drop_duplicates(subset=["id"])
    keywords["id"] = keywords["id"].apply(lambda value: int(value) if pd.notna(value) and str(value).isdigit() else None)
    keywords["keywords"] = keywords["keywords"].fillna("[]").astype(str)
    return keywords
