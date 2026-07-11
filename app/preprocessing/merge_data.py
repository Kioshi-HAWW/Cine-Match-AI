from __future__ import annotations

import ast
from typing import Any, Dict, Iterable, Optional

import pandas as pd


def _normalize_tmdb_id(series: pd.Series) -> pd.Series:
    return series.apply(lambda value: int(value) if pd.notna(value) and str(value).isdigit() else None)


def _safe_parse_json(value: Any) -> Any:
    if pd.isna(value):
        return []
    if isinstance(value, (dict, list)):
        return value
    text = str(value).strip()
    if not text:
        return []
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return []


def _extract_release_year(value: Any) -> Optional[int]:
    text = str(value).strip()
    if not text:
        return None
    parts = text.split("-")
    if parts and parts[0].isdigit():
        return int(parts[0])
    return None


def _flatten_keywords(raw_keywords: Any) -> str:
    parsed = _safe_parse_json(raw_keywords)
    if isinstance(parsed, list):
        names = [str(item.get("name", "")).strip().lower().replace(" ", "_") for item in parsed if isinstance(item, dict) and item.get("name")]
        return " ".join(sorted(set(names)))
    return str(raw_keywords).strip().lower()


def _extract_director(raw_crew: Any) -> str:
    parsed = _safe_parse_json(raw_crew)
    if isinstance(parsed, list):
        for member in parsed:
            if isinstance(member, dict) and member.get("job", "").lower() == "director":
                return str(member.get("name", "")).strip().lower()
    return ""


def _extract_cast(raw_cast: Any, top_n: int = 5) -> str:
    parsed = _safe_parse_json(raw_cast)
    if isinstance(parsed, list):
        actors = [str(member.get("name", "")).strip().lower() for member in parsed if isinstance(member, dict) and member.get("name")]
        return " ".join(actors[:top_n])
    return ""


def _build_rating_stats(ratings: pd.DataFrame) -> pd.DataFrame:
    stats = (
        ratings.groupby("movieId")["rating"]
        .agg(average_rating="mean", rating_count="count")
        .reset_index()
    )
    stats["average_rating"] = stats["average_rating"].fillna(0.0)
    stats["rating_count"] = stats["rating_count"].fillna(0).astype(int)
    return stats


def merge_movielens_and_tmdb(
    movies: pd.DataFrame,
    ratings: pd.DataFrame,
    links: pd.DataFrame,
    tmdb_metadata: Optional[pd.DataFrame] = None,
    tmdb_credits: Optional[pd.DataFrame] = None,
    tmdb_keywords: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Merge MovieLens and TMDb datasets using tmdbId.

    The result includes movieId, tmdbId, title, genres, overview, keywords,
    cast, director, release_year, and rating statistics.

    Missing TMDb records are handled gracefully by preserving MovieLens rows and
    populating TMDb-derived fields with empty values.
    """
    merged = movies.copy()
    merged = merged.merge(links[["movieId", "tmdbId"]], on="movieId", how="left")
    merged["tmdbId"] = _normalize_tmdb_id(merged["tmdbId"])

    tmdb_data: Dict[str, pd.DataFrame] = {}
    if tmdb_metadata is not None:
        metadata = tmdb_metadata.copy()
        metadata["id"] = _normalize_tmdb_id(metadata["id"])
        tmdb_data["metadata"] = metadata

    if tmdb_keywords is not None:
        keywords = tmdb_keywords.copy()
        keywords["id"] = _normalize_tmdb_id(keywords["id"])
        tmdb_data["keywords"] = keywords

    if tmdb_credits is not None:
        credits = tmdb_credits.copy()
        credits["id"] = _normalize_tmdb_id(credits["id"])
        tmdb_data["credits"] = credits

    if "metadata" in tmdb_data:
        merged = merged.merge(
            tmdb_data["metadata"].rename(columns={"id": "tmdb_id"}),
            left_on="tmdbId",
            right_on="tmdb_id",
            how="left",
            suffixes=("", "_tmdb"),
        )
    else:
        merged["overview"] = ""
        merged["genres"] = merged["genres"].fillna("")

    if "keywords" in tmdb_data:
        merged = merged.merge(
            tmdb_data["keywords"].rename(columns={"id": "tmdb_id"}),
            left_on="tmdbId",
            right_on="tmdb_id",
            how="left",
        )
    else:
        merged["keywords"] = ""

    if "credits" in tmdb_data:
        merged = merged.merge(
            tmdb_data["credits"].rename(columns={"id": "tmdb_id"}),
            left_on="tmdbId",
            right_on="tmdb_id",
            how="left",
        )
    else:
        merged["cast"] = ""
        merged["director"] = ""

    rating_stats = _build_rating_stats(ratings)
    merged = merged.merge(rating_stats, on="movieId", how="left")

    merged["overview"] = merged.get("overview", "").fillna("").astype(str)
    merged["keywords"] = merged.get("keywords", "").fillna("").astype(str)
    merged["cast"] = merged.get("cast", "").fillna("").astype(str)
    merged["director"] = merged.get("director", "").fillna("").astype(str)
    merged["release_year"] = merged.get("release_date", "")
    merged["release_year"] = merged["release_year"].apply(_extract_release_year)

    if "keywords" in merged.columns and merged["keywords"].dtype == object:
        merged["keywords"] = merged["keywords"].apply(_flatten_keywords)

    if "cast" in merged.columns:
        merged["cast"] = merged["cast"].apply(_extract_cast)

    if "director" in merged.columns:
        merged["director"] = merged["director"].apply(_extract_director)

    merged["average_rating"] = merged["average_rating"].fillna(0.0)
    merged["rating_count"] = merged["rating_count"].fillna(0).astype(int)

    result_columns = [
        "movieId",
        "tmdbId",
        "title",
        "genres",
        "overview",
        "keywords",
        "cast",
        "director",
        "release_year",
        "average_rating",
        "rating_count",
    ]

    return merged.loc[:, [col for col in result_columns if col in merged.columns]]


__all__ = ["merge_movielens_and_tmdb"]
