from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


STOP_WORDS: set[str] = {
    "a", "an", "the", "and", "or", "but", "if", "while", "with",
    "for", "to", "of", "in", "on", "at", "by", "from", "up",
    "about", "as", "is", "it", "this", "that", "these", "those",
}


def _safe_parse_json_like(value: object) -> Any:
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


def _normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def _tokenize_text(text: str) -> list[str]:
    tokens = [token for token in text.split() if token]
    return [token for token in tokens if token not in STOP_WORDS]


def _join_tokens(tokens: Iterable[str]) -> str:
    return " ".join(token.strip().lower() for token in tokens if token and token not in STOP_WORDS)


def _flatten_keywords(raw_keywords: object) -> list[str]:
    parsed = _safe_parse_json_like(raw_keywords)
    if isinstance(parsed, list):
        names = [str(item.get("name", "")).strip().lower().replace(" ", "_") for item in parsed if isinstance(item, dict) and item.get("name")]
        return [token for token in names if token and token not in STOP_WORDS]
    return []


def _extract_director(raw_crew: object) -> str:
    parsed = _safe_parse_json_like(raw_crew)
    if isinstance(parsed, list):
        for member in parsed:
            if isinstance(member, dict) and member.get("job", "").lower() == "director":
                return str(member.get("name", "")).strip().lower().replace(" ", "_")
    return ""


def _extract_top_actors(raw_cast: object, top_n: int = 5) -> list[str]:
    parsed = _safe_parse_json_like(raw_cast)
    if isinstance(parsed, list):
        actors = [str(member.get("name", "")).strip().lower().replace(" ", "_") for member in parsed if isinstance(member, dict) and member.get("name")]
        return [actor for actor in actors[:top_n] if actor not in STOP_WORDS]
    return []


def _clean_column_values(values: Iterable[str]) -> str:
    tokens = []
    for raw in values:
        if raw is None:
            continue
        normalized = str(raw).strip().lower()
        tokens.extend(_tokenize_text(normalized))
    return " ".join(tokens)


def create_content_column(movies: pd.DataFrame, top_actor_count: int = 5) -> pd.DataFrame:
    """Create a cleaned single content field for each movie."""
    movies = movies.copy()

    def _get_column_series(column: str) -> pd.Series:
        return movies[column] if column in movies.columns else pd.Series("", index=movies.index)

    movies["title"] = _get_column_series("title").apply(_normalize_text)
    movies["genres"] = _get_column_series("genres").fillna("").astype(str).apply(
        lambda raw: " ".join([genre.strip().lower().replace(" ", "_") for genre in raw.split("|") if genre.strip()])
    )
    movies["overview"] = _get_column_series("overview").fillna("").astype(str).apply(_normalize_text)

    if "tmdb_keywords" in movies.columns:
        movies["keywords_list"] = movies["tmdb_keywords"].apply(_flatten_keywords)
    else:
        movies["keywords_list"] = _get_column_series("keywords").apply(_flatten_keywords)

    if "cast" in movies.columns:
        movies["top_actors"] = movies["cast"].apply(lambda raw: _extract_top_actors(raw, top_actor_count))
    else:
        movies["top_actors"] = [[] for _ in range(len(movies))]

    if "crew" in movies.columns:
        movies["director"] = movies["crew"].apply(_extract_director)
    else:
        movies["director"] = _get_column_series("director").astype(str).apply(_normalize_text)

    movies["content"] = (
        movies["title"].fillna("")
        + " "
        + movies["genres"].fillna("")
        + " "
        + movies["overview"].fillna("")
        + " "
        + movies["keywords_list"].apply(_join_tokens)
        + " "
        + movies["director"].fillna("")
        + " "
        + movies["top_actors"].apply(lambda actors: " ".join(actors))
    )
    movies["content"] = movies["content"].str.replace(r"\s+", " ", regex=True).str.strip()
    return movies


def build_text_features(
    movies: pd.DataFrame,
    use_tfidf: bool = True,
    max_features: int = 10_000,
) -> tuple[pd.DataFrame, Any, Any]:
    movies = movies.copy()
    movies = create_content_column(movies)
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english") if use_tfidf else CountVectorizer(max_features=max_features, stop_words="english")
    feature_matrix = vectorizer.fit_transform(movies["content"].fillna(""))
    feature_df = pd.DataFrame(
        feature_matrix.toarray(),
        index=movies["movieId"].astype(int),
        columns=vectorizer.get_feature_names_out(),
    )
    return movies, vectorizer, feature_df


def save_features(feature_matrix: np.ndarray, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, feature_matrix)
    return output_path
