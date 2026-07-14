from __future__ import annotations

import os
import re
from functools import lru_cache
from typing import Any, Dict, Optional

import requests

WIKIPEDIA_API_URL = os.environ.get("WIKIPEDIA_API_URL", "https://en.wikipedia.org/w/api.php")
WIKIPEDIA_TIMEOUT_SECONDS = float(os.environ.get("WIKIPEDIA_TIMEOUT_SECONDS", "3"))
WIKIPEDIA_USER_AGENT = os.environ.get(
    "WIKIPEDIA_USER_AGENT",
    "CineMatchAI/1.0 (https://github.com/uditnarayan/Cine-Match-AI)",
)

_TITLE_YEAR_RE = re.compile(r"\s*\((\d{4})\)\s*$")


def split_movielens_title(title: str) -> tuple[str, Optional[str]]:
    """Split a MovieLens title like 'Toy Story (1995)' into title and year."""
    clean_title = str(title or "").strip()
    match = _TITLE_YEAR_RE.search(clean_title)
    if not match:
        return clean_title, None
    return clean_title[: match.start()].strip(), match.group(1)


def _request_json(params: Dict[str, Any]) -> Dict[str, Any]:
    response = requests.get(
        WIKIPEDIA_API_URL,
        params=params,
        headers={"User-Agent": WIKIPEDIA_USER_AGENT},
        timeout=WIKIPEDIA_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    payload = response.json()
    return payload if isinstance(payload, dict) else {}


def _best_search_title(title: str, year: Optional[str]) -> Optional[str]:
    query = f"{title} {year} film" if year else f"{title} film"
    payload = _request_json(
        {
            "action": "query",
            "format": "json",
            "list": "search",
            "srsearch": query,
            "srlimit": 1,
            "utf8": 1,
        }
    )
    results = payload.get("query", {}).get("search", [])
    if not results:
        return None
    first = results[0]
    page_title = first.get("title")
    return str(page_title) if page_title else None


def _summary_for_page(page_title: str) -> Dict[str, Any]:
    payload = _request_json(
        {
            "action": "query",
            "format": "json",
            "prop": "extracts|pageimages|info",
            "exintro": 1,
            "explaintext": 1,
            "piprop": "thumbnail|original",
            "pithumbsize": 500,
            "inprop": "url",
            "redirects": 1,
            "titles": page_title,
            "utf8": 1,
        }
    )
    pages = payload.get("query", {}).get("pages", {})
    if not pages:
        return {}
    page = next(iter(pages.values()))
    if not isinstance(page, dict) or page.get("missing") is not None:
        return {}

    thumbnail = page.get("thumbnail") or {}
    original = page.get("original") or {}
    image_url = thumbnail.get("source") or original.get("source")
    extract = str(page.get("extract") or "").strip()

    return {
        "overview": extract,
        "poster_path": image_url,
        "poster_url": image_url,
        "wikipedia_title": page.get("title"),
        "wikipedia_url": page.get("fullurl"),
    }


@lru_cache(maxsize=1024)
def fetch_movie_summary(title: str) -> Dict[str, Any]:
    """Fetch a best-effort movie overview and image from Wikipedia by movie title."""
    movie_title, year = split_movielens_title(title)
    if not movie_title:
        return {}

    try:
        for candidate in [f"{movie_title} ({year} film)" if year else "", f"{movie_title} (film)"]:
            if not candidate:
                continue
            summary = _summary_for_page(candidate)
            if summary.get("overview") or summary.get("poster_url"):
                return summary

        search_title = _best_search_title(movie_title, year)
        if not search_title:
            return {}
        return _summary_for_page(search_title)
    except requests.RequestException:
        return {}
    except ValueError:
        return {}


def fill_missing_with_wikipedia(movie: Dict[str, Any]) -> Dict[str, Any]:
    """Fill missing overview/poster fields in an API movie dict from Wikipedia."""
    if str(os.environ.get("ENABLE_WIKIPEDIA_ENRICHMENT", "true")).lower() in {"0", "false", "no"}:
        return movie

    has_overview = bool(str(movie.get("overview") or "").strip()) and str(movie.get("overview")) != "N/A"
    has_poster = bool(movie.get("poster_url") or movie.get("poster_path")) and movie.get("poster_url") != "N/A"
    if has_overview and has_poster:
        return movie

    wiki = fetch_movie_summary(str(movie.get("title", "")))

    if not has_overview and wiki.get("overview"):
        movie["overview"] = wiki["overview"]
    if not has_poster and (wiki.get("poster_url") or wiki.get("poster_path")):
        movie["poster_url"] = wiki.get("poster_url") or wiki.get("poster_path")
        movie["poster_path"] = wiki.get("poster_path") or wiki.get("poster_url")
    if wiki.get("wikipedia_title"):
        movie["wikipedia_title"] = wiki["wikipedia_title"]
    if wiki.get("wikipedia_url"):
        movie["wikipedia_url"] = wiki["wikipedia_url"]
    return movie
