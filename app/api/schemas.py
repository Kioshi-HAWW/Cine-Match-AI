from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class MovieSchema(BaseModel):
    movieId: int = Field(..., description="MovieLens movie identifier")
    title: str = Field(..., description="Movie title")
    genres: str = Field("", description="Genre labels separated by '|' if available")
    overview: Optional[str] = Field(None, description="Movie synopsis if available")
    poster_path: Optional[str] = Field(None, description="Poster path or image URL if available")
    poster_url: Optional[str] = Field(None, description="Full poster image URL if available")
    runtime: Optional[float] = Field(None, description="Runtime in minutes if available")
    release_date: Optional[str] = Field(None, description="Release date if available")
    vote_average: Optional[float] = Field(None, description="External metadata average vote if available")
    imdb_id: Optional[str] = Field(None, description="IMDb identifier if available")
    wikipedia_title: Optional[str] = Field(None, description="Wikipedia page title used for fallback metadata")
    wikipedia_url: Optional[str] = Field(None, description="Wikipedia page URL used for fallback metadata")


class GenreSchema(BaseModel):
    name: str = Field(..., description="Genre name")


class PopularMovieSchema(BaseModel):
    movieId: int
    title: str
    genres: str
    overview: str
    weighted_rating: Optional[float] = None
    average_rating: Optional[float] = None
    rating_count: Optional[int] = None
    trending_score: Optional[float] = None
    poster_path: Optional[str] = None
    poster_url: Optional[str] = None
    runtime: Optional[float] = None
    release_date: Optional[str] = None
    vote_average: Optional[float] = None
    imdb_id: Optional[str] = None
    wikipedia_title: Optional[str] = None
    wikipedia_url: Optional[str] = None


class PopularResponse(BaseModel):
    top_rated: List[PopularMovieSchema]
    trending: List[PopularMovieSchema]
    most_rated: List[PopularMovieSchema]


class RecommendationMovieSchema(BaseModel):
    title: str
    genres: str
    overview: Optional[str] = None
    poster_path: Optional[str] = None
    poster_url: Optional[str] = None
    runtime: Optional[float] = None
    release_date: Optional[str] = None
    vote_average: Optional[float] = None
    imdb_id: Optional[str] = None
    wikipedia_title: Optional[str] = None
    wikipedia_url: Optional[str] = None
    similarity_score: Optional[float] = None
    predicted_rating: Optional[float] = None
    content_score: Optional[float] = None
    collaborative_score: Optional[float] = None
    popularity_score: Optional[float] = None
    combined_score: Optional[float] = None


class RecommendationResponse(BaseModel):
    recommendations: List[RecommendationMovieSchema]


class RecommendMovieRequest(BaseModel):
    title: str = Field(..., description="Movie title to search for")
    top_n: int = Field(10, ge=1, le=50, description="Number of similar movies to return")
    min_match_score: int = Field(50, ge=0, le=100, description="Minimum fuzzy title match threshold")


class RecommendInterestRequest(BaseModel):
    interest: str = Field(..., description="User interest text for movie recommendations")
    top_n: int = Field(10, ge=1, le=50, description="Number of recommendations to return")


class RecommendUserRequest(BaseModel):
    user_id: int = Field(..., ge=1, description="MovieLens user ID for collaborative recommendation")
    top_n: int = Field(10, ge=1, le=50, description="Number of recommendations to return")
    model_name: str = Field("SVD", description="Collaborative model to use, e.g. SVD, NMF, KNNBasic")


class RecommendHybridRequest(BaseModel):
    user_id: int = Field(..., ge=1, description="MovieLens user ID for hybrid recommendation")
    interest: str = Field(..., description="Interest text for hybrid recommendation")
    top_n: int = Field(10, ge=1, le=50, description="Number of recommendations to return")
    model_name: str = Field("SVD", description="Collaborative model to use for hybrid recommendation")
