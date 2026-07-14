export interface MovieItem {
  movieId: number
  title: string
  genres: string
  overview: string
  poster_path?: string | null
  poster_url?: string | null
  runtime?: number | null
  release_date?: string | null
  vote_average?: number | null
  imdb_id?: string | null
  wikipedia_title?: string | null
  wikipedia_url?: string | null
}

export interface PopularMovie extends MovieItem {
  weighted_rating?: number
  average_rating?: number
  rating_count?: number
  trending_score?: number
  content_score?: number
  collaborative_score?: number
  popularity_score?: number
  combined_score?: number
}

export interface RecommendationResponse {
  recommendations: PopularMovie[]
}
