export interface MovieItem {
  movieId: number
  title: string
  genres: string
  overview: string
  poster_path?: string | null
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
