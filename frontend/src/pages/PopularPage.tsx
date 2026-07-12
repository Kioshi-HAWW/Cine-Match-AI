import { useEffect, useState } from 'react'
import type { PopularMovie } from '../types'
import { fetchJson } from '../lib/api'
import MovieCard from '../components/MovieCard'

interface PopularData {
  top_rated: PopularMovie[]
  trending: PopularMovie[]
  most_rated: PopularMovie[]
}

export default function PopularPage() {
  const [data, setData] = useState<PopularData | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  useEffect(() => {
    fetchJson<PopularData>('/popular')
      .then((data: PopularData) => setData(data))
      .catch(err => setError(err instanceof Error ? err.message : 'Unable to load popular movies'))
      .finally(() => setLoading(false))
  }, [])

  if (loading) {
    return <div className="rounded-3xl border border-dashed border-slate-700 bg-slate-900/80 p-8 text-slate-400">Loading popular movies…</div>
  }

  if (error || !data) {
    return <div className="rounded-3xl border border-rose-700/40 bg-slate-900/80 p-8 text-rose-300">{error || 'No popular movies available.'}</div>
  }

  return (
    <section className="space-y-10">
      <div className="space-y-4">
        <div className="rounded-3xl border border-slate-800 bg-slate-900/80 p-8 shadow-xl shadow-slate-950/20">
          <h1 className="text-3xl font-semibold text-white">Popular Movies</h1>
          <p className="mt-3 text-slate-400">Browse the top rated, trending, and most rated movies from the MovieLens data.</p>
        </div>

        <section className="space-y-6">
          <div>
            <h2 className="text-2xl font-semibold text-white">Top Rated</h2>
            <p className="mt-1 text-slate-400">High weighted rating movies.</p>
            <div className="mt-4 grid gap-4 xl:grid-cols-2">
              {data.top_rated.map(movie => (
                <MovieCard key={`top-${movie.movieId}`} movie={movie} />
              ))}
            </div>
          </div>

          <div>
            <h2 className="text-2xl font-semibold text-white">Trending</h2>
            <p className="mt-1 text-slate-400">Recent trending movies based on ratings.</p>
            <div className="mt-4 grid gap-4 xl:grid-cols-2">
              {data.trending.map(movie => (
                <MovieCard key={`trend-${movie.movieId}`} movie={movie} />
              ))}
            </div>
          </div>

          <div>
            <h2 className="text-2xl font-semibold text-white">Most Rated</h2>
            <p className="mt-1 text-slate-400">Highest vote count movies.</p>
            <div className="mt-4 grid gap-4 xl:grid-cols-2">
              {data.most_rated.map(movie => (
                <MovieCard key={`most-${movie.movieId}`} movie={movie} />
              ))}
            </div>
          </div>
        </section>
      </div>
    </section>
  )
}
