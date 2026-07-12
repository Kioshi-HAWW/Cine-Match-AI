import { useEffect, useState } from 'react'
import type { MovieItem, RecommendationResponse } from '../types'
import { fetchJson } from '../lib/api'
import MovieCard from '../components/MovieCard'

export default function GenrePage() {
  const [genres, setGenres] = useState<string[]>([])
  const [selected, setSelected] = useState('')
  const [recommendations, setRecommendations] = useState<MovieItem[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    fetchJson<Array<{ name: string }>>('/genres')
      .then((data: Array<{ name: string }>) => setGenres(data.map((item) => item.name)))
      .catch(() => setGenres([]))
  }, [])

  async function handleGenre() {
    if (!selected) return
    setError('')
    setLoading(true)

    try {
      const data: RecommendationResponse = await fetchJson('/recommend/interest', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ interest: selected, top_n: 12 }),
      })
      setRecommendations(data.recommendations)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error')
      setRecommendations([])
    } finally {
      setLoading(false)
    }
  }

  return (
    <section className="space-y-8">
      <div className="rounded-3xl border border-slate-800 bg-slate-900/80 p-8 shadow-xl shadow-slate-950/20">
        <h1 className="text-3xl font-semibold text-white">Genre Recommendation</h1>
        <p className="mt-3 text-slate-400">Choose a genre and get movie recommendations related to that genre.</p>

        <div className="mt-6 flex flex-col gap-4 sm:flex-row sm:items-center">
          <select
            value={selected}
            onChange={event => setSelected(event.target.value)}
            className="w-full rounded-3xl border border-slate-700 bg-slate-950/90 px-4 py-3 text-base text-slate-100 outline-none transition focus:border-slate-500 focus:ring-2 focus:ring-slate-500/40"
          >
            <option value="">Select a genre...</option>
            {genres.map(genre => (
              <option key={genre} value={genre}>
                {genre}
              </option>
            ))}
          </select>
          <button
            className="rounded-3xl bg-emerald-500 px-6 py-3 text-sm font-semibold uppercase tracking-[0.16em] text-slate-950 transition hover:bg-emerald-400 disabled:cursor-not-allowed disabled:opacity-60"
            disabled={!selected || loading}
            onClick={handleGenre}
          >
            {loading ? 'Loading…' : 'Recommend'}
          </button>
        </div>
        {error ? <p className="mt-4 text-sm text-rose-400">{error}</p> : null}
      </div>

      <div className="space-y-4">
        {recommendations.length > 0 ? (
          <div className="grid gap-4 xl:grid-cols-2">
            {recommendations.map(movie => (
              <MovieCard key={movie.movieId} movie={movie} />
            ))}
          </div>
        ) : (
          <div className="rounded-3xl border border-dashed border-slate-700 bg-slate-900/80 p-8 text-slate-500">
            {loading ? 'Loading genre recommendations…' : 'Pick a genre to view recommended films.'}
          </div>
        )}
      </div>
    </section>
  )
}
