import { FormEvent, useState } from 'react'
import type { PopularMovie, RecommendationResponse } from '../types'
import { fetchJson } from '../lib/api'
import MovieCard from '../components/MovieCard'

export default function InterestPage() {
  const [interest, setInterest] = useState('')
  const [results, setResults] = useState<PopularMovie[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    setError('')
    setLoading(true)

    try {
      const data: RecommendationResponse = await fetchJson('/recommend/interest', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ interest, top_n: 12 }),
      })
      setResults(data.recommendations)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error')
      setResults([])
    } finally {
      setLoading(false)
    }
  }

  return (
    <section className="space-y-8">
      <div className="rounded-3xl border border-slate-800 bg-slate-900/80 p-8 shadow-xl shadow-slate-950/20">
        <h1 className="text-3xl font-semibold text-white">Interest Recommendation</h1>
        <p className="mt-3 text-slate-400">Describe your interests and get movies that match the theme.</p>

        <form onSubmit={handleSubmit} className="mt-6 space-y-4">
          <textarea
            value={interest}
            onChange={event => setInterest(event.target.value)}
            rows={4}
            placeholder="e.g. space exploration, romantic thrillers, or science fiction drama"
            className="w-full rounded-3xl border border-slate-700 bg-slate-950/90 p-4 text-base text-slate-100 outline-none transition focus:border-slate-500 focus:ring-2 focus:ring-slate-500/40"
          />
          <button
            type="submit"
            className="rounded-3xl bg-emerald-500 px-6 py-3 text-sm font-semibold uppercase tracking-[0.16em] text-slate-950 transition hover:bg-emerald-400 disabled:cursor-not-allowed disabled:opacity-60"
            disabled={loading || !interest.trim()}
          >
            {loading ? 'Finding movies…' : 'Recommend'}
          </button>
        </form>

        {error ? <p className="mt-4 text-sm text-rose-400">{error}</p> : null}
      </div>

      <div className="space-y-4">
        {results.length > 0 ? (
          <div className="grid gap-4 xl:grid-cols-2">
            {results.map(movie => (
              <MovieCard key={movie.movieId} movie={movie} />
            ))}
          </div>
        ) : (
          <div className="rounded-3xl border border-dashed border-slate-700 bg-slate-900/80 p-8 text-slate-500">
            {loading ? 'Loading interest-based recommendations…' : 'Share your interests to explore recommended movies.'}
          </div>
        )}
      </div>
    </section>
  )
}
