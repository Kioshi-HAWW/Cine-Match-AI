import type { PopularMovie } from '../types'

export default function MovieCard({ movie }: { movie: PopularMovie }) {
  return (
    <article className="rounded-3xl border border-slate-800 bg-slate-900/80 p-5 shadow-xl shadow-slate-950/20">
      <div className="flex flex-col gap-4 lg:flex-row lg:items-start">
        <div className="min-w-[150px] overflow-hidden rounded-3xl bg-slate-800">
          {movie.poster_url || movie.poster_path ? (
            <img
              src={movie.poster_url || movie.poster_path || ''}
              alt={`${movie.title} poster`}
              className="aspect-[2/3] w-[150px] object-cover"
              loading="lazy"
            />
          ) : (
            <div className="aspect-[2/3] w-[150px] bg-gradient-to-br from-slate-800 via-slate-900 to-slate-950 p-4">
              <div className="h-full w-full rounded-2xl bg-slate-950" />
            </div>
          )}
        </div>
        <div className="flex-1 space-y-3">
          <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
            <h2 className="text-xl font-semibold text-white">{movie.title}</h2>
            {movie.weighted_rating !== undefined ? (
              <span className="rounded-full bg-emerald-600/10 px-3 py-1 text-xs font-semibold uppercase tracking-[0.18em] text-emerald-300">
                Rating {movie.weighted_rating.toFixed(2)}
              </span>
            ) : null}
          </div>
          <p className="text-sm leading-6 text-slate-300">{movie.overview || 'No overview available.'}</p>
          <div className="flex flex-wrap gap-3 text-xs text-slate-400">
            {movie.release_date ? <span>Released {movie.release_date}</span> : null}
            {movie.runtime ? <span>{Math.round(movie.runtime)} min</span> : null}
            {movie.imdb_id ? <span>IMDb {movie.imdb_id}</span> : null}
            {movie.wikipedia_url ? (
              <a href={movie.wikipedia_url} target="_blank" rel="noreferrer" className="text-sky-300 hover:text-sky-200">
                Wikipedia
              </a>
            ) : null}
          </div>
          <div className="flex flex-wrap gap-2">
            {movie.genres?.split('|').map(genre =>
              genre ? (
                <span key={genre} className="rounded-full border border-slate-700 bg-slate-800 px-3 py-1 text-xs text-slate-300">
                  {genre}
                </span>
              ) : null,
            )}
          </div>
        </div>
      </div>
    </article>
  )
}
