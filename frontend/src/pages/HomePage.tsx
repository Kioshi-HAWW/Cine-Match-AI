import { Link } from 'react-router-dom'

const cards = [
  { title: 'Search Movie', description: 'Find similar titles based on a movie name.', path: '/search' },
  { title: 'Genre Recommendation', description: 'Discover movies by genre categories.', path: '/genres' },
  { title: 'Interest Recommendation', description: 'Get recommendations from an interest phrase.', path: '/interest' },
  { title: 'Popular Movies', description: 'Browse top rated, trending, and most rated films.', path: '/popular' },
  { title: 'About', description: 'Learn more about the app and architecture.', path: '/about' },
]

export default function HomePage() {
  return (
    <section className="space-y-8">
      <div className="rounded-3xl border border-slate-800 bg-slate-900/80 p-8 shadow-xl shadow-slate-950/20">
        <p className="text-sm uppercase tracking-[0.3em] text-emerald-300/80">Movie Recommendation Hub</p>
        <h1 className="mt-4 text-4xl font-semibold text-white sm:text-5xl">Discover movies with AI-powered recommendations</h1>
        <p className="mt-4 max-w-3xl text-lg leading-8 text-slate-300">
          Explore movies via title search, genre, user interests, and popularity. The frontend connects directly to FastAPI endpoints to return recommendations in real time.
        </p>
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        {cards.map(card => (
          <Link
            key={card.path}
            to={card.path}
            className="rounded-3xl border border-slate-800 bg-slate-900/80 p-6 transition hover:border-slate-500 hover:bg-slate-800"
          >
            <h2 className="text-xl font-semibold text-white">{card.title}</h2>
            <p className="mt-2 text-slate-400">{card.description}</p>
          </Link>
        ))}
      </div>
    </section>
  )
}
