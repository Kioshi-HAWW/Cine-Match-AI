import { Link } from 'react-router-dom'

const navItems = [
  { label: 'Home', path: '/' },
  { label: 'Search Movie', path: '/search' },
  { label: 'Genre Recommendation', path: '/genres' },
  { label: 'Interest Recommendation', path: '/interest' },
  { label: 'Popular Movies', path: '/popular' },
  { label: 'About', path: '/about' },
]

export default function Layout({ children }: { children: React.ReactNode }) {
  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      <header className="border-b border-slate-800 bg-slate-900/95 backdrop-blur-sm sticky top-0 z-20">
        <div className="mx-auto flex max-w-7xl flex-col gap-4 px-4 py-4 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <h1 className="text-2xl font-semibold tracking-tight text-white">Movie Recommendation Hub</h1>
            <p className="text-sm text-slate-400">FastAPI + React + Tailwind movie discovery.</p>
          </div>
          <nav className="flex flex-wrap gap-2">
            {navItems.map(item => (
              <Link
                key={item.path}
                to={item.path}
                className="rounded-full border border-slate-700 px-4 py-2 text-sm text-slate-200 transition hover:border-slate-500 hover:bg-slate-800"
              >
                {item.label}
              </Link>
            ))}
          </nav>
        </div>
      </header>

      <main className="mx-auto max-w-7xl px-4 py-8">{children}</main>

      <footer className="border-t border-slate-800 bg-slate-950/95 px-4 py-6 text-center text-sm text-slate-500">
        Built with React, Tailwind CSS, and FastAPI.
      </footer>
    </div>
  )
}
