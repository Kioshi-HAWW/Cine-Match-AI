export default function AboutPage() {
  return (
    <section className="space-y-8">
      <div className="rounded-3xl border border-slate-800 bg-slate-900/80 p-8 shadow-xl shadow-slate-950/20">
        <h1 className="text-3xl font-semibold text-white">About Movie Recommendation Hub</h1>
        <p className="mt-4 text-slate-300 leading-8">
          This project pairs a FastAPI backend with a modern React frontend to provide movie discovery across multiple recommendation strategies.
          It showcases content-based matching, interest-driven search, genre recommendations, and popularity analytics.
        </p>
        <div className="grid gap-4 sm:grid-cols-2">
          <div className="rounded-3xl bg-slate-950/80 p-6">
            <h2 className="text-xl font-semibold text-white">Backend</h2>
            <p className="mt-2 text-slate-400">FastAPI serves JSON endpoints for movie search, genre discovery, and recommendations.</p>
          </div>
          <div className="rounded-3xl bg-slate-950/80 p-6">
            <h2 className="text-xl font-semibold text-white">Frontend</h2>
            <p className="mt-2 text-slate-400">React and Tailwind CSS deliver a responsive, clean UI for browsing movies and viewing recommendations.</p>
          </div>
        </div>
      </div>
    </section>
  )
}
