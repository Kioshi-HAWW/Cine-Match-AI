import { Route, Routes } from 'react-router-dom'
import HomePage from './pages/HomePage'
import SearchPage from './pages/SearchPage'
import GenrePage from './pages/GenrePage'
import InterestPage from './pages/InterestPage'
import PopularPage from './pages/PopularPage'
import AboutPage from './pages/AboutPage'
import Layout from './components/Layout'

function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/search" element={<SearchPage />} />
        <Route path="/genres" element={<GenrePage />} />
        <Route path="/interest" element={<InterestPage />} />
        <Route path="/popular" element={<PopularPage />} />
        <Route path="/about" element={<AboutPage />} />
      </Routes>
    </Layout>
  )
}

export default App
