import { Routes, Route } from 'react-router-dom'
import Navbar from './components/Navbar'
import Home from './pages/Home'
import Analyzer from './pages/Analyzer'
import About from './pages/About'
import { Toaster } from 'react-hot-toast'

export default function App() {
  return (
    <div className="app">
      <Toaster position="top-right" toastOptions={{ style: { background: '#1e2035', color: '#fff', border: '1px solid #3d4270' } }} />
      <Navbar />
      <Routes>
        <Route path="/"         element={<Home />} />
        <Route path="/analyzer" element={<Analyzer />} />
        <Route path="/about"    element={<About />} />
      </Routes>
    </div>
  )
}
