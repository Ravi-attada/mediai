import { Link, NavLink } from 'react-router-dom'

export default function Navbar() {
  return (
    <nav className="navbar">
      <Link to="/" className="navbar-logo">
        <div className="logo-icon">🏥</div>
        <span className="logo-text">Medi<span>AI</span></span>
      </Link>

      <ul className="nav-links">
        <li><NavLink to="/" end>Home</NavLink></li>
        <li><NavLink to="/analyzer">Analyzer</NavLink></li>
        <li><NavLink to="/about">Research</NavLink></li>
      </ul>

      <NavLink to="/analyzer" className="nav-cta">Launch Analyzer →</NavLink>
    </nav>
  )
}
