import React from 'react';
import { BrowserRouter as Router, Routes, Route, NavLink, Link } from 'react-router-dom';
import { Toaster } from 'react-hot-toast';
import Landing from './pages/Landing';
import Upload from './pages/Upload';
import Dashboard from './pages/Dashboard';
import StoryMode from './pages/StoryMode';
import Mitigate from './pages/Mitigate';
import Simulator from './pages/Simulator';
import Monitor from './pages/Monitor';
import Report from './pages/Report';
import './index.css';

function Navbar() {
  return (
    <nav className="navbar">
      <Link to="/" className="navbar-brand">
        <span>⚖️</span>
        <span>Fair<span className="logo-dot">Sight</span></span>
      </Link>
      <div className="navbar-links">
        <NavLink to="/upload">Upload</NavLink>
        <NavLink to="/dashboard">Dashboard</NavLink>
        <NavLink to="/story">Story Mode</NavLink>
        <NavLink to="/mitigate">Mitigate</NavLink>
        <NavLink to="/simulator">Simulator</NavLink>
        <NavLink to="/monitor">Monitor</NavLink>
        <NavLink to="/report">Report</NavLink>
      </div>
    </nav>
  );
}

function Footer() {
  return (
    <footer className="footer">
      <p>
        FairSight v1.0 — AI Bias Detection & Compliance Platform &nbsp;|&nbsp;
        Powered by <a href="https://deepmind.google/technologies/gemini/" target="_blank" rel="noopener noreferrer">Gemini AI</a> &nbsp;|&nbsp;
        <a href="https://github.com/fairsight-ai" target="_blank" rel="noopener noreferrer">GitHub</a>
      </p>
    </footer>
  );
}

export default function App() {
  return (
    <Router>
      <div className="app-layout">
        <Navbar />
        <main className="main-content">
          <Routes>
            <Route path="/" element={<Landing />} />
            <Route path="/upload" element={<Upload />} />
            <Route path="/dashboard" element={<Dashboard />} />
            <Route path="/story" element={<StoryMode />} />
            <Route path="/mitigate" element={<Mitigate />} />
            <Route path="/simulator" element={<Simulator />} />
            <Route path="/monitor" element={<Monitor />} />
            <Route path="/report" element={<Report />} />
          </Routes>
        </main>
        <Footer />
        <Toaster position="top-right" toastOptions={{ style: { fontFamily: 'Inter, sans-serif', fontSize: '0.875rem' } }} />
      </div>
    </Router>
  );
}
