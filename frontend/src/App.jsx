import { useState } from 'react'
import PredictionForm from './components/PredictionForm'
import MetricsDashboard from './components/MetricsDashboard'
import DriftReport from './components/DriftReport'

const PAGES = [
  { id: 'predict', label: 'Predict Churn', icon: '🎯' },
  { id: 'metrics', label: 'Model Metrics', icon: '📊' },
  { id: 'drift',   label: 'Drift Monitor', icon: '📡' },
]

/**
 * ChurnShield MLOps Platform — Root Application Component
 * Renders the navbar and active page based on navigation state.
 */
export default function App() {
  const [activePage, setActivePage] = useState('predict')

  return (
    <div className="app-wrapper">
      {/* ── Navigation ───────────────────────────────── */}
      <nav className="navbar" role="navigation" aria-label="Main navigation">
        <div className="navbar-inner">
          {/* Brand */}
          <div className="navbar-brand">
            <div className="brand-icon" aria-hidden="true">🛡️</div>
            <div>
              <div className="brand-name">ChurnShield</div>
              <div className="brand-tagline">MLOps Platform</div>
            </div>
          </div>

          {/* Page Links */}
          <ul className="nav-links" role="list">
            {PAGES.map((page) => (
              <li key={page.id}>
                <button
                  id={`nav-${page.id}`}
                  className={`nav-link${activePage === page.id ? ' active' : ''}`}
                  onClick={() => setActivePage(page.id)}
                  aria-current={activePage === page.id ? 'page' : undefined}
                >
                  <span className="nav-icon" aria-hidden="true">{page.icon}</span>
                  {page.label}
                </button>
              </li>
            ))}
          </ul>
        </div>
      </nav>

      {/* ── Page Content ─────────────────────────────── */}
      <main className="main-content" id="main-content">
        {activePage === 'predict' && <PredictionForm />}
        {activePage === 'metrics' && <MetricsDashboard />}
        {activePage === 'drift'   && <DriftReport />}
      </main>

      {/* ── Footer ───────────────────────────────────── */}
      <footer className="app-footer">
        ChurnShield MLOps Platform · Built with FastAPI + MLflow + Prefect + Evidently AI
      </footer>
    </div>
  )
}
