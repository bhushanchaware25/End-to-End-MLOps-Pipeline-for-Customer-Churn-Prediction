import { useState, useEffect, useCallback } from 'react'
import axios from 'axios'

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

const METRIC_CONFIG = [
  { key: 'roc_auc',   label: 'ROC-AUC',   colorClass: 'metric-blue',    icon: '📈', description: 'Area Under ROC Curve' },
  { key: 'f1_score',  label: 'F1 Score',   colorClass: 'metric-purple',  icon: '⚖️', description: 'Harmonic mean of Precision & Recall' },
  { key: 'accuracy',  label: 'Accuracy',   colorClass: 'metric-cyan',    icon: '🎯', description: 'Overall prediction accuracy' },
  { key: 'precision', label: 'Precision',  colorClass: 'metric-emerald', icon: '🔬', description: 'True positives / all positives predicted' },
  { key: 'recall',    label: 'Recall',     colorClass: 'metric-amber',   icon: '🔍', description: 'True positives / all actual positives' },
]

const QUALITY_THRESHOLDS = {
  roc_auc:   { good: 0.85, fair: 0.75 },
  f1_score:  { good: 0.75, fair: 0.65 },
  accuracy:  { good: 0.85, fair: 0.75 },
  precision: { good: 0.80, fair: 0.65 },
  recall:    { good: 0.75, fair: 0.60 },
}

/**
 * Returns a CSS color variable name based on metric value vs thresholds.
 * @param {string} metricKey
 * @param {number} value
 * @returns {string} CSS variable name
 */
function getMetricColor(metricKey, value) {
  const t = QUALITY_THRESHOLDS[metricKey]
  if (!t) return 'var(--text-primary)'
  if (value >= t.good)  return 'var(--risk-low)'
  if (value >= t.fair)  return 'var(--risk-medium)'
  return 'var(--risk-high)'
}

/**
 * MetricsDashboard — Model performance metrics page.
 * Fetches from GET /metrics and GET /health to display live model stats.
 */
export default function MetricsDashboard() {
  const [metrics, setMetrics]   = useState(null)
  const [health, setHealth]     = useState(null)
  const [loading, setLoading]   = useState(true)
  const [error, setError]       = useState(null)
  const [lastRefresh, setLastRefresh] = useState(null)

  const fetchData = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const [metricsRes, healthRes] = await Promise.all([
        axios.get(`${API_BASE}/metrics`, { timeout: 10000 }),
        axios.get(`${API_BASE}/health`,  { timeout: 10000 }),
      ])
      setMetrics(metricsRes.data)
      setHealth(healthRes.data)
      setLastRefresh(new Date())
    } catch (err) {
      const detail = err.response?.data?.detail || err.message || 'Failed to fetch metrics'
      setError(detail)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchData()
  }, [fetchData])

  if (loading) {
    return (
      <div>
        <div className="page-header">
          <h1 className="page-title">📊 Model Performance Metrics</h1>
        </div>
        <div className="glass-card spinner-wrapper">
          <div className="spinner" />
          <p className="spinner-text">Loading model metrics from MLflow…</p>
        </div>
      </div>
    )
  }

  return (
    <div>
      {/* Page Header */}
      <div className="page-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '1rem' }}>
        <div>
          <h1 className="page-title">📊 Model Performance Metrics</h1>
          <p className="page-subtitle">
            Live metrics from the Production model in the MLflow Model Registry.
          </p>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', flexWrap: 'wrap' }}>
          {lastRefresh && (
            <span style={{ color: 'var(--text-muted)', fontSize: '0.75rem' }}>
              Updated {lastRefresh.toLocaleTimeString()}
            </span>
          )}
          <button
            id="btn-refresh-metrics"
            className="btn btn-secondary"
            onClick={fetchData}
            style={{ padding: '8px 16px', fontSize: '0.82rem' }}
          >
            ↻ Refresh
          </button>
        </div>
      </div>

      {/* Error */}
      {error && (
        <div className="alert alert-danger" role="alert">
          <span className="alert-icon">❌</span>
          <div>
            <div className="alert-title">Failed to load metrics</div>
            {error}. Make sure the API server is running and a model is registered.
          </div>
        </div>
      )}

      {/* Model Info Banner */}
      {metrics && (
        <>
          <div className="glass-card" style={{ padding: '1.5rem', marginBottom: '1.5rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '1rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                <div style={{
                  width: 48, height: 48,
                  background: 'linear-gradient(135deg, var(--accent-blue), var(--accent-purple))',
                  borderRadius: 12, display: 'flex', alignItems: 'center', justifyContent: 'center',
                  fontSize: 22, boxShadow: 'var(--shadow-glow)',
                }}>🤖</div>
                <div>
                  <div style={{ fontWeight: 700, fontSize: '1.05rem', color: 'var(--text-primary)' }}>
                    {metrics.model_name}
                  </div>
                  <div style={{ color: 'var(--text-muted)', fontSize: '0.8rem', marginTop: 2 }}>
                    Stage: <span style={{ color: 'var(--accent-emerald)', fontWeight: 600 }}>{metrics.stage}</span>
                    &nbsp;·&nbsp;Run: <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.75rem' }}>{metrics.run_id?.substring(0, 8)}…</span>
                  </div>
                </div>
              </div>
              <div style={{ display: 'flex', gap: '0.75rem', alignItems: 'center', flexWrap: 'wrap' }}>
                <span className="version-badge">v{metrics.model_version}</span>
                {health && (
                  <span className={`risk-badge ${health.model_loaded ? 'risk-low' : 'risk-high'}`}>
                    {health.model_loaded ? '● Online' : '● Offline'}
                  </span>
                )}
              </div>
            </div>
          </div>

          {/* Metric Cards */}
          <div className="metrics-grid">
            {METRIC_CONFIG.map(({ key, label, colorClass, icon, description }) => {
              const value = metrics.metrics?.[key]
              const hasValue = value !== undefined && value !== null
              const color = hasValue ? getMetricColor(key, value) : 'var(--text-muted)'
              return (
                <div key={key} className="glass-card metric-card" id={`metric-${key}`} title={description}>
                  <div style={{ fontSize: 24, marginBottom: 8 }}>{icon}</div>
                  <div className="metric-value" style={{ color, fontSize: '2.2rem' }}>
                    {hasValue ? (value * 100).toFixed(1) + '%' : '—'}
                  </div>
                  <div className="metric-label">{label}</div>
                  <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginTop: 4 }}>
                    {description}
                  </div>
                </div>
              )
            })}
          </div>

          {/* All Raw Metrics Table */}
          {metrics.metrics && Object.keys(metrics.metrics).length > 0 && (
            <div className="glass-card" style={{ padding: '1.5rem' }}>
              <h2 style={{ fontSize: '1rem', fontWeight: 700, marginBottom: '1rem', color: 'var(--text-primary)' }}>
                📋 All Logged Metrics
              </h2>
              <div style={{ overflowX: 'auto' }}>
                <table className="data-table" aria-label="All model metrics">
                  <thead>
                    <tr>
                      <th>Metric</th>
                      <th>Value</th>
                      <th>Quality</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(metrics.metrics)
                      .sort(([a], [b]) => a.localeCompare(b))
                      .map(([key, value]) => {
                        const color = getMetricColor(key, value)
                        const pct = typeof value === 'number' && value <= 1
                          ? `${(value * 100).toFixed(2)}%`
                          : value?.toFixed ? value.toFixed(4) : String(value)
                        const quality = QUALITY_THRESHOLDS[key]
                          ? value >= QUALITY_THRESHOLDS[key].good ? '✅ Good'
                            : value >= QUALITY_THRESHOLDS[key].fair ? '⚠️ Fair'
                            : '❌ Poor'
                          : '—'
                        return (
                          <tr key={key}>
                            <td style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-secondary)', fontSize: '0.82rem' }}>
                              {key}
                            </td>
                            <td style={{ fontFamily: 'var(--font-mono)', fontWeight: 700, color }}>
                              {pct}
                            </td>
                            <td style={{ fontSize: '0.82rem' }}>{quality}</td>
                          </tr>
                        )
                      })}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </>
      )}

      {/* Health Details */}
      {health && (
        <div className="glass-card" style={{ padding: '1.5rem', marginTop: '1.5rem' }}>
          <h2 style={{ fontSize: '1rem', fontWeight: 700, marginBottom: '1rem' }}>🔧 Service Health</h2>
          <div className="stat-row">
            {[
              { label: 'API Status',     value: health.status === 'healthy' ? '✅ Healthy' : '⚠️ Degraded' },
              { label: 'Model Loaded',   value: health.model_loaded ? 'Yes' : 'No' },
              { label: 'Model Version',  value: `v${health.model_version}` },
            ].map(({ label, value }) => (
              <div key={label} className="stat-item">
                <div className="stat-value" style={{ fontSize: '1rem', fontFamily: 'var(--font-mono)' }}>{value}</div>
                <div className="stat-label">{label}</div>
              </div>
            ))}
          </div>
          {health.loaded_at && (
            <p style={{ color: 'var(--text-muted)', fontSize: '0.75rem', marginTop: '1rem', fontFamily: 'var(--font-mono)' }}>
              Model loaded at: {new Date(health.loaded_at).toLocaleString()}
            </p>
          )}
        </div>
      )}
    </div>
  )
}
