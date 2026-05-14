import { useState, useEffect, useCallback } from 'react'
import axios from 'axios'

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

/**
 * Returns score color based on drift threshold.
 * @param {number} score
 * @returns {string} CSS color variable
 */
function getDriftColor(score) {
  if (score < 0.15) return 'var(--risk-low)'
  if (score < 0.30) return 'var(--risk-medium)'
  return 'var(--risk-high)'
}

/**
 * DriftReport — Data drift monitoring page.
 * Fetches from GET /drift-report and displays drift status,
 * alert banner if drift detected, and per-feature drift table.
 */
export default function DriftReport() {
  const [report, setReport]       = useState(null)
  const [loading, setLoading]     = useState(true)
  const [error, setError]         = useState(null)
  const [lastRefresh, setLastRefresh] = useState(null)

  const fetchReport = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const { data } = await axios.get(`${API_BASE}/drift-report`, { timeout: 10000 })
      setReport(data)
      setLastRefresh(new Date())
    } catch (err) {
      const detail = err.response?.data?.detail || err.message || 'Failed to fetch drift report'
      setError(detail)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchReport()
  }, [fetchReport])

  if (loading) {
    return (
      <div>
        <div className="page-header">
          <h1 className="page-title">📡 Drift Monitor</h1>
        </div>
        <div className="glass-card spinner-wrapper">
          <div className="spinner" />
          <p className="spinner-text">Loading drift report from Evidently AI…</p>
        </div>
      </div>
    )
  }

  const driftPct = report ? Math.round((report.drift_score || 0) * 100) : 0
  const driftColor = report ? getDriftColor(report.drift_score || 0) : 'var(--text-muted)'
  const featureEntries = report?.features ? Object.entries(report.features) : []
  const driftedFeatures = featureEntries.filter(([, info]) => info.drifted)

  return (
    <div>
      {/* Page Header */}
      <div className="page-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '1rem' }}>
        <div>
          <h1 className="page-title">📡 Data Drift Monitor</h1>
          <p className="page-subtitle">
            Evidently AI comparison of reference vs current dataset. Detects distribution shift that may degrade model performance.
          </p>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
          {lastRefresh && (
            <span style={{ color: 'var(--text-muted)', fontSize: '0.75rem' }}>
              Updated {lastRefresh.toLocaleTimeString()}
            </span>
          )}
          <button
            id="btn-refresh-drift"
            className="btn btn-secondary"
            onClick={fetchReport}
            style={{ padding: '8px 16px', fontSize: '0.82rem' }}
          >
            ↻ Refresh
          </button>
        </div>
      </div>

      {/* Fetch Error */}
      {error && (
        <div className="alert alert-danger" role="alert">
          <span className="alert-icon">❌</span>
          <div>
            <div className="alert-title">Failed to load drift report</div>
            {error}. Run <code style={{ fontFamily: 'var(--font-mono)', background: 'rgba(255,255,255,0.06)', padding: '1px 6px', borderRadius: 4 }}>make drift-report</code> to generate one.
          </div>
        </div>
      )}

      {report && (
        <>
          {/* Drift Alert Banner */}
          {report.drift_detected ? (
            <div className="alert alert-danger" role="alert" id="drift-alert-banner">
              <span className="alert-icon">🚨</span>
              <div>
                <div className="alert-title">Data Drift Detected!</div>
                Significant distribution shift found in the current dataset compared to the reference baseline.
                Consider triggering a retraining pipeline run: <code style={{ fontFamily: 'var(--font-mono)', background: 'rgba(255,255,255,0.06)', padding: '1px 6px', borderRadius: 4 }}>make pipeline</code>
              </div>
            </div>
          ) : (
            <div className="alert alert-success" role="status" id="drift-status-ok">
              <span className="alert-icon">✅</span>
              <div>
                <div className="alert-title">No Significant Drift Detected</div>
                Current data distribution is consistent with the reference baseline. Model performance should be stable.
              </div>
            </div>
          )}

          {/* Summary Cards */}
          <div className="stat-row" style={{ marginBottom: '1.5rem' }}>
            <div className="stat-item">
              <div className="stat-value" style={{ color: driftColor, fontFamily: 'var(--font-mono)' }}>
                {(report.drift_score || 0).toFixed(3)}
              </div>
              <div className="stat-label">Overall Drift Score</div>
            </div>
            <div className="stat-item">
              <div className="stat-value" style={{ color: report.drift_detected ? 'var(--risk-high)' : 'var(--risk-low)' }}>
                {report.drift_detected ? 'Yes' : 'No'}
              </div>
              <div className="stat-label">Drift Detected</div>
            </div>
            <div className="stat-item">
              <div className="stat-value" style={{ color: driftedFeatures.length > 0 ? 'var(--risk-high)' : 'var(--risk-low)', fontFamily: 'var(--font-mono)' }}>
                {driftedFeatures.length} / {featureEntries.length}
              </div>
              <div className="stat-label">Features Drifted</div>
            </div>
            <div className="stat-item">
              <div className="stat-value" style={{ fontFamily: 'var(--font-mono)', fontSize: '0.95rem' }}>
                {report.reference_rows?.toLocaleString() || '—'}
              </div>
              <div className="stat-label">Reference Rows</div>
            </div>
            <div className="stat-item">
              <div className="stat-value" style={{ fontFamily: 'var(--font-mono)', fontSize: '0.95rem' }}>
                {report.current_rows?.toLocaleString() || '—'}
              </div>
              <div className="stat-label">Current Rows</div>
            </div>
          </div>

          {/* Drift Score Progress Bar */}
          <div className="glass-card" style={{ padding: '1.5rem', marginBottom: '1.5rem' }}>
            <h2 style={{ fontSize: '1rem', fontWeight: 700, marginBottom: '1rem', color: 'var(--text-primary)' }}>
              Overall Drift Score
            </h2>
            <div className="progress-container">
              <div className="progress-label">
                <span style={{ color: 'var(--text-secondary)', fontSize: '0.82rem' }}>
                  0 — No Drift
                </span>
                <span style={{ color: driftColor, fontFamily: 'var(--font-mono)', fontWeight: 700 }}>
                  {(report.drift_score || 0).toFixed(4)}
                </span>
                <span style={{ color: 'var(--text-secondary)', fontSize: '0.82rem' }}>
                  1.0 — Full Drift
                </span>
              </div>
              <div className="progress-bar-track" style={{ height: 12 }}>
                <div
                  className={`progress-bar-fill ${
                    (report.drift_score || 0) < 0.15 ? 'progress-low'
                    : (report.drift_score || 0) < 0.30 ? 'progress-medium'
                    : 'progress-high'
                  }`}
                  style={{ width: `${Math.min(driftPct, 100)}%` }}
                  role="progressbar"
                  aria-valuenow={driftPct}
                  aria-valuemin={0}
                  aria-valuemax={100}
                />
              </div>
              {/* Threshold marker */}
              <div style={{ position: 'relative', height: 18, marginTop: 4 }}>
                <div style={{
                  position: 'absolute',
                  left: '30%',
                  transform: 'translateX(-50%)',
                  fontSize: '0.68rem',
                  color: 'var(--risk-high)',
                  fontFamily: 'var(--font-mono)',
                }}>
                  ▲ 0.30 threshold
                </div>
              </div>
            </div>
          </div>

          {/* Per-Feature Drift Table */}
          {featureEntries.length > 0 && (
            <div className="glass-card" style={{ padding: '1.5rem' }}>
              <h2 style={{ fontSize: '1rem', fontWeight: 700, marginBottom: '1rem', color: 'var(--text-primary)' }}>
                🔬 Feature-Level Drift Scores
              </h2>
              <div style={{ overflowX: 'auto' }}>
                <table className="data-table" aria-label="Feature drift scores">
                  <thead>
                    <tr>
                      <th>Feature</th>
                      <th>Drift Score</th>
                      <th>Score Bar</th>
                      <th>Status</th>
                      <th>Details</th>
                    </tr>
                  </thead>
                  <tbody>
                    {featureEntries
                      .sort(([, a], [, b]) => (b.drift_score || 0) - (a.drift_score || 0))
                      .map(([feature, info]) => {
                        const score = info.drift_score || 0
                        const color = getDriftColor(score)
                        const barWidth = Math.min(score * 200, 100)  // scale for visibility
                        return (
                          <tr key={feature}>
                            <td style={{ fontFamily: 'var(--font-mono)', fontSize: '0.82rem', color: 'var(--text-secondary)', whiteSpace: 'nowrap' }}>
                              {feature}
                            </td>
                            <td style={{ fontFamily: 'var(--font-mono)', fontWeight: 700, color }}>
                              {score.toFixed(4)}
                            </td>
                            <td style={{ minWidth: 120 }}>
                              <div className="drift-score-bar">
                                <div className="progress-bar-track" style={{ flex: 1, height: 6 }}>
                                  <div
                                    className={`progress-bar-fill ${
                                      score < 0.15 ? 'progress-low'
                                      : score < 0.30 ? 'progress-medium'
                                      : 'progress-high'
                                    }`}
                                    style={{ width: `${barWidth}%` }}
                                  />
                                </div>
                              </div>
                            </td>
                            <td>
                              {info.drifted
                                ? <span className="risk-badge risk-high" style={{ fontSize: '0.72rem', padding: '3px 10px' }}>⚠️ Drifted</span>
                                : <span className="risk-badge risk-low"  style={{ fontSize: '0.72rem', padding: '3px 10px' }}>✅ Stable</span>}
                            </td>
                            <td style={{ fontSize: '0.75rem', color: 'var(--text-muted)', maxWidth: 200 }}>
                              {info.reference_mean !== undefined && (
                                <span>ref: {info.reference_mean} → cur: {info.current_mean}</span>
                              )}
                              {info.total_variation_distance !== undefined && (
                                <span>TVD: {info.total_variation_distance}</span>
                              )}
                            </td>
                          </tr>
                        )
                      })}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Metadata Footer */}
          <div style={{ marginTop: '1.5rem', display: 'flex', gap: '1rem', flexWrap: 'wrap', alignItems: 'center' }}>
            <span className="version-badge">
              🔍 {report.method || 'statistical'}
            </span>
            {report.timestamp && report.timestamp !== 'N/A' && (
              <span style={{ color: 'var(--text-muted)', fontSize: '0.75rem', fontFamily: 'var(--font-mono)' }}>
                Report: {report.timestamp}
              </span>
            )}
            {report.report_path && (
              <span style={{ color: 'var(--text-muted)', fontSize: '0.75rem', fontFamily: 'var(--font-mono)', wordBreak: 'break-all' }}>
                📄 {report.report_path.split('/').pop()}
              </span>
            )}
          </div>
        </>
      )}

      {/* Empty State */}
      {!report && !error && !loading && (
        <div className="glass-card" style={{ padding: '3rem', textAlign: 'center' }}>
          <div style={{ fontSize: 48, marginBottom: '1rem' }}>📭</div>
          <h2 style={{ color: 'var(--text-primary)', marginBottom: '0.5rem' }}>No Drift Report Found</h2>
          <p style={{ color: 'var(--text-secondary)', marginBottom: '1.5rem' }}>
            Generate your first drift report by running the monitoring pipeline.
          </p>
          <code style={{
            display: 'inline-block',
            background: 'rgba(255,255,255,0.05)',
            border: '1px solid var(--border-subtle)',
            borderRadius: 8,
            padding: '8px 16px',
            fontFamily: 'var(--font-mono)',
            fontSize: '0.875rem',
            color: 'var(--accent-cyan)',
          }}>
            make drift-report
          </code>
        </div>
      )}
    </div>
  )
}
