import { useState } from 'react'
import axios from 'axios'

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

const FIELD_DEFS = [
  { key: 'gender',          label: 'Gender',           type: 'select', options: ['Male', 'Female'] },
  { key: 'SeniorCitizen',   label: 'Senior Citizen',   type: 'select', options: [{ v: 0, l: 'No' }, { v: 1, l: 'Yes' }] },
  { key: 'Partner',         label: 'Has Partner',      type: 'select', options: ['Yes', 'No'] },
  { key: 'Dependents',      label: 'Has Dependents',   type: 'select', options: ['Yes', 'No'] },
  { key: 'tenure',          label: 'Tenure (months)',  type: 'number', min: 0, max: 120 },
  { key: 'PhoneService',    label: 'Phone Service',    type: 'select', options: ['Yes', 'No'] },
  { key: 'MultipleLines',   label: 'Multiple Lines',   type: 'select', options: ['Yes', 'No', 'No phone service'] },
  { key: 'InternetService', label: 'Internet Service', type: 'select', options: ['DSL', 'Fiber optic', 'No'] },
  { key: 'OnlineSecurity',  label: 'Online Security',  type: 'select', options: ['Yes', 'No', 'No internet service'] },
  { key: 'OnlineBackup',    label: 'Online Backup',    type: 'select', options: ['Yes', 'No', 'No internet service'] },
  { key: 'DeviceProtection',label: 'Device Protection',type: 'select', options: ['Yes', 'No', 'No internet service'] },
  { key: 'TechSupport',     label: 'Tech Support',     type: 'select', options: ['Yes', 'No', 'No internet service'] },
  { key: 'StreamingTV',     label: 'Streaming TV',     type: 'select', options: ['Yes', 'No', 'No internet service'] },
  { key: 'StreamingMovies', label: 'Streaming Movies', type: 'select', options: ['Yes', 'No', 'No internet service'] },
  { key: 'Contract',        label: 'Contract',         type: 'select', options: ['Month-to-month', 'One year', 'Two year'] },
  { key: 'PaperlessBilling',label: 'Paperless Billing',type: 'select', options: ['Yes', 'No'] },
  { key: 'PaymentMethod',   label: 'Payment Method',   type: 'select', options: ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'] },
  { key: 'MonthlyCharges',  label: 'Monthly Charges ($)', type: 'number', min: 0, step: 0.01 },
  { key: 'TotalCharges',    label: 'Total Charges ($)', type: 'number', min: 0, step: 0.01 },
]

const DEFAULT_VALUES = {
  gender: 'Male', SeniorCitizen: 0, Partner: 'Yes', Dependents: 'No',
  tenure: 12, PhoneService: 'Yes', MultipleLines: 'No',
  InternetService: 'DSL', OnlineSecurity: 'No', OnlineBackup: 'Yes',
  DeviceProtection: 'No', TechSupport: 'No', StreamingTV: 'No',
  StreamingMovies: 'No', Contract: 'Month-to-month', PaperlessBilling: 'Yes',
  PaymentMethod: 'Electronic check', MonthlyCharges: 29.85, TotalCharges: 357.20,
}

const RISK_CONFIG = {
  Low:    { icon: '✅', className: 'risk-low',    progressClass: 'progress-low',    emoji: '🟢' },
  Medium: { icon: '⚠️', className: 'risk-medium', progressClass: 'progress-medium', emoji: '🟡' },
  High:   { icon: '🚨', className: 'risk-high',   progressClass: 'progress-high',   emoji: '🔴' },
}

/**
 * PredictionForm — Customer churn prediction page.
 * Submits all 19 Telco features to POST /predict and displays
 * the churn probability as a progress bar with a color-coded risk badge.
 */
export default function PredictionForm() {
  const [formData, setFormData]   = useState(DEFAULT_VALUES)
  const [result, setResult]       = useState(null)
  const [loading, setLoading]     = useState(false)
  const [error, setError]         = useState(null)

  const handleChange = (key, value) => {
    setFormData(prev => ({
      ...prev,
      [key]: ['SeniorCitizen'].includes(key) ? Number(value)
           : ['tenure', 'MonthlyCharges', 'TotalCharges'].includes(key) ? parseFloat(value) || 0
           : value,
    }))
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const { data } = await axios.post(`${API_BASE}/predict`, formData, {
        headers: { 'Content-Type': 'application/json' },
        timeout: 15000,
      })
      setResult(data)
    } catch (err) {
      const detail = err.response?.data?.detail || err.message || 'Unknown error'
      setError(`Prediction failed: ${detail}`)
    } finally {
      setLoading(false)
    }
  }

  const handleReset = () => {
    setFormData(DEFAULT_VALUES)
    setResult(null)
    setError(null)
  }

  const riskCfg = result ? RISK_CONFIG[result.risk_level] : null
  const probPct = result ? Math.round(result.churn_probability * 100) : 0

  const renderField = (field) => {
    const value = formData[field.key]
    if (field.type === 'select') {
      return (
        <select
          id={`field-${field.key}`}
          className="form-select"
          value={value}
          onChange={(e) => handleChange(field.key, e.target.value)}
        >
          {field.options.map((opt) => {
            const v = typeof opt === 'object' ? opt.v : opt
            const l = typeof opt === 'object' ? opt.l : opt
            return <option key={v} value={v}>{l}</option>
          })}
        </select>
      )
    }
    return (
      <input
        id={`field-${field.key}`}
        type="number"
        className="form-input"
        value={value}
        min={field.min}
        max={field.max}
        step={field.step || 1}
        onChange={(e) => handleChange(field.key, e.target.value)}
      />
    )
  }

  return (
    <div>
      {/* Page Header */}
      <div className="page-header">
        <h1 className="page-title">🎯 Predict Customer Churn</h1>
        <p className="page-subtitle">
          Enter customer details to get a real-time churn probability prediction from the Production model.
        </p>
      </div>

      {/* Form Card */}
      <div className="glass-card" style={{ padding: '2rem' }}>
        <form id="prediction-form" onSubmit={handleSubmit} noValidate>
          <div className="form-grid">
            {FIELD_DEFS.map((field) => (
              <div key={field.key} className="form-group">
                <label className="form-label" htmlFor={`field-${field.key}`}>
                  {field.label}
                </label>
                {renderField(field)}
              </div>
            ))}
          </div>

          <hr className="section-divider" />

          <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap' }}>
            <button
              id="btn-predict"
              type="submit"
              className="btn btn-primary"
              disabled={loading}
            >
              {loading ? (
                <>
                  <span className="spinner" style={{ width: 18, height: 18, borderWidth: 2 }} />
                  Predicting…
                </>
              ) : '🔍 Predict Churn'}
            </button>
            <button
              id="btn-reset"
              type="button"
              className="btn btn-secondary"
              onClick={handleReset}
              disabled={loading}
            >
              ↺ Reset
            </button>
          </div>
        </form>
      </div>

      {/* Error */}
      {error && (
        <div className="alert alert-danger" role="alert" style={{ marginTop: '1.5rem' }}>
          <span className="alert-icon">❌</span>
          <div>
            <div className="alert-title">Prediction Error</div>
            {error}
          </div>
        </div>
      )}

      {/* Result Panel */}
      {result && riskCfg && (
        <div className="glass-card result-panel" id="prediction-result">
          <div className="result-header">
            <div className="result-title">Prediction Result</div>
            <span className={`risk-badge ${riskCfg.className}`}>
              {riskCfg.icon} {result.risk_level} Risk
            </span>
          </div>

          {/* Churn Probability Progress Bar */}
          <div className="progress-container">
            <div className="progress-label">
              <span>Churn Probability</span>
              <strong style={{ color: 'var(--text-primary)', fontFamily: 'var(--font-mono)' }}>
                {probPct}%
              </strong>
            </div>
            <div className="progress-bar-track">
              <div
                className={`progress-bar-fill ${riskCfg.progressClass}`}
                style={{ width: `${probPct}%` }}
                role="progressbar"
                aria-valuenow={probPct}
                aria-valuemin={0}
                aria-valuemax={100}
              />
            </div>
          </div>

          {/* Stats Row */}
          <div className="stat-row" style={{ marginTop: '1.5rem' }}>
            <div className="stat-item">
              <div className="stat-value" style={{ color: result.churn_prediction ? 'var(--risk-high)' : 'var(--risk-low)' }}>
                {result.churn_prediction ? 'Yes' : 'No'}
              </div>
              <div className="stat-label">Will Churn</div>
            </div>
            <div className="stat-item">
              <div className="stat-value" style={{ fontFamily: 'var(--font-mono)', fontSize: '1.3rem' }}>
                {(result.churn_probability * 100).toFixed(1)}%
              </div>
              <div className="stat-label">Probability</div>
            </div>
            <div className="stat-item">
              <div className="stat-value" style={{ fontFamily: 'var(--font-mono)', fontSize: '1.1rem' }}>
                v{result.model_version}
              </div>
              <div className="stat-label">Model Version</div>
            </div>
            <div className="stat-item">
              <div className="stat-value" style={{ fontSize: '1.1rem' }}>
                {riskCfg.emoji}
              </div>
              <div className="stat-label">Risk Level</div>
            </div>
          </div>

          {/* Metadata */}
          <div style={{ marginTop: '1.2rem', display: 'flex', alignItems: 'center', gap: '0.75rem', flexWrap: 'wrap' }}>
            <span className="version-badge">🤖 {result.model_name}</span>
            <span style={{ color: 'var(--text-muted)', fontSize: '0.75rem', fontFamily: 'var(--font-mono)' }}>
              ID: {result.prediction_id?.substring(0, 8)}…
            </span>
            <span style={{ color: 'var(--text-muted)', fontSize: '0.75rem' }}>
              {new Date(result.timestamp).toLocaleString()}
            </span>
          </div>
        </div>
      )}
    </div>
  )
}
