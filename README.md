# 🛡️ ChurnShield MLOps Platform

<div align="center">

**End-to-End Production MLOps Pipeline for Telecom Customer Churn Prediction**

[![CI](https://github.com/YOUR_USERNAME/churnshield-mlops/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/churnshield-mlops/actions/workflows/ci.yml)
[![CD](https://github.com/YOUR_USERNAME/churnshield-mlops/actions/workflows/cd.yml/badge.svg)](https://github.com/YOUR_USERNAME/churnshield-mlops/actions/workflows/cd.yml)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![MLflow](https://img.shields.io/badge/MLflow-2.12-0194E2?logo=mlflow)](https://mlflow.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## 📌 Business Problem

Telecom companies lose significant revenue to customer churn. ChurnShield predicts which customers are likely to leave, enabling proactive retention campaigns. This project demonstrates not just a churn model — but a **complete production system** around it: automated pipelines, experiment tracking, model serving, drift monitoring, and CI/CD.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    ChurnShield MLOps Platform                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────┐    ┌──────────┐    ┌────────────┐    ┌─────────┐  │
│  │  Data    │───▶│Validation│───▶│Preprocess  │───▶│Training │  │
│  │Ingestion │    │   (GE)   │    │(sklearn)   │    │(MLflow) │  │
│  └──────────┘    └──────────┘    └────────────┘    └────┬────┘  │
│                                                          │        │
│  ┌──────────┐    ┌──────────┐    ┌────────────┐    ┌────▼────┐  │
│  │  React   │◀───│ FastAPI  │◀───│  MLflow    │◀───│Evaluate │  │
│  │Dashboard │    │  /predict│    │  Registry  │    │+Register│  │
│  └──────────┘    └──────────┘    └────────────┘    └─────────┘  │
│                       │                                           │
│  ┌──────────┐    ┌────▼─────┐                                    │
│  │Evidently │◀───│  Drift   │◀─── Scheduled Weekly (Prefect)    │
│  │  Report  │    │ Monitor  │                                     │
│  └──────────┘    └──────────┘                                    │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  GitHub Actions: CI (lint+test) → CD (build+push GHCR)  │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

### Prefect Pipeline Flow
```
ingest_data ──▶ validate_data ──▶ preprocess_data ──▶ train_models
                                                              │
                                                              ▼
                                          evaluate_and_register ──▶ generate_drift_report
```

---

## 🛠️ Tech Stack

| Category | Technology |
|---|---|
| **Language** | Python 3.10+ |
| **Data Validation** | Great Expectations |
| **Data Versioning** | DVC |
| **Experiment Tracking** | MLflow + PostgreSQL backend |
| **Pipeline Orchestration** | Prefect 2.x |
| **Model Training** | Scikit-learn, XGBoost |
| **Model Serving** | FastAPI + Uvicorn + Pydantic v2 |
| **Drift Monitoring** | Evidently AI |
| **Containerization** | Docker + Docker Compose |
| **CI/CD** | GitHub Actions → GHCR |
| **Testing** | Pytest + Coverage |
| **Code Quality** | Black, Flake8, isort, pre-commit |
| **Frontend** | React 18 + Vite + Axios |

---

## 📁 Project Structure

```
MLOps/
├── data/
│   ├── raw/                    # Raw Telco CSV (DVC tracked)
│   ├── processed/              # Cleaned + split datasets
│   └── reference/              # Reference dataset for drift detection
├── src/
│   ├── ingestion/ingest.py     # Load/generate data, clean, save
│   ├── validation/validate.py  # 30+ Great Expectations checks
│   ├── preprocessing/preprocess.py  # Feature eng + sklearn Pipeline
│   ├── training/train.py       # 3 models + MLflow logging
│   ├── evaluation/evaluate.py  # Best model → Registry → Production
│   ├── serving/app.py          # FastAPI REST API
│   └── monitoring/monitor.py   # Evidently drift detection
├── pipelines/
│   └── training_pipeline.py   # Prefect flow (all 6 stages)
├── tests/                      # Pytest test suites (70%+ coverage)
├── frontend/                   # React.js Vite dashboard
├── docker/                     # Multi-stage Dockerfiles
├── .github/workflows/          # CI + CD GitHub Actions
├── dvc.yaml                    # DVC pipeline stages
├── docker-compose.yml          # Full 5-service stack
├── Makefile                    # Developer shortcuts
└── README.md
```

---

## ✅ Prerequisites

- Python 3.10+
- Docker + Docker Compose
- Node.js 20+ (for frontend development)
- Git

---

## 🚀 Quick Start

### Option A: Docker Compose (Recommended)

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/churnshield-mlops.git
cd churnshield-mlops

# 2. Copy and configure environment variables
cp .env.example .env
# Edit .env with your settings (defaults work for local development)

# 3. Start all services
make docker-up

# 4. Run the training pipeline inside the API container
docker exec churnshield-api python -m src.ingestion.ingest --synthetic
docker exec churnshield-api python pipelines/training_pipeline.py --synthetic
```

### Option B: Local Development

```bash
# 1. Install dependencies
make install-dev

# 2. Generate synthetic data and run full pipeline
make train

# 3. Start the FastAPI server
make serve

# 4. Start the React frontend (new terminal)
cd frontend && npm install && npm run dev
```

---

## 🔄 Running the Full Pipeline

### Step by Step

```bash
# Step 1: Data Ingestion (generates synthetic Telco data if no CSV)
python -m src.ingestion.ingest

# Step 2: Data Validation (30+ Great Expectations checks)
python -m src.validation.validate

# Step 3: Feature Engineering + Preprocessing
python -m src.preprocessing.preprocess

# Step 4: Train 3 Models → Log to MLflow
python -m src.training.train

# Step 5: Evaluate + Register Best Model → Production
python -m src.evaluation.evaluate

# Step 6: Generate Drift Report
python -m src.monitoring.monitor
```

### Or run everything at once:
```bash
make train
```

### Or via Prefect:
```bash
# Run the full Prefect flow
python pipelines/training_pipeline.py

# With synthetic data
python pipelines/training_pipeline.py --synthetic

# Create weekly deployment
python pipelines/training_pipeline.py --deploy
```

---

## 🌐 Service URLs

| Service | URL | Description |
|---|---|---|
| **FastAPI** | http://localhost:8000 | Prediction REST API |
| **FastAPI Docs** | http://localhost:8000/docs | Swagger UI |
| **MLflow UI** | http://localhost:5000 | Experiment tracking |
| **Prefect UI** | http://localhost:4200 | Pipeline orchestration |
| **React App** | http://localhost:3000 | Dashboard |

---

## 📡 API Reference

### `POST /predict`

Predict churn probability for a customer.

**Request:**
```json
{
  "gender": "Male",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 12,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "DSL",
  "OnlineSecurity": "No",
  "OnlineBackup": "Yes",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "No",
  "StreamingMovies": "No",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 29.85,
  "TotalCharges": 357.20
}
```

**Response:**
```json
{
  "churn_probability": 0.7234,
  "churn_prediction": true,
  "risk_level": "High",
  "model_version": "3",
  "model_name": "ChurnShield-Model",
  "prediction_id": "a1b2c3d4-...",
  "timestamp": "2025-01-15T10:30:00.000Z"
}
```

### `GET /health`
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_version": "3",
  "model_name": "ChurnShield-Model",
  "loaded_at": "2025-01-15T10:00:00Z",
  "timestamp": "2025-01-15T10:30:00Z"
}
```

### `GET /metrics`
Returns production model metrics: `accuracy`, `precision`, `recall`, `f1_score`, `roc_auc`.

### `GET /drift-report`
Returns latest Evidently drift analysis: `drift_detected`, `drift_score`, per-feature scores.

---

## 🔧 Developer Commands

```bash
make install        # Install production dependencies
make install-dev    # Install all dependencies (incl. dev + pre-commit)
make train          # Run full training pipeline
make serve          # Start FastAPI server
make test           # Run tests with coverage report
make lint           # Run flake8 linting
make format         # Run black + isort formatting
make docker-up      # Start all Docker services
make docker-down    # Stop all Docker services
make drift-report   # Generate Evidently drift report
```

---

## ⚙️ CI/CD Pipeline

### Continuous Integration (every push/PR)
1. **Lint** — Black format check + isort + Flake8
2. **Test** — Pytest on Python 3.10 & 3.11 with 70% coverage gate
3. **Coverage** — Upload to Codecov
4. **Docker Validate** — Build (no push) on PRs

### Continuous Deployment (merge to `main`)
1. **Test Gate** — Full test suite must pass
2. **Build API** — Multi-stage Docker build → push to `ghcr.io`
3. **Build Frontend** — Vite build → Nginx image → push to `ghcr.io`
4. **Release** — GitHub Release created on tags

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_serving.py -v

# Open HTML coverage report
open htmlcov/index.html
```

---

## 📸 Screenshots

> **Add screenshots here after running the application.**

| Page | Description |
|---|---|
| `screenshots/prediction_form.png` | Customer churn prediction form |
| `screenshots/metrics_dashboard.png` | Model performance metrics |
| `screenshots/drift_monitor.png` | Drift detection dashboard |
| `screenshots/mlflow_ui.png` | MLflow experiment tracking |

---

## 🔮 Future Improvements

- [ ] Add SHAP explainability endpoint (`/explain`)
- [ ] Implement A/B testing between model versions
- [ ] Add Prometheus + Grafana monitoring stack
- [ ] Kubernetes deployment manifests (Helm chart)
- [ ] Add hyperparameter tuning with Optuna
- [ ] Real-time streaming predictions via Kafka
- [ ] Multi-tenant support for different business units
- [ ] Model retraining trigger via webhook on drift detection

---

## 📄 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

<div align="center">

Built with ❤️ as a production-grade MLOps portfolio project.

**[FastAPI](https://fastapi.tiangolo.com) · [MLflow](https://mlflow.org) · [Prefect](https://prefect.io) · [Evidently](https://evidentlyai.com) · [DVC](https://dvc.org)**

</div>
