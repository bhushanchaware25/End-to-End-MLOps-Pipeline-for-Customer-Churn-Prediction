# ============================================================
# ChurnShield MLOps Platform - Makefile
# ============================================================
# Usage: make <target>
# ============================================================

.PHONY: help install install-dev train serve test lint format \
        docker-up docker-down docker-build clean dvc-init \
        pipeline drift-report

# --- Default target ---
help:
	@echo ""
	@echo "╔══════════════════════════════════════════════════════╗"
	@echo "║       ChurnShield MLOps Platform - Make Commands     ║"
	@echo "╚══════════════════════════════════════════════════════╝"
	@echo ""
	@echo "  Setup:"
	@echo "    make install        Install production dependencies"
	@echo "    make install-dev    Install all dependencies (incl. dev)"
	@echo "    make dvc-init       Initialize DVC and add data"
	@echo ""
	@echo "  Pipeline:"
	@echo "    make train          Run the full training pipeline"
	@echo "    make pipeline       Run Prefect training pipeline"
	@echo "    make drift-report   Generate Evidently drift report"
	@echo ""
	@echo "  Serving:"
	@echo "    make serve          Start FastAPI server (local)"
	@echo ""
	@echo "  Quality:"
	@echo "    make test           Run all pytest tests with coverage"
	@echo "    make lint           Run flake8 linting"
	@echo "    make format         Run black code formatter"
	@echo ""
	@echo "  Docker:"
	@echo "    make docker-build   Build all Docker images"
	@echo "    make docker-up      Start all services (docker-compose)"
	@echo "    make docker-down    Stop all services"
	@echo ""
	@echo "  Cleanup:"
	@echo "    make clean          Remove build artifacts and cache"
	@echo ""

# --- Setup ---
install:
	pip install --upgrade pip
	pip install -r requirements.txt

install-dev:
	pip install --upgrade pip
	pip install -r requirements-dev.txt
	pre-commit install

dvc-init:
	dvc init
	dvc add data/raw/telco_churn.csv
	git add data/raw/telco_churn.csv.dvc .gitignore
	@echo "DVC initialized. Configure remote with: dvc remote add -d myremote <url>"

# --- Training ---
train:
	@echo "Starting ChurnShield training pipeline..."
	python -m src.ingestion.ingest
	python -m src.validation.validate
	python -m src.preprocessing.preprocess
	python -m src.training.train
	python -m src.evaluation.evaluate

pipeline:
	@echo "Running Prefect orchestration pipeline..."
	python pipelines/training_pipeline.py

drift-report:
	@echo "Generating Evidently drift report..."
	python -m src.monitoring.monitor

# --- Serving ---
serve:
	@echo "Starting FastAPI server at http://localhost:8000"
	@echo "API Docs: http://localhost:8000/docs"
	uvicorn src.serving.app:app --host 0.0.0.0 --port 8000 --reload

# --- Quality ---
test:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing

lint:
	flake8 src/ tests/ pipelines/ --max-line-length=88 --exclude=__pycache__

format:
	black src/ tests/ pipelines/ --line-length=88
	isort src/ tests/ pipelines/

# --- Docker ---
docker-build:
	docker build -f docker/Dockerfile.api -t churnshield-api:latest .
	docker build -f docker/Dockerfile.frontend -t churnshield-frontend:latest .

docker-up:
	docker-compose up -d
	@echo ""
	@echo "Services started:"
	@echo "  FastAPI:   http://localhost:8000"
	@echo "  FastAPI Docs: http://localhost:8000/docs"
	@echo "  MLflow UI: http://localhost:5000"
	@echo "  Prefect UI: http://localhost:4200"
	@echo "  Frontend:  http://localhost:3000"

docker-down:
	docker-compose down -v
	@echo "All services stopped."

# --- Cleanup ---
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name "htmlcov" -exec rm -rf {} +
	find . -type f -name ".coverage" -delete
	@echo "Cleanup complete."
