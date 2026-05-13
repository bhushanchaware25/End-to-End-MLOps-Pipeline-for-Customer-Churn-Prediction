"""
ChurnShield MLOps Platform - FastAPI Model Serving
===================================================
Production REST API serving the best registered churn prediction model
from the MLflow Model Registry.

Endpoints:
    POST /predict      — Churn probability + risk level
    GET  /health       — Service health + model status
    GET  /metrics      — Latest model performance metrics
    GET  /drift-report — Latest Evidently drift summary

Features:
    - CORS middleware (allows React frontend)
    - Global exception handler
    - Request/response structured logging middleware
    - Startup model loading from MLflow Production registry

Usage:
    uvicorn src.serving.app:app --host 0.0.0.0 --port 8000 --reload
"""

import json
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
from mlflow.tracking import MlflowClient
from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
)
logger.add(
    "logs/serving.log",
    rotation="10 MB",
    retention="30 days",
    level="DEBUG",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{line} - {message}",
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_REGISTRY_NAME: str = os.getenv("MLFLOW_MODEL_NAME", "ChurnShield-Model")
MODEL_STAGE: str = os.getenv("MODEL_STAGE", "Production")
DRIFT_REPORT_PATH: str = os.getenv("DRIFT_REPORT_PATH", "monitoring/reports")
ALLOWED_ORIGINS: list = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:3000,http://localhost:5173,http://127.0.0.1:3000",
).split(",")

# ---------------------------------------------------------------------------
# Global Model State
# ---------------------------------------------------------------------------
model_state: Dict[str, Any] = {
    "model": None,
    "model_version": "unknown",
    "model_name": "unknown",
    "loaded_at": None,
    "is_loaded": False,
}


# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------
class CustomerFeatures(BaseModel):
    """Input schema matching all Telco Customer Churn dataset features."""

    gender: str = Field(..., description="Customer gender", examples=["Male"])
    SeniorCitizen: int = Field(..., ge=0, le=1, description="1 if senior citizen", examples=[0])
    Partner: str = Field(..., description="Has a partner", examples=["Yes"])
    Dependents: str = Field(..., description="Has dependents", examples=["No"])
    tenure: int = Field(..., ge=0, description="Months with the company", examples=[12])
    PhoneService: str = Field(..., description="Has phone service", examples=["Yes"])
    MultipleLines: str = Field(
        ..., description="Has multiple lines", examples=["No phone service"]
    )
    InternetService: str = Field(
        ..., description="Internet service type", examples=["DSL"]
    )
    OnlineSecurity: str = Field(
        ..., description="Has online security", examples=["No"]
    )
    OnlineBackup: str = Field(..., description="Has online backup", examples=["Yes"])
    DeviceProtection: str = Field(
        ..., description="Has device protection", examples=["No"]
    )
    TechSupport: str = Field(..., description="Has tech support", examples=["No"])
    StreamingTV: str = Field(..., description="Streams TV", examples=["No"])
    StreamingMovies: str = Field(..., description="Streams movies", examples=["No"])
    Contract: str = Field(
        ..., description="Contract type", examples=["Month-to-month"]
    )
    PaperlessBilling: str = Field(
        ..., description="Uses paperless billing", examples=["Yes"]
    )
    PaymentMethod: str = Field(
        ..., description="Payment method", examples=["Electronic check"]
    )
    MonthlyCharges: float = Field(
        ..., gt=0, description="Monthly charge in USD", examples=[29.85]
    )
    TotalCharges: float = Field(
        ..., ge=0, description="Total charges to date in USD", examples=[357.20]
    )

    @field_validator("gender")
    @classmethod
    def validate_gender(cls, v: str) -> str:
        """Ensure gender is Male or Female."""
        if v not in {"Male", "Female"}:
            raise ValueError("gender must be 'Male' or 'Female'")
        return v

    @field_validator("Contract")
    @classmethod
    def validate_contract(cls, v: str) -> str:
        """Ensure contract type is one of the three valid values."""
        valid = {"Month-to-month", "One year", "Two year"}
        if v not in valid:
            raise ValueError(f"contract must be one of {valid}")
        return v

    model_config = {"json_schema_extra": {
        "example": {
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
            "TotalCharges": 357.20,
        }
    }}


class PredictionResponse(BaseModel):
    """Output schema for POST /predict."""

    churn_probability: float = Field(..., description="Probability of churn (0–1)")
    churn_prediction: bool = Field(..., description="True if predicted to churn")
    risk_level: str = Field(..., description="Risk level: Low | Medium | High")
    model_version: str = Field(..., description="Registered model version used")
    model_name: str = Field(..., description="Registered model name")
    prediction_id: str = Field(..., description="Unique ID for this prediction")
    timestamp: str = Field(..., description="ISO 8601 prediction timestamp")


class HealthResponse(BaseModel):
    """Output schema for GET /health."""

    status: str
    model_loaded: bool
    model_version: str
    model_name: str
    loaded_at: Optional[str]
    timestamp: str


class MetricsResponse(BaseModel):
    """Output schema for GET /metrics."""

    model_name: str
    model_version: str
    stage: str
    metrics: Dict[str, float]
    run_id: str
    timestamp: str


class DriftReportResponse(BaseModel):
    """Output schema for GET /drift-report."""

    drift_detected: bool
    drift_score: float
    method: str
    timestamp: str
    reference_rows: int
    current_rows: int
    report_path: Optional[str]
    features: Optional[Dict[str, Any]]


# ---------------------------------------------------------------------------
# Model Loading Utilities
# ---------------------------------------------------------------------------
def _determine_risk_level(probability: float) -> str:
    """
    Map a churn probability to a categorical risk level.

    Args:
        probability: Churn probability in [0, 1].

    Returns:
        str: 'Low' (< 0.3), 'Medium' (0.3–0.6), or 'High' (> 0.6).
    """
    if probability < 0.30:
        return "Low"
    elif probability < 0.60:
        return "Medium"
    return "High"


def _apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the same feature engineering used during training to inference input.

    Args:
        df: Raw customer features DataFrame (1 row).

    Returns:
        pd.DataFrame: DataFrame with engineered features appended.
    """
    df = df.copy()
    df["tenure_months_squared"] = df["tenure"].astype(float) ** 2
    df["avg_monthly_spend"] = df["TotalCharges"] / (df["tenure"].astype(float) + 1)

    addon_cols = [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    df["has_any_addon"] = (
        df[addon_cols].isin(["Yes"]).any(axis=1).astype(int)
    )

    service_cols = [
        "PhoneService", "MultipleLines", "InternetService",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    df["services_count"] = (
        df[service_cols].isin(["Yes", "DSL", "Fiber optic"]).sum(axis=1).astype(int)
    )

    df["is_long_term_contract"] = (
        df["Contract"].isin(["One year", "Two year"]).astype(int)
    )
    return df


def load_production_model() -> bool:
    """
    Load the Production-stage model from the MLflow Model Registry into global state.

    Returns:
        bool: True if loading succeeded, False otherwise.
    """
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

        # Find the Production version
        versions = client.search_model_versions(f"name='{MODEL_REGISTRY_NAME}'")
        prod_versions = [v for v in versions if v.current_stage == MODEL_STAGE]

        if not prod_versions:
            logger.warning(
                f"No '{MODEL_STAGE}' version found for '{MODEL_REGISTRY_NAME}'. "
                "Attempting to load latest version."
            )
            if not versions:
                raise RuntimeError(
                    f"No versions found for model '{MODEL_REGISTRY_NAME}'. "
                    "Run the training pipeline first."
                )
            prod_versions = [max(versions, key=lambda v: int(v.version))]

        latest_prod = max(prod_versions, key=lambda v: int(v.version))
        model_uri = f"models:/{MODEL_REGISTRY_NAME}/{latest_prod.version}"

        logger.info(f"Loading model: {model_uri}")
        model = mlflow.pyfunc.load_model(model_uri)

        model_state["model"] = model
        model_state["model_version"] = str(latest_prod.version)
        model_state["model_name"] = MODEL_REGISTRY_NAME
        model_state["loaded_at"] = datetime.utcnow().isoformat()
        model_state["is_loaded"] = True
        model_state["run_id"] = latest_prod.run_id

        logger.success(
            f"Model loaded: {MODEL_REGISTRY_NAME} v{latest_prod.version} "
            f"(run_id={latest_prod.run_id[:8]}...)"
        )
        return True

    except Exception as exc:
        logger.error(f"Failed to load model from MLflow registry: {exc}")
        model_state["is_loaded"] = False
        return False


def get_model_metrics_from_mlflow() -> Dict[str, Any]:
    """
    Fetch the production model's logged metrics from the MLflow run.

    Returns:
        Dict with run_id, metrics, and model info. Returns defaults on failure.
    """
    try:
        client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
        run_id = model_state.get("run_id", "")
        if not run_id:
            return {"error": "No run_id available", "metrics": {}}
        run = client.get_run(run_id)
        return {
            "run_id": run_id,
            "metrics": dict(run.data.metrics),
            "model_type": run.data.tags.get("model_type", "Unknown"),
        }
    except Exception as exc:
        logger.warning(f"Could not fetch metrics from MLflow: {exc}")
        return {"run_id": "", "metrics": {}, "model_type": "Unknown"}


def get_latest_drift_report() -> Dict[str, Any]:
    """
    Load the most recent drift summary JSON from the monitoring/reports/ directory.

    Returns:
        Dict with drift info. Returns a 'no_report' status dict if none found.
    """
    reports_dir = Path(DRIFT_REPORT_PATH)
    if not reports_dir.exists():
        return {
            "drift_detected": False,
            "drift_score": 0.0,
            "method": "none",
            "status": "no_reports_directory",
            "timestamp": "N/A",
            "reference_rows": 0,
            "current_rows": 0,
        }

    summary_files = sorted(reports_dir.glob("drift_summary_*.json"), reverse=True)
    if not summary_files:
        return {
            "drift_detected": False,
            "drift_score": 0.0,
            "method": "none",
            "status": "no_reports_found",
            "timestamp": "N/A",
            "reference_rows": 0,
            "current_rows": 0,
        }

    latest = summary_files[0]
    try:
        with open(latest) as f:
            data = json.load(f)
        data["report_file"] = str(latest.name)
        return data
    except Exception as exc:
        logger.error(f"Failed to parse drift report {latest}: {exc}")
        return {"drift_detected": False, "drift_score": 0.0, "error": str(exc)}


# ---------------------------------------------------------------------------
# Application Lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager for startup and shutdown events.
    Loads the Production model on startup.
    """
    logger.info("=" * 60)
    logger.info("ChurnShield API — Starting Up")
    logger.info(f"MLflow URI  : {MLFLOW_TRACKING_URI}")
    logger.info(f"Model name  : {MODEL_REGISTRY_NAME}")
    logger.info(f"Model stage : {MODEL_STAGE}")
    logger.info("=" * 60)

    Path("logs").mkdir(exist_ok=True)
    success = load_production_model()
    if not success:
        logger.warning(
            "Model could not be loaded on startup. "
            "/predict will return 503 until model is available."
        )

    yield  # Application runs here

    logger.info("ChurnShield API — Shutting Down")


# ---------------------------------------------------------------------------
# FastAPI Application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="ChurnShield MLOps API",
    description=(
        "Production REST API for Telco Customer Churn Prediction. "
        "Serves the best registered model from the MLflow Model Registry."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# CORS Middleware
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Request/Response Logging Middleware
# ---------------------------------------------------------------------------
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """
    Log every incoming request and outgoing response with timing.

    Args:
        request: Incoming HTTP request.
        call_next: ASGI call chain handler.

    Returns:
        Response with X-Request-ID and X-Process-Time headers added.
    """
    request_id = str(uuid.uuid4())[:8]
    start_time = time.perf_counter()

    logger.info(
        f"→ [{request_id}] {request.method} {request.url.path} "
        f"| client={request.client.host if request.client else 'unknown'}"
    )

    response = await call_next(request)

    process_time_ms = (time.perf_counter() - start_time) * 1000
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time"] = f"{process_time_ms:.1f}ms"

    logger.info(
        f"← [{request_id}] {response.status_code} | {process_time_ms:.1f}ms"
    )
    return response


# ---------------------------------------------------------------------------
# Global Exception Handler
# ---------------------------------------------------------------------------
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Catch-all exception handler returning structured JSON error responses.

    Args:
        request: The HTTP request that caused the exception.
        exc: The unhandled exception.

    Returns:
        JSONResponse with status 500 and error details.
    """
    logger.error(
        f"Unhandled exception on {request.method} {request.url.path}: "
        f"{type(exc).__name__}: {exc}",
        exc_info=True,
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "path": str(request.url.path),
            "timestamp": datetime.utcnow().isoformat(),
        },
    )


# ---------------------------------------------------------------------------
# Endpoint: GET /health
# ---------------------------------------------------------------------------
@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    tags=["Operations"],
)
async def health_check() -> HealthResponse:
    """
    Return the health status of the API and model loading state.

    Returns:
        HealthResponse with status, model_loaded flag, and model metadata.
    """
    return HealthResponse(
        status="healthy" if model_state["is_loaded"] else "degraded",
        model_loaded=model_state["is_loaded"],
        model_version=model_state["model_version"],
        model_name=model_state["model_name"],
        loaded_at=model_state["loaded_at"],
        timestamp=datetime.utcnow().isoformat(),
    )


# ---------------------------------------------------------------------------
# Endpoint: POST /predict
# ---------------------------------------------------------------------------
@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict customer churn",
    tags=["Prediction"],
    status_code=status.HTTP_200_OK,
)
async def predict_churn(customer: CustomerFeatures) -> PredictionResponse:
    """
    Predict churn probability for a single customer.

    Applies feature engineering, runs the Production model, and returns
    the churn probability, binary prediction, and color-coded risk level.

    Args:
        customer: CustomerFeatures Pydantic model with all Telco features.

    Returns:
        PredictionResponse with probability, prediction, risk level, and metadata.

    Raises:
        HTTPException 503: If the model is not loaded.
        HTTPException 422: If input validation fails (handled by FastAPI).
        HTTPException 500: If prediction fails unexpectedly.
    """
    if not model_state["is_loaded"] or model_state["model"] is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Model not loaded. The MLflow Production model could not be found. "
                "Run the training pipeline first."
            ),
        )

    try:
        # -- Build raw feature DataFrame
        input_dict = customer.model_dump()
        df_raw = pd.DataFrame([input_dict])

        # -- Apply feature engineering (same as training)
        df_engineered = _apply_feature_engineering(df_raw)

        # -- Drop non-feature columns
        drop_cols = ["customerID"] if "customerID" in df_engineered.columns else []
        df_input = df_engineered.drop(columns=drop_cols, errors="ignore")

        # -- Predict
        proba_array = model_state["model"].predict_proba(df_input)
        churn_probability = float(proba_array[0][1])
        churn_prediction = churn_probability >= 0.5
        risk_level = _determine_risk_level(churn_probability)

        logger.info(
            f"Prediction: prob={churn_probability:.4f}, "
            f"prediction={churn_prediction}, risk={risk_level} | "
            f"tenure={customer.tenure}, contract={customer.Contract}"
        )

        return PredictionResponse(
            churn_probability=round(churn_probability, 6),
            churn_prediction=churn_prediction,
            risk_level=risk_level,
            model_version=model_state["model_version"],
            model_name=model_state["model_name"],
            prediction_id=str(uuid.uuid4()),
            timestamp=datetime.utcnow().isoformat(),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Prediction failed: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(exc)}",
        )


# ---------------------------------------------------------------------------
# Endpoint: GET /metrics
# ---------------------------------------------------------------------------
@app.get(
    "/metrics",
    response_model=MetricsResponse,
    summary="Get latest model performance metrics",
    tags=["Operations"],
)
async def get_metrics() -> MetricsResponse:
    """
    Retrieve the production model's logged performance metrics from MLflow.

    Returns:
        MetricsResponse with accuracy, precision, recall, f1_score, roc_auc.

    Raises:
        HTTPException 503: If model is not loaded.
        HTTPException 500: If MLflow query fails.
    """
    if not model_state["is_loaded"]:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Cannot retrieve metrics.",
        )

    try:
        mlflow_data = get_model_metrics_from_mlflow()
        return MetricsResponse(
            model_name=model_state["model_name"],
            model_version=model_state["model_version"],
            stage=MODEL_STAGE,
            metrics=mlflow_data.get("metrics", {}),
            run_id=mlflow_data.get("run_id", ""),
            timestamp=datetime.utcnow().isoformat(),
        )
    except Exception as exc:
        logger.error(f"Metrics fetch failed: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve metrics: {str(exc)}",
        )


# ---------------------------------------------------------------------------
# Endpoint: GET /drift-report
# ---------------------------------------------------------------------------
@app.get(
    "/drift-report",
    response_model=DriftReportResponse,
    summary="Get latest drift monitoring report",
    tags=["Monitoring"],
)
async def get_drift_report() -> DriftReportResponse:
    """
    Return the most recent data drift report summary from Evidently AI monitoring.

    Returns:
        DriftReportResponse with drift_detected, drift_score, and feature scores.

    Raises:
        HTTPException 500: If report parsing fails.
    """
    try:
        report = get_latest_drift_report()
        return DriftReportResponse(
            drift_detected=bool(report.get("drift_detected", False)),
            drift_score=float(report.get("drift_score", 0.0)),
            method=str(report.get("method", "none")),
            timestamp=str(report.get("timestamp", "N/A")),
            reference_rows=int(report.get("reference_rows", 0)),
            current_rows=int(report.get("current_rows", 0)),
            report_path=report.get("summary_path"),
            features=report.get("feature_drift"),
        )
    except Exception as exc:
        logger.error(f"Drift report fetch failed: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve drift report: {str(exc)}",
        )


# ---------------------------------------------------------------------------
# Endpoint: POST /reload-model
# ---------------------------------------------------------------------------
@app.post(
    "/reload-model",
    summary="Reload Production model from registry",
    tags=["Operations"],
)
async def reload_model() -> Dict[str, Any]:
    """
    Trigger a hot-reload of the Production model from the MLflow Model Registry.
    Useful after a new model version has been promoted to Production.

    Returns:
        Dict with success status and new model version.
    """
    logger.info("Model reload requested via API.")
    success = load_production_model()
    if success:
        return {
            "status": "success",
            "message": "Model reloaded successfully.",
            "model_version": model_state["model_version"],
            "loaded_at": model_state["loaded_at"],
        }
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="Model reload failed. Check MLflow registry connection.",
    )
