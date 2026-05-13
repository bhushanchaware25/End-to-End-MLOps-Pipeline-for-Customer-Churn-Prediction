"""
Tests for src/serving/app.py
==============================
Tests all FastAPI endpoints using TestClient and mocked MLflow model loading.

Tests cover:
  - GET /health (model loaded + unloaded states)
  - POST /predict (valid input, unloaded model, validation errors)
  - GET /metrics (success + unloaded state)
  - GET /drift-report (no reports dir, with mock report)
  - GET /docs (OpenAPI schema available)
  - Risk level logic (_determine_risk_level)
  - Feature engineering (_apply_feature_engineering)
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Fixtures & Helpers
# ---------------------------------------------------------------------------
VALID_CUSTOMER: Dict[str, Any] = {
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


def _make_mock_model(churn_prob: float = 0.75) -> MagicMock:
    """Create a mock MLflow pyfunc model that returns a fixed churn probability."""
    mock_model = MagicMock()
    mock_model.predict_proba.return_value = np.array([[1 - churn_prob, churn_prob]])
    return mock_model


@pytest.fixture
def client_with_model():
    """TestClient with a mocked loaded model (high churn probability)."""
    from src.serving.app import app, model_state

    original_state = model_state.copy()
    model_state.update({
        "model": _make_mock_model(churn_prob=0.75),
        "model_version": "3",
        "model_name": "ChurnShield-Model",
        "loaded_at": "2025-01-01T00:00:00",
        "is_loaded": True,
        "run_id": "abc123def456",
    })

    with TestClient(app, raise_server_exceptions=False) as c:
        yield c

    # Restore original state
    model_state.update(original_state)


@pytest.fixture
def client_no_model():
    """TestClient with no model loaded (simulates startup failure)."""
    from src.serving.app import app, model_state

    original_state = model_state.copy()
    model_state.update({
        "model": None,
        "model_version": "unknown",
        "model_name": "unknown",
        "loaded_at": None,
        "is_loaded": False,
    })

    with TestClient(app, raise_server_exceptions=False) as c:
        yield c

    model_state.update(original_state)


# ---------------------------------------------------------------------------
# Test: GET /health
# ---------------------------------------------------------------------------
class TestHealthEndpoint:
    def test_health_when_model_loaded(self, client_with_model: TestClient) -> None:
        """Health endpoint should return 'healthy' when model is loaded."""
        resp = client_with_model.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        assert data["model_version"] == "3"

    def test_health_when_model_not_loaded(self, client_no_model: TestClient) -> None:
        """Health endpoint should return 'degraded' when model is not loaded."""
        resp = client_no_model.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "degraded"
        assert data["model_loaded"] is False

    def test_health_has_timestamp(self, client_with_model: TestClient) -> None:
        """Health response must include a timestamp field."""
        resp = client_with_model.get("/health")
        assert "timestamp" in resp.json()

    def test_health_has_required_keys(self, client_with_model: TestClient) -> None:
        """Health response must have all required schema fields."""
        resp = client_with_model.get("/health")
        data = resp.json()
        for key in ["status", "model_loaded", "model_version", "model_name", "timestamp"]:
            assert key in data, f"Missing key: {key}"


# ---------------------------------------------------------------------------
# Test: POST /predict
# ---------------------------------------------------------------------------
class TestPredictEndpoint:
    def test_predict_valid_input(self, client_with_model: TestClient) -> None:
        """Valid input should return 200 with a prediction response."""
        resp = client_with_model.post("/predict", json=VALID_CUSTOMER)
        assert resp.status_code == 200

    def test_predict_response_schema(self, client_with_model: TestClient) -> None:
        """Prediction response must contain all required fields."""
        resp = client_with_model.post("/predict", json=VALID_CUSTOMER)
        data = resp.json()
        for key in [
            "churn_probability", "churn_prediction", "risk_level",
            "model_version", "model_name", "prediction_id", "timestamp",
        ]:
            assert key in data, f"Missing field: {key}"

    def test_predict_high_churn_probability(self, client_with_model: TestClient) -> None:
        """Mock model returning 0.75 should produce High risk level."""
        resp = client_with_model.post("/predict", json=VALID_CUSTOMER)
        data = resp.json()
        assert data["churn_probability"] == pytest.approx(0.75, abs=0.01)
        assert data["churn_prediction"] is True
        assert data["risk_level"] == "High"

    def test_predict_low_risk(self, client_with_model: TestClient) -> None:
        """Mock model with low probability should produce Low risk level."""
        from src.serving.app import model_state
        model_state["model"] = _make_mock_model(churn_prob=0.1)

        resp = client_with_model.post("/predict", json=VALID_CUSTOMER)
        data = resp.json()
        assert data["risk_level"] == "Low"
        assert data["churn_prediction"] is False

    def test_predict_medium_risk(self, client_with_model: TestClient) -> None:
        """Mock model with 0.45 probability should produce Medium risk level."""
        from src.serving.app import model_state
        model_state["model"] = _make_mock_model(churn_prob=0.45)

        resp = client_with_model.post("/predict", json=VALID_CUSTOMER)
        data = resp.json()
        assert data["risk_level"] == "Medium"

    def test_predict_503_when_no_model(self, client_no_model: TestClient) -> None:
        """Predict should return 503 when no model is loaded."""
        resp = client_no_model.post("/predict", json=VALID_CUSTOMER)
        assert resp.status_code == 503

    def test_predict_invalid_gender_422(self, client_with_model: TestClient) -> None:
        """Invalid gender value should return 422 validation error."""
        bad_input = {**VALID_CUSTOMER, "gender": "Unknown"}
        resp = client_with_model.post("/predict", json=bad_input)
        assert resp.status_code == 422

    def test_predict_invalid_contract_422(self, client_with_model: TestClient) -> None:
        """Invalid contract type should return 422 validation error."""
        bad_input = {**VALID_CUSTOMER, "Contract": "Quarterly"}
        resp = client_with_model.post("/predict", json=bad_input)
        assert resp.status_code == 422

    def test_predict_negative_tenure_422(self, client_with_model: TestClient) -> None:
        """Negative tenure should fail validation with 422."""
        bad_input = {**VALID_CUSTOMER, "tenure": -1}
        resp = client_with_model.post("/predict", json=bad_input)
        assert resp.status_code == 422

    def test_predict_missing_field_422(self, client_with_model: TestClient) -> None:
        """Missing required field should return 422 validation error."""
        incomplete = {k: v for k, v in VALID_CUSTOMER.items() if k != "MonthlyCharges"}
        resp = client_with_model.post("/predict", json=incomplete)
        assert resp.status_code == 422

    def test_predict_returns_unique_prediction_id(self, client_with_model: TestClient) -> None:
        """Each prediction should have a unique prediction_id."""
        r1 = client_with_model.post("/predict", json=VALID_CUSTOMER).json()
        r2 = client_with_model.post("/predict", json=VALID_CUSTOMER).json()
        assert r1["prediction_id"] != r2["prediction_id"]


# ---------------------------------------------------------------------------
# Test: GET /metrics
# ---------------------------------------------------------------------------
class TestMetricsEndpoint:
    def test_metrics_503_when_no_model(self, client_no_model: TestClient) -> None:
        """Metrics endpoint should return 503 if model not loaded."""
        resp = client_no_model.get("/metrics")
        assert resp.status_code == 503

    def test_metrics_200_when_model_loaded(self, client_with_model: TestClient) -> None:
        """Metrics endpoint should return 200 when model is loaded."""
        with patch("src.serving.app.get_model_metrics_from_mlflow") as mock_metrics:
            mock_metrics.return_value = {
                "run_id": "abc123",
                "metrics": {
                    "accuracy": 0.82,
                    "roc_auc": 0.88,
                    "f1_score": 0.74,
                },
                "model_type": "XGBoost",
            }
            resp = client_with_model.get("/metrics")
        assert resp.status_code == 200

    def test_metrics_response_has_required_keys(self, client_with_model: TestClient) -> None:
        """Metrics response must contain required schema fields."""
        with patch("src.serving.app.get_model_metrics_from_mlflow") as mock_metrics:
            mock_metrics.return_value = {
                "run_id": "abc123",
                "metrics": {"roc_auc": 0.88},
                "model_type": "XGBoost",
            }
            resp = client_with_model.get("/metrics")
        data = resp.json()
        for key in ["model_name", "model_version", "stage", "metrics", "run_id", "timestamp"]:
            assert key in data, f"Missing key: {key}"


# ---------------------------------------------------------------------------
# Test: GET /drift-report
# ---------------------------------------------------------------------------
class TestDriftReportEndpoint:
    def test_drift_report_200_no_reports(self, client_with_model: TestClient) -> None:
        """Drift report should return 200 even when no reports directory exists."""
        with patch("src.serving.app.DRIFT_REPORT_PATH", "/nonexistent/path/reports"):
            resp = client_with_model.get("/drift-report")
        assert resp.status_code == 200

    def test_drift_report_with_mock_data(self, client_with_model: TestClient) -> None:
        """Drift report should return data from the latest summary JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            summary = {
                "drift_detected": True,
                "drift_score": 0.45,
                "method": "simple_statistical",
                "timestamp": "20250101_120000",
                "reference_rows": 1000,
                "current_rows": 500,
                "drift_threshold": 0.3,
            }
            summary_path = Path(tmpdir) / "drift_summary_20250101_120000.json"
            with open(summary_path, "w") as f:
                json.dump(summary, f)

            with patch("src.serving.app.DRIFT_REPORT_PATH", tmpdir):
                resp = client_with_model.get("/drift-report")

        assert resp.status_code == 200
        data = resp.json()
        assert data["drift_detected"] is True
        assert data["drift_score"] == pytest.approx(0.45)

    def test_drift_report_has_required_keys(self, client_with_model: TestClient) -> None:
        """Drift report response must contain all required schema keys."""
        resp = client_with_model.get("/drift-report")
        data = resp.json()
        for key in ["drift_detected", "drift_score", "method", "timestamp"]:
            assert key in data, f"Missing key: {key}"


# ---------------------------------------------------------------------------
# Test: GET /docs
# ---------------------------------------------------------------------------
class TestOpenAPISchema:
    def test_docs_endpoint_accessible(self, client_with_model: TestClient) -> None:
        """Swagger UI /docs should be accessible (returns 200)."""
        resp = client_with_model.get("/docs")
        assert resp.status_code == 200

    def test_openapi_json_accessible(self, client_with_model: TestClient) -> None:
        """OpenAPI JSON schema at /openapi.json should be accessible."""
        resp = client_with_model.get("/openapi.json")
        assert resp.status_code == 200
        schema = resp.json()
        assert "paths" in schema
        assert "/predict" in schema["paths"]
        assert "/health" in schema["paths"]


# ---------------------------------------------------------------------------
# Test: _determine_risk_level
# ---------------------------------------------------------------------------
class TestRiskLevel:
    def test_low_risk(self) -> None:
        from src.serving.app import _determine_risk_level
        assert _determine_risk_level(0.10) == "Low"
        assert _determine_risk_level(0.29) == "Low"

    def test_medium_risk(self) -> None:
        from src.serving.app import _determine_risk_level
        assert _determine_risk_level(0.30) == "Medium"
        assert _determine_risk_level(0.59) == "Medium"

    def test_high_risk(self) -> None:
        from src.serving.app import _determine_risk_level
        assert _determine_risk_level(0.60) == "High"
        assert _determine_risk_level(0.99) == "High"


# ---------------------------------------------------------------------------
# Test: _apply_feature_engineering
# ---------------------------------------------------------------------------
class TestFeatureEngineering:
    def test_all_engineered_features_added(self) -> None:
        """All 5 engineered features should be added to inference input."""
        from src.serving.app import _apply_feature_engineering
        df = pd.DataFrame([VALID_CUSTOMER])
        df_eng = _apply_feature_engineering(df)
        for feat in [
            "tenure_months_squared", "avg_monthly_spend", "has_any_addon",
            "services_count", "is_long_term_contract",
        ]:
            assert feat in df_eng.columns, f"Missing: {feat}"

    def test_tenure_squared_correct(self) -> None:
        from src.serving.app import _apply_feature_engineering
        df = pd.DataFrame([VALID_CUSTOMER])
        df_eng = _apply_feature_engineering(df)
        expected = VALID_CUSTOMER["tenure"] ** 2
        assert df_eng["tenure_months_squared"].iloc[0] == expected
