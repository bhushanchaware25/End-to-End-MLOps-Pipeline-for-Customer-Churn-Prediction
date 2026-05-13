"""
Tests for src/monitoring/monitor.py
=====================================
Tests cover:
  - _load_dataset() missing file handling
  - _compute_simple_drift() numerical and categorical drift scores
  - generate_drift_report() with temp CSV files
  - Drift detection flag correctness
  - Report JSON saved to correct location
"""

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.monitoring.monitor import (
    DRIFT_THRESHOLD,
    _compute_simple_drift,
    _load_dataset,
    generate_drift_report,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def _make_telco_df(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """Create a minimal Telco-like DataFrame for testing."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "customerID": [f"C{i:04d}" for i in range(n)],
            "gender": rng.choice(["Male", "Female"], n),
            "SeniorCitizen": rng.choice([0, 1], n),
            "Partner": rng.choice(["Yes", "No"], n),
            "Dependents": rng.choice(["Yes", "No"], n),
            "tenure": rng.integers(0, 72, n),
            "PhoneService": rng.choice(["Yes", "No"], n),
            "MultipleLines": rng.choice(["Yes", "No", "No phone service"], n),
            "InternetService": rng.choice(["DSL", "Fiber optic", "No"], n),
            "OnlineSecurity": rng.choice(["Yes", "No", "No internet service"], n),
            "OnlineBackup": rng.choice(["Yes", "No", "No internet service"], n),
            "DeviceProtection": rng.choice(["Yes", "No", "No internet service"], n),
            "TechSupport": rng.choice(["Yes", "No", "No internet service"], n),
            "StreamingTV": rng.choice(["Yes", "No", "No internet service"], n),
            "StreamingMovies": rng.choice(["Yes", "No", "No internet service"], n),
            "Contract": rng.choice(["Month-to-month", "One year", "Two year"], n),
            "PaperlessBilling": rng.choice(["Yes", "No"], n),
            "PaymentMethod": rng.choice(
                ["Electronic check", "Mailed check",
                 "Bank transfer (automatic)", "Credit card (automatic)"], n
            ),
            "MonthlyCharges": np.round(rng.uniform(18, 118, n), 2),
            "TotalCharges": np.round(rng.uniform(18, 8000, n), 2),
            "Churn": rng.choice(["Yes", "No"], n, p=[0.27, 0.73]),
        }
    )


@pytest.fixture
def reference_df() -> pd.DataFrame:
    """Reference dataset (stable distribution)."""
    return _make_telco_df(n=500, seed=42)


@pytest.fixture
def similar_current_df() -> pd.DataFrame:
    """Current dataset with similar distribution — minimal drift expected."""
    return _make_telco_df(n=300, seed=99)


@pytest.fixture
def drifted_current_df(reference_df: pd.DataFrame) -> pd.DataFrame:
    """Current dataset with intentional heavy drift in numerical features."""
    df = reference_df.copy()
    # Massively shift MonthlyCharges and tenure to force drift
    df["MonthlyCharges"] = df["MonthlyCharges"] + 80.0
    df["tenure"] = df["tenure"] + 60
    df["Contract"] = "Two year"  # categorical drift
    return df


# ---------------------------------------------------------------------------
# Test: _load_dataset
# ---------------------------------------------------------------------------
class TestLoadDataset:
    def test_loads_valid_csv(self, reference_df: pd.DataFrame) -> None:
        """Should load a valid CSV and return a DataFrame."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "ref.csv")
            reference_df.to_csv(path, index=False)
            df = _load_dataset(path, "reference")
            assert isinstance(df, pd.DataFrame)
            assert len(df) == len(reference_df)

    def test_raises_on_missing_file(self) -> None:
        """Should raise FileNotFoundError for non-existent file."""
        with pytest.raises(FileNotFoundError):
            _load_dataset("/nonexistent/path/data.csv", "reference")


# ---------------------------------------------------------------------------
# Test: _compute_simple_drift
# ---------------------------------------------------------------------------
class TestComputeSimpleDrift:
    def test_returns_required_keys(
        self, reference_df: pd.DataFrame, similar_current_df: pd.DataFrame
    ) -> None:
        """Result must contain all required keys."""
        result = _compute_simple_drift(
            reference=reference_df,
            current=similar_current_df,
            numerical_cols=["tenure", "MonthlyCharges"],
            categorical_cols=["Contract"],
        )
        for key in ["drift_detected", "drift_score", "feature_drift",
                    "drifted_features", "method"]:
            assert key in result, f"Missing key: {key}"

    def test_no_drift_similar_distributions(
        self, reference_df: pd.DataFrame, similar_current_df: pd.DataFrame
    ) -> None:
        """Similar distributions should produce low drift score / no detection."""
        result = _compute_simple_drift(
            reference=reference_df,
            current=similar_current_df,
            numerical_cols=["tenure", "MonthlyCharges"],
            categorical_cols=["gender"],
        )
        # Similar data should not have extreme drift
        assert result["drift_score"] < 5.0  # Allow for random variation

    def test_detects_heavy_drift(
        self, reference_df: pd.DataFrame, drifted_current_df: pd.DataFrame
    ) -> None:
        """Heavily shifted distributions should be flagged as drifted."""
        result = _compute_simple_drift(
            reference=reference_df,
            current=drifted_current_df,
            numerical_cols=["MonthlyCharges", "tenure"],
            categorical_cols=["Contract"],
        )
        assert result["drift_detected"] is True
        assert result["n_drifted_features"] > 0

    def test_feature_drift_scores_are_floats(
        self, reference_df: pd.DataFrame, similar_current_df: pd.DataFrame
    ) -> None:
        """Drift scores for each feature should be numeric."""
        result = _compute_simple_drift(
            reference=reference_df,
            current=similar_current_df,
            numerical_cols=["tenure"],
            categorical_cols=["gender"],
        )
        for feat, info in result["feature_drift"].items():
            assert isinstance(info["drift_score"], float)

    def test_missing_column_skipped_gracefully(
        self, reference_df: pd.DataFrame, similar_current_df: pd.DataFrame
    ) -> None:
        """Columns missing from either dataset should be silently skipped."""
        result = _compute_simple_drift(
            reference=reference_df,
            current=similar_current_df,
            numerical_cols=["NonExistentColumn"],
            categorical_cols=[],
        )
        assert "NonExistentColumn" not in result["feature_drift"]


# ---------------------------------------------------------------------------
# Test: generate_drift_report
# ---------------------------------------------------------------------------
class TestGenerateDriftReport:
    def test_saves_summary_json(
        self, reference_df: pd.DataFrame, similar_current_df: pd.DataFrame
    ) -> None:
        """generate_drift_report() should save a drift_summary_*.json file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ref_path = os.path.join(tmpdir, "reference.csv")
            cur_path = os.path.join(tmpdir, "current.csv")
            reports_dir = os.path.join(tmpdir, "reports")
            reference_df.to_csv(ref_path, index=False)
            similar_current_df.to_csv(cur_path, index=False)

            result = generate_drift_report(
                reference_path=ref_path,
                current_path=cur_path,
                reports_dir=reports_dir,
            )

            summary_path = Path(result["summary_path"])
            assert summary_path.exists(), "Summary JSON not saved"
            with open(summary_path) as f:
                saved = json.load(f)
            assert "drift_detected" in saved

    def test_returns_drift_detected_flag(
        self, reference_df: pd.DataFrame, similar_current_df: pd.DataFrame
    ) -> None:
        """Result must contain a boolean drift_detected key."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ref_path = os.path.join(tmpdir, "ref.csv")
            cur_path = os.path.join(tmpdir, "cur.csv")
            reference_df.to_csv(ref_path, index=False)
            similar_current_df.to_csv(cur_path, index=False)

            result = generate_drift_report(
                reference_path=ref_path,
                current_path=cur_path,
                reports_dir=os.path.join(tmpdir, "reports"),
            )

            assert isinstance(result["drift_detected"], bool)

    def test_detects_heavy_drift(
        self, reference_df: pd.DataFrame, drifted_current_df: pd.DataFrame
    ) -> None:
        """Heavy drift should be detected (drift_detected=True)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ref_path = os.path.join(tmpdir, "ref.csv")
            cur_path = os.path.join(tmpdir, "cur.csv")
            reference_df.to_csv(ref_path, index=False)
            drifted_current_df.to_csv(cur_path, index=False)

            result = generate_drift_report(
                reference_path=ref_path,
                current_path=cur_path,
                reports_dir=os.path.join(tmpdir, "reports"),
            )

            assert result["drift_detected"] is True

    def test_raises_on_missing_reference(self) -> None:
        """Missing reference file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            generate_drift_report(
                reference_path="/nonexistent/ref.csv",
                current_path="/nonexistent/cur.csv",
            )

    def test_result_contains_row_counts(
        self, reference_df: pd.DataFrame, similar_current_df: pd.DataFrame
    ) -> None:
        """Result should contain reference_rows and current_rows."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ref_path = os.path.join(tmpdir, "ref.csv")
            cur_path = os.path.join(tmpdir, "cur.csv")
            reference_df.to_csv(ref_path, index=False)
            similar_current_df.to_csv(cur_path, index=False)

            result = generate_drift_report(
                reference_path=ref_path,
                current_path=cur_path,
                reports_dir=os.path.join(tmpdir, "reports"),
            )

            assert result["reference_rows"] == len(reference_df)
            assert result["current_rows"] == len(similar_current_df)
