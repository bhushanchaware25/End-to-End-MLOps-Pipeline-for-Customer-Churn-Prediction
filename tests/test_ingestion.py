"""
Tests for src/ingestion/ingest.py
==================================
Tests cover:
  - Synthetic data generation shape and schema
  - Schema validation (pass + fail cases)
  - Data cleaning: TotalCharges conversion, imputation, dedup
  - Full ingest_data() integration with temp directories
"""

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.ingestion.ingest import (
    EXPECTED_COLUMNS,
    clean_data,
    generate_synthetic_telco_data,
    ingest_data,
    validate_schema,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Return a minimal valid Telco-like DataFrame for testing."""
    return pd.DataFrame(
        {
            "customerID": ["C001", "C002", "C003"],
            "gender": ["Male", "Female", "Male"],
            "SeniorCitizen": [0, 1, 0],
            "Partner": ["Yes", "No", "Yes"],
            "Dependents": ["No", "No", "Yes"],
            "tenure": [12, 0, 60],
            "PhoneService": ["Yes", "Yes", "No"],
            "MultipleLines": ["No", "Yes", "No phone service"],
            "InternetService": ["DSL", "Fiber optic", "No"],
            "OnlineSecurity": ["Yes", "No", "No internet service"],
            "OnlineBackup": ["No", "Yes", "No internet service"],
            "DeviceProtection": ["No", "Yes", "No internet service"],
            "TechSupport": ["No", "No", "No internet service"],
            "StreamingTV": ["No", "Yes", "No internet service"],
            "StreamingMovies": ["No", "Yes", "No internet service"],
            "Contract": ["Month-to-month", "One year", "Two year"],
            "PaperlessBilling": ["Yes", "No", "Yes"],
            "PaymentMethod": [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
            ],
            "MonthlyCharges": [29.85, 56.95, 53.85],
            "TotalCharges": ["357.20", " ", "3227.10"],
            "Churn": ["No", "No", "Yes"],
        }
    )


@pytest.fixture
def dirty_df(sample_df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with intentional data quality issues."""
    df = sample_df.copy()
    # Add duplicate
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    # Add whitespace in strings
    df.loc[0, "gender"] = "  Male  "
    return df


# ---------------------------------------------------------------------------
# Test: Synthetic Data Generation
# ---------------------------------------------------------------------------
class TestSyntheticDataGeneration:
    def test_shape(self) -> None:
        """Generated DataFrame should have 7043 rows and 21 columns."""
        df = generate_synthetic_telco_data(n_samples=7043)
        assert df.shape == (7043, 21)

    def test_all_expected_columns_present(self) -> None:
        """All 21 expected columns should be present."""
        df = generate_synthetic_telco_data(n_samples=500)
        for col in EXPECTED_COLUMNS:
            assert col in df.columns, f"Missing column: {col}"

    def test_churn_values(self) -> None:
        """Churn column should only contain 'Yes' or 'No'."""
        df = generate_synthetic_telco_data(n_samples=500)
        unique_churn = set(df["Churn"].unique())
        assert unique_churn.issubset({"Yes", "No"})

    def test_tenure_non_negative(self) -> None:
        """tenure column should not have negative values."""
        df = generate_synthetic_telco_data(n_samples=500)
        assert (df["tenure"] >= 0).all()

    def test_reproducibility(self) -> None:
        """Same seed should produce identical DataFrames."""
        df1 = generate_synthetic_telco_data(n_samples=100, random_seed=99)
        df2 = generate_synthetic_telco_data(n_samples=100, random_seed=99)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_differ(self) -> None:
        """Different seeds should produce different DataFrames."""
        df1 = generate_synthetic_telco_data(n_samples=100, random_seed=1)
        df2 = generate_synthetic_telco_data(n_samples=100, random_seed=2)
        assert not df1.equals(df2)

    def test_churn_rate_realistic(self) -> None:
        """Churn rate should be between 20% and 40% (realistic for telecom)."""
        df = generate_synthetic_telco_data(n_samples=5000)
        churn_rate = (df["Churn"] == "Yes").mean()
        assert 0.20 <= churn_rate <= 0.55, f"Churn rate out of range: {churn_rate:.2f}"


# ---------------------------------------------------------------------------
# Test: Schema Validation
# ---------------------------------------------------------------------------
class TestSchemaValidation:
    def test_valid_schema_passes(self, sample_df: pd.DataFrame) -> None:
        """Valid DataFrame with all expected columns should pass validation."""
        assert validate_schema(sample_df) is True

    def test_missing_column_raises(self, sample_df: pd.DataFrame) -> None:
        """DataFrame missing required columns should raise ValueError."""
        df_bad = sample_df.drop(columns=["Churn", "TotalCharges"])
        with pytest.raises(ValueError, match="Schema validation failed"):
            validate_schema(df_bad)

    def test_extra_columns_allowed(self, sample_df: pd.DataFrame) -> None:
        """DataFrame with extra columns beyond expected should still pass."""
        df_extra = sample_df.copy()
        df_extra["ExtraColumn"] = 1
        assert validate_schema(df_extra) is True


# ---------------------------------------------------------------------------
# Test: Data Cleaning
# ---------------------------------------------------------------------------
class TestDataCleaning:
    def test_total_charges_converted_to_float(self, sample_df: pd.DataFrame) -> None:
        """TotalCharges with whitespace strings should be converted to float."""
        df_clean, _ = clean_data(sample_df)
        assert pd.api.types.is_float_dtype(df_clean["TotalCharges"])

    def test_missing_total_charges_imputed(self, sample_df: pd.DataFrame) -> None:
        """Missing TotalCharges (row with ' ') should be imputed, not null."""
        df_clean, report = clean_data(sample_df)
        assert df_clean["TotalCharges"].isna().sum() == 0
        assert report["missing_total_charges_imputed"] == 1

    def test_imputation_logic(self, sample_df: pd.DataFrame) -> None:
        """Imputed TotalCharges should be tenure × MonthlyCharges for missing rows."""
        df_clean, _ = clean_data(sample_df)
        # Row index 1 had TotalCharges = ' ', tenure=0, MonthlyCharges=56.95
        # imputed = 0 * 56.95 = 0.0
        assert df_clean.loc[1, "TotalCharges"] == pytest.approx(
            df_clean.loc[1, "tenure"] * df_clean.loc[1, "MonthlyCharges"]
        )

    def test_duplicate_removal(self, dirty_df: pd.DataFrame) -> None:
        """Duplicate customerID rows should be removed."""
        df_clean, report = clean_data(dirty_df)
        assert df_clean["customerID"].nunique() == len(df_clean)
        assert report["duplicates_removed"] == 1

    def test_whitespace_stripped(self, dirty_df: pd.DataFrame) -> None:
        """String columns should have leading/trailing whitespace removed."""
        df_clean, _ = clean_data(dirty_df)
        assert df_clean.loc[0, "gender"] == "Male"

    def test_churn_standardized(self, sample_df: pd.DataFrame) -> None:
        """Churn column should only have 'Yes' or 'No' after cleaning."""
        df_clean, _ = clean_data(sample_df)
        assert set(df_clean["Churn"].unique()).issubset({"Yes", "No"})

    def test_correct_dtypes_after_clean(self, sample_df: pd.DataFrame) -> None:
        """Numeric columns should have correct dtypes after cleaning."""
        df_clean, _ = clean_data(sample_df)
        assert pd.api.types.is_integer_dtype(df_clean["SeniorCitizen"])
        assert pd.api.types.is_integer_dtype(df_clean["tenure"])
        assert pd.api.types.is_float_dtype(df_clean["MonthlyCharges"])
        assert pd.api.types.is_float_dtype(df_clean["TotalCharges"])

    def test_no_nulls_in_key_columns(self, sample_df: pd.DataFrame) -> None:
        """After cleaning, key columns should have no null values."""
        df_clean, _ = clean_data(sample_df)
        key_cols = ["customerID", "Churn", "tenure", "MonthlyCharges", "TotalCharges"]
        for col in key_cols:
            assert df_clean[col].isna().sum() == 0, f"Null found in column: {col}"


# ---------------------------------------------------------------------------
# Test: Full Ingestion Integration
# ---------------------------------------------------------------------------
class TestIngestData:
    def test_ingest_generates_synthetic_when_no_file(self) -> None:
        """ingest_data() should generate synthetic data if CSV does not exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_path = os.path.join(tmpdir, "raw", "telco.csv")
            out_path = os.path.join(tmpdir, "processed", "clean.csv")

            df = ingest_data(
                raw_data_path=raw_path,
                output_path=out_path,
                generate_synthetic=True,
            )
            assert len(df) > 0
            assert "Churn" in df.columns

    def test_ingest_saves_processed_csv(self) -> None:
        """ingest_data() should save the processed CSV to the output path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "processed", "clean.csv")
            ingest_data(output_path=out_path, generate_synthetic=True)
            assert Path(out_path).exists()

    def test_ingest_loads_existing_csv(self, sample_df: pd.DataFrame) -> None:
        """ingest_data() should load and process an existing CSV correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_path = os.path.join(tmpdir, "raw", "telco.csv")
            out_path = os.path.join(tmpdir, "processed", "clean.csv")
            Path(raw_path).parent.mkdir(parents=True, exist_ok=True)
            sample_df.to_csv(raw_path, index=False)

            df = ingest_data(raw_data_path=raw_path, output_path=out_path)
            assert len(df) == len(sample_df)  # no dupes to remove here

    def test_ingest_returns_dataframe(self) -> None:
        """ingest_data() should return a pandas DataFrame."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "processed", "clean.csv")
            result = ingest_data(output_path=out_path, generate_synthetic=True)
            assert isinstance(result, pd.DataFrame)

    def test_ingest_output_has_no_nulls_in_key_cols(self) -> None:
        """Processed output should have no nulls in key columns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "processed", "clean.csv")
            df = ingest_data(output_path=out_path, generate_synthetic=True)
            key_cols = ["Churn", "tenure", "MonthlyCharges", "TotalCharges"]
            for col in key_cols:
                assert df[col].isna().sum() == 0
