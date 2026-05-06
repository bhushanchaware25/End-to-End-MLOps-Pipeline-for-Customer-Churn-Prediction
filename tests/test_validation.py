"""
Tests for src/validation/validate.py
=====================================
Tests cover:
  - Individual expectation methods (pass + fail for each type)
  - Full suite execution on valid and invalid data
  - validate_data() integration with temp files
  - raise_on_failure flag behavior
  - Report JSON structure and content
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Dict, Any

import pandas as pd
import pytest

from src.validation.validate import (
    CRITICAL_COLUMNS,
    EXPECTED_CHURN,
    EXPECTED_CONTRACT,
    TelcoDataValidator,
    validate_data,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def valid_df() -> pd.DataFrame:
    """Return a minimal fully valid processed Telco DataFrame."""
    return pd.DataFrame(
        {
            "customerID": ["C001", "C002", "C003", "C004", "C005"],
            "gender": ["Male", "Female", "Male", "Female", "Male"],
            "SeniorCitizen": [0, 1, 0, 0, 1],
            "Partner": ["Yes", "No", "Yes", "No", "Yes"],
            "Dependents": ["No", "No", "Yes", "Yes", "No"],
            "tenure": [12, 1, 60, 24, 5],
            "PhoneService": ["Yes", "Yes", "No", "Yes", "Yes"],
            "MultipleLines": ["No", "Yes", "No phone service", "No", "Yes"],
            "InternetService": ["DSL", "Fiber optic", "No", "DSL", "Fiber optic"],
            "OnlineSecurity": ["Yes", "No", "No internet service", "Yes", "No"],
            "OnlineBackup": ["No", "Yes", "No internet service", "Yes", "No"],
            "DeviceProtection": ["No", "Yes", "No internet service", "No", "Yes"],
            "TechSupport": ["No", "No", "No internet service", "Yes", "No"],
            "StreamingTV": ["No", "Yes", "No internet service", "No", "Yes"],
            "StreamingMovies": ["No", "Yes", "No internet service", "No", "Yes"],
            "Contract": [
                "Month-to-month",
                "One year",
                "Two year",
                "Month-to-month",
                "One year",
            ],
            "PaperlessBilling": ["Yes", "No", "Yes", "No", "Yes"],
            "PaymentMethod": [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
                "Electronic check",
            ],
            "MonthlyCharges": [29.85, 56.95, 53.85, 42.30, 70.00],
            "TotalCharges": [357.20, 56.95, 3227.10, 1015.20, 350.00],
            "Churn": ["No", "No", "Yes", "No", "Yes"],
        }
    )


@pytest.fixture
def invalid_df(valid_df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with intentional validation failures."""
    df = valid_df.copy()
    df.loc[0, "Churn"] = "Maybe"          # invalid categorical
    df.loc[1, "tenure"] = -5              # negative tenure
    df.loc[2, "TotalCharges"] = None      # null in critical column
    df.loc[3, "Contract"] = "Unknown"     # invalid contract type
    return df


@pytest.fixture
def validator(valid_df: pd.DataFrame) -> TelcoDataValidator:
    """Return a validator instance loaded with valid data."""
    return TelcoDataValidator(valid_df)


@pytest.fixture
def invalid_validator(invalid_df: pd.DataFrame) -> TelcoDataValidator:
    """Return a validator instance loaded with invalid data."""
    return TelcoDataValidator(invalid_df)


# ---------------------------------------------------------------------------
# Test: expect_table_columns_to_match
# ---------------------------------------------------------------------------
class TestExpectTableColumnsToMatch:
    def test_all_columns_present_passes(self, validator: TelcoDataValidator) -> None:
        """Expectation should pass when all required columns are present."""
        from src.ingestion.ingest import EXPECTED_COLUMNS
        validator.expect_table_columns_to_match(EXPECTED_COLUMNS)
        results = [
            e for e in validator.report["expectations"]
            if e["expectation_type"] == "expect_table_columns_to_match_ordered_list"
        ]
        assert results[-1]["success"] is True

    def test_missing_column_fails(self, valid_df: pd.DataFrame) -> None:
        """Expectation should fail when a required column is missing."""
        df_missing = valid_df.drop(columns=["Churn"])
        v = TelcoDataValidator(df_missing)
        from src.ingestion.ingest import EXPECTED_COLUMNS
        v.expect_table_columns_to_match(EXPECTED_COLUMNS)
        results = v.report["expectations"]
        assert results[-1]["success"] is False


# ---------------------------------------------------------------------------
# Test: expect_column_values_to_not_be_null
# ---------------------------------------------------------------------------
class TestExpectNoNull:
    def test_no_nulls_passes(self, validator: TelcoDataValidator) -> None:
        """Column with no nulls should pass the not-null expectation."""
        validator.expect_column_values_to_not_be_null("tenure")
        result = validator.report["expectations"][-1]
        assert result["success"] is True

    def test_null_present_fails(self, invalid_validator: TelcoDataValidator) -> None:
        """Column with null values should fail the not-null expectation."""
        invalid_validator.expect_column_values_to_not_be_null("TotalCharges")
        result = invalid_validator.report["expectations"][-1]
        assert result["success"] is False

    def test_nonexistent_column_fails(self, validator: TelcoDataValidator) -> None:
        """Checking null on a missing column should record failure."""
        validator.expect_column_values_to_not_be_null("NonExistentColumn")
        result = validator.report["expectations"][-1]
        assert result["success"] is False


# ---------------------------------------------------------------------------
# Test: expect_column_values_to_be_of_type
# ---------------------------------------------------------------------------
class TestExpectColumnType:
    def test_float_column_passes(self, validator: TelcoDataValidator) -> None:
        """MonthlyCharges (float) should pass float type expectation."""
        validator.expect_column_values_to_be_of_type("MonthlyCharges", "float")
        result = validator.report["expectations"][-1]
        assert result["success"] is True

    def test_int_column_passes(self, validator: TelcoDataValidator) -> None:
        """tenure (int) should pass int type expectation."""
        validator.expect_column_values_to_be_of_type("tenure", "int")
        result = validator.report["expectations"][-1]
        assert result["success"] is True

    def test_wrong_type_fails(self, validator: TelcoDataValidator) -> None:
        """String column checked for int type should fail."""
        validator.expect_column_values_to_be_of_type("gender", "int")
        result = validator.report["expectations"][-1]
        assert result["success"] is False


# ---------------------------------------------------------------------------
# Test: expect_column_values_to_be_between
# ---------------------------------------------------------------------------
class TestExpectValuesBetween:
    def test_tenure_non_negative_passes(self, validator: TelcoDataValidator) -> None:
        """Valid tenure values (≥0) should pass the range check."""
        validator.expect_column_values_to_be_between("tenure", min_value=0, max_value=None)
        result = validator.report["expectations"][-1]
        assert result["success"] is True

    def test_negative_tenure_fails(self, invalid_validator: TelcoDataValidator) -> None:
        """Negative tenure value should fail the range check."""
        invalid_validator.expect_column_values_to_be_between(
            "tenure", min_value=0, max_value=None
        )
        result = invalid_validator.report["expectations"][-1]
        assert result["success"] is False

    def test_senior_citizen_0_or_1_passes(self, validator: TelcoDataValidator) -> None:
        """SeniorCitizen values of 0 and 1 should pass [0, 1] range check."""
        validator.expect_column_values_to_be_between("SeniorCitizen", 0, 1)
        result = validator.report["expectations"][-1]
        assert result["success"] is True

    def test_monthly_charges_range_passes(self, validator: TelcoDataValidator) -> None:
        """MonthlyCharges within realistic range should pass."""
        validator.expect_column_values_to_be_between("MonthlyCharges", 0.0, 200.0)
        result = validator.report["expectations"][-1]
        assert result["success"] is True


# ---------------------------------------------------------------------------
# Test: expect_column_values_to_be_in_set
# ---------------------------------------------------------------------------
class TestExpectValuesInSet:
    def test_churn_yes_no_passes(self, validator: TelcoDataValidator) -> None:
        """Valid Churn column (Yes/No) should pass in-set check."""
        validator.expect_column_values_to_be_in_set("Churn", EXPECTED_CHURN)
        result = validator.report["expectations"][-1]
        assert result["success"] is True

    def test_invalid_churn_fails(self, invalid_validator: TelcoDataValidator) -> None:
        """Churn column with 'Maybe' should fail in-set check."""
        invalid_validator.expect_column_values_to_be_in_set("Churn", EXPECTED_CHURN)
        result = invalid_validator.report["expectations"][-1]
        assert result["success"] is False

    def test_invalid_contract_fails(self, invalid_validator: TelcoDataValidator) -> None:
        """Contract column with 'Unknown' should fail in-set check."""
        invalid_validator.expect_column_values_to_be_in_set("Contract", EXPECTED_CONTRACT)
        result = invalid_validator.report["expectations"][-1]
        assert result["success"] is False

    def test_valid_contract_passes(self, validator: TelcoDataValidator) -> None:
        """Valid Contract values should pass in-set check."""
        validator.expect_column_values_to_be_in_set("Contract", EXPECTED_CONTRACT)
        result = validator.report["expectations"][-1]
        assert result["success"] is True


# ---------------------------------------------------------------------------
# Test: expect_table_row_count_to_be_between
# ---------------------------------------------------------------------------
class TestExpectRowCount:
    def test_sufficient_rows_passes(self, validator: TelcoDataValidator) -> None:
        """DataFrame with ≥ min_value rows should pass."""
        validator.expect_table_row_count_to_be_between(min_value=1)
        result = validator.report["expectations"][-1]
        assert result["success"] is True

    def test_too_few_rows_fails(self, validator: TelcoDataValidator) -> None:
        """DataFrame with fewer rows than min_value should fail."""
        validator.expect_table_row_count_to_be_between(min_value=9999)
        result = validator.report["expectations"][-1]
        assert result["success"] is False


# ---------------------------------------------------------------------------
# Test: Report Structure
# ---------------------------------------------------------------------------
class TestReportStructure:
    def test_report_has_required_keys(self, valid_df: pd.DataFrame) -> None:
        """Validation report dict must contain all required top-level keys."""
        v = TelcoDataValidator(valid_df)
        v.expect_column_to_exist("Churn")
        report = v.report
        assert "timestamp" in report
        assert "dataset_shape" in report
        assert "expectations" in report
        assert "summary" in report

    def test_summary_counts_correct(self, valid_df: pd.DataFrame) -> None:
        """Summary passed + failed counts should equal total."""
        v = TelcoDataValidator(valid_df)
        v.run_full_suite()
        s = v.report["summary"]
        assert s["passed"] + s["failed"] == s["total"]

    def test_success_rate_is_between_0_and_1(self, valid_df: pd.DataFrame) -> None:
        """Success rate must be a valid fraction in [0, 1]."""
        v = TelcoDataValidator(valid_df)
        v.run_full_suite()
        rate = v.report["summary"]["success_rate"]
        assert 0.0 <= rate <= 1.0


# ---------------------------------------------------------------------------
# Test: Full Suite on Valid Data
# ---------------------------------------------------------------------------
class TestFullSuiteValidData:
    def test_full_suite_passes_on_valid_data(self, valid_df: pd.DataFrame) -> None:
        """Full suite should pass with 100% success on valid data."""
        v = TelcoDataValidator(valid_df)
        report = v.run_full_suite()
        assert report["summary"]["failed"] == 0

    def test_full_suite_catches_invalid_data(self, invalid_df: pd.DataFrame) -> None:
        """Full suite should detect failures in invalid data."""
        v = TelcoDataValidator(invalid_df)
        report = v.run_full_suite()
        assert report["summary"]["failed"] > 0


# ---------------------------------------------------------------------------
# Test: validate_data() Integration
# ---------------------------------------------------------------------------
class TestValidateDataIntegration:
    def test_validates_saved_csv(self, valid_df: pd.DataFrame) -> None:
        """validate_data() should load and validate a saved CSV successfully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "processed.csv")
            report_path = os.path.join(tmpdir, "report.json")
            valid_df.to_csv(csv_path, index=False)

            report = validate_data(
                data_path=csv_path,
                report_output_path=report_path,
                raise_on_failure=False,
            )
            assert report["summary"]["total"] > 0

    def test_saves_json_report(self, valid_df: pd.DataFrame) -> None:
        """validate_data() should write a valid JSON report file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "processed.csv")
            report_path = os.path.join(tmpdir, "report.json")
            valid_df.to_csv(csv_path, index=False)

            validate_data(
                data_path=csv_path,
                report_output_path=report_path,
                raise_on_failure=False,
            )
            assert Path(report_path).exists()
            with open(report_path) as f:
                saved = json.load(f)
            assert "summary" in saved

    def test_raises_on_missing_file(self) -> None:
        """validate_data() should raise FileNotFoundError for missing CSV."""
        with pytest.raises(FileNotFoundError):
            validate_data(data_path="/nonexistent/path/data.csv")

    def test_raises_on_invalid_data_when_flag_set(
        self, invalid_df: pd.DataFrame
    ) -> None:
        """validate_data() should raise RuntimeError on failure when raise_on_failure=True."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "invalid.csv")
            report_path = os.path.join(tmpdir, "report.json")
            invalid_df.to_csv(csv_path, index=False)

            with pytest.raises(RuntimeError, match="Data validation failed"):
                validate_data(
                    data_path=csv_path,
                    report_output_path=report_path,
                    raise_on_failure=True,
                )

    def test_no_raise_on_invalid_when_flag_false(
        self, invalid_df: pd.DataFrame
    ) -> None:
        """validate_data() should NOT raise when raise_on_failure=False."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "invalid.csv")
            report_path = os.path.join(tmpdir, "report.json")
            invalid_df.to_csv(csv_path, index=False)

            report = validate_data(
                data_path=csv_path,
                report_output_path=report_path,
                raise_on_failure=False,
            )
            assert report["summary"]["failed"] > 0
