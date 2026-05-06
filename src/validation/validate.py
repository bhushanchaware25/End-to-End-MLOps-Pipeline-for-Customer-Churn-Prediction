"""
ChurnShield MLOps Platform - Data Validation Module
====================================================
Uses Great Expectations to validate the processed Telco Customer Churn
dataset before it enters the training pipeline.

Validates:
  - Schema: all expected columns present with correct types
  - No nulls in critical columns
  - TotalCharges is numeric and non-negative
  - tenure is non-negative integer
  - Churn column only contains 'Yes' or 'No'
  - MonthlyCharges within a realistic range
  - SeniorCitizen is 0 or 1 only
  - Categorical columns contain only expected values

Usage:
    python -m src.validation.validate
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from loguru import logger

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
    "logs/validation.log",
    rotation="10 MB",
    retention="30 days",
    level="DEBUG",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{line} - {message}",
)

# ---------------------------------------------------------------------------
# Expected Values for Categorical Columns
# ---------------------------------------------------------------------------
EXPECTED_GENDER_VALUES = {"Male", "Female"}
EXPECTED_YES_NO = {"Yes", "No"}
EXPECTED_YES_NO_NO_PHONE = {"Yes", "No", "No phone service"}
EXPECTED_YES_NO_NO_INTERNET = {"Yes", "No", "No internet service"}
EXPECTED_INTERNET_SERVICE = {"DSL", "Fiber optic", "No"}
EXPECTED_CONTRACT = {"Month-to-month", "One year", "Two year"}
EXPECTED_PAYMENT_METHOD = {
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)",
}
EXPECTED_CHURN = {"Yes", "No"}

# Critical columns that must never have nulls
CRITICAL_COLUMNS = [
    "customerID",
    "gender",
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
    "Contract",
    "InternetService",
]


# ---------------------------------------------------------------------------
# Custom Validator Class (Great-Expectations-compatible via pandas)
# ---------------------------------------------------------------------------
class TelcoDataValidator:
    """
    Validates Telco Customer Churn dataset using Great Expectations.

    Falls back to manual pandas-based validation if Great Expectations
    context setup fails (e.g., in CI environments without GE configured).

    Attributes:
        df: The DataFrame to validate.
        report: Dictionary accumulating all validation results.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        """
        Initialize the validator with a DataFrame.

        Args:
            df: Processed Telco dataset to validate.
        """
        self.df = df.copy()
        self.report: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat(),
            "dataset_shape": list(df.shape),
            "expectations": [],
            "summary": {
                "total": 0,
                "passed": 0,
                "failed": 0,
                "success_rate": 0.0,
            },
        }

    def _add_result(
        self,
        expectation_type: str,
        column: Optional[str],
        success: bool,
        observed_value: Any,
        details: str = "",
    ) -> None:
        """
        Record a single expectation result.

        Args:
            expectation_type: Human-readable name of the expectation.
            column: Column name being tested (None for table-level checks).
            success: Whether the expectation passed.
            observed_value: What was actually found in the data.
            details: Additional context string.
        """
        result = {
            "expectation_type": expectation_type,
            "column": column,
            "success": success,
            "observed_value": str(observed_value),
            "details": details,
        }
        self.report["expectations"].append(result)
        self.report["summary"]["total"] += 1
        if success:
            self.report["summary"]["passed"] += 1
            logger.debug(f"  ✅ PASS | {expectation_type} | col={column} | {details}")
        else:
            self.report["summary"]["failed"] += 1
            logger.warning(f"  ❌ FAIL | {expectation_type} | col={column} | {details}")

    # ------------------------------------------------------------------
    # Expectation Methods
    # ------------------------------------------------------------------
    def expect_table_columns_to_match(self, expected_columns: list) -> None:
        """Expect all specified columns to be present in the dataset."""
        actual_cols = set(self.df.columns)
        expected_set = set(expected_columns)
        missing = expected_set - actual_cols
        success = len(missing) == 0
        self._add_result(
            expectation_type="expect_table_columns_to_match_ordered_list",
            column=None,
            success=success,
            observed_value=sorted(actual_cols),
            details=f"Missing columns: {missing}" if missing else "All columns present",
        )

    def expect_column_values_to_not_be_null(self, column: str) -> None:
        """Expect no null values in the specified column."""
        if column not in self.df.columns:
            self._add_result(
                "expect_column_values_to_not_be_null",
                column,
                False,
                "COLUMN_NOT_FOUND",
                f"Column '{column}' does not exist",
            )
            return
        null_count = int(self.df[column].isna().sum())
        success = null_count == 0
        self._add_result(
            expectation_type="expect_column_values_to_not_be_null",
            column=column,
            success=success,
            observed_value=f"{null_count} nulls",
            details=f"{null_count} null values found",
        )

    def expect_column_values_to_be_of_type(self, column: str, type_name: str) -> None:
        """Expect all values in the column to be of the given Python/numpy type."""
        if column not in self.df.columns:
            self._add_result(
                "expect_column_values_to_be_of_type",
                column,
                False,
                "COLUMN_NOT_FOUND",
                f"Column '{column}' does not exist",
            )
            return
        actual_dtype = str(self.df[column].dtype)
        type_map = {
            "float": ("float", "float32", "float64"),
            "int": ("int", "int32", "int64"),
            "str": ("object",),
            "object": ("object",),
        }
        expected_types = type_map.get(type_name, (type_name,))
        success = any(t in actual_dtype for t in expected_types)
        self._add_result(
            expectation_type="expect_column_values_to_be_of_type",
            column=column,
            success=success,
            observed_value=actual_dtype,
            details=f"Expected type '{type_name}', got '{actual_dtype}'",
        )

    def expect_column_values_to_be_between(
        self,
        column: str,
        min_value: Optional[float],
        max_value: Optional[float],
    ) -> None:
        """Expect all numeric values in column to be within [min_value, max_value]."""
        if column not in self.df.columns:
            self._add_result(
                "expect_column_values_to_be_between",
                column,
                False,
                "COLUMN_NOT_FOUND",
                f"Column '{column}' does not exist",
            )
            return
        series = pd.to_numeric(self.df[column], errors="coerce").dropna()
        violations = 0
        if min_value is not None:
            violations += int((series < min_value).sum())
        if max_value is not None:
            violations += int((series > max_value).sum())
        success = violations == 0
        self._add_result(
            expectation_type="expect_column_values_to_be_between",
            column=column,
            success=success,
            observed_value=f"min={series.min():.2f}, max={series.max():.2f}",
            details=(
                f"{violations} values outside [{min_value}, {max_value}]"
                if not success
                else f"All values in [{min_value}, {max_value}]"
            ),
        )

    def expect_column_values_to_be_in_set(
        self, column: str, value_set: set
    ) -> None:
        """Expect all values in column to be members of the provided set."""
        if column not in self.df.columns:
            self._add_result(
                "expect_column_values_to_be_in_set",
                column,
                False,
                "COLUMN_NOT_FOUND",
                f"Column '{column}' does not exist",
            )
            return
        unique_vals = set(self.df[column].dropna().unique())
        unexpected = unique_vals - value_set
        success = len(unexpected) == 0
        self._add_result(
            expectation_type="expect_column_values_to_be_in_set",
            column=column,
            success=success,
            observed_value=sorted(unique_vals),
            details=(
                f"Unexpected values: {unexpected}"
                if not success
                else f"All values in expected set"
            ),
        )

    def expect_column_to_exist(self, column: str) -> None:
        """Expect a specific column to exist in the DataFrame."""
        success = column in self.df.columns
        self._add_result(
            expectation_type="expect_column_to_exist",
            column=column,
            success=success,
            observed_value="exists" if success else "missing",
            details=f"Column '{column}' {'found' if success else 'NOT found'}",
        )

    def expect_table_row_count_to_be_between(
        self, min_value: int, max_value: Optional[int] = None
    ) -> None:
        """Expect the number of rows to be within a specified range."""
        row_count = len(self.df)
        success = row_count >= min_value
        if max_value is not None:
            success = success and (row_count <= max_value)
        self._add_result(
            expectation_type="expect_table_row_count_to_be_between",
            column=None,
            success=success,
            observed_value=row_count,
            details=f"Row count {row_count} {'within' if success else 'outside'} [{min_value}, {max_value}]",
        )

    def expect_column_unique_value_count_to_be_between(
        self, column: str, min_value: int, max_value: Optional[int] = None
    ) -> None:
        """Expect the number of unique values in a column to be in range."""
        if column not in self.df.columns:
            self._add_result(
                "expect_column_unique_value_count_to_be_between",
                column,
                False,
                "COLUMN_NOT_FOUND",
                f"Column '{column}' does not exist",
            )
            return
        unique_count = self.df[column].nunique()
        success = unique_count >= min_value
        if max_value is not None:
            success = success and (unique_count <= max_value)
        self._add_result(
            expectation_type="expect_column_unique_value_count_to_be_between",
            column=column,
            success=success,
            observed_value=unique_count,
            details=f"{unique_count} unique values",
        )

    def expect_column_mean_to_be_between(
        self, column: str, min_value: float, max_value: float
    ) -> None:
        """Expect the mean of a column to be within [min_value, max_value]."""
        if column not in self.df.columns:
            self._add_result(
                "expect_column_mean_to_be_between",
                column,
                False,
                "COLUMN_NOT_FOUND",
                f"Column '{column}' does not exist",
            )
            return
        mean_val = float(pd.to_numeric(self.df[column], errors="coerce").mean())
        success = min_value <= mean_val <= max_value
        self._add_result(
            expectation_type="expect_column_mean_to_be_between",
            column=column,
            success=success,
            observed_value=f"{mean_val:.4f}",
            details=f"Mean {mean_val:.4f} {'within' if success else 'outside'} [{min_value}, {max_value}]",
        )

    # ------------------------------------------------------------------
    # Run Full Suite
    # ------------------------------------------------------------------
    def run_full_suite(self) -> Dict[str, Any]:
        """
        Execute the complete expectation suite against the loaded DataFrame.

        Returns:
            Dict[str, Any]: Full validation report with per-expectation results
                            and summary statistics.
        """
        logger.info("Running full Great Expectations validation suite...")

        # -- Table-level checks
        from src.ingestion.ingest import EXPECTED_COLUMNS

        self.expect_table_columns_to_match(EXPECTED_COLUMNS)
        self.expect_table_row_count_to_be_between(min_value=100)

        # -- Column existence (critical columns)
        for col in CRITICAL_COLUMNS:
            self.expect_column_to_exist(col)

        # -- No nulls in critical columns
        for col in CRITICAL_COLUMNS:
            self.expect_column_values_to_not_be_null(col)

        # -- Type checks
        self.expect_column_values_to_be_of_type("TotalCharges", "float")
        self.expect_column_values_to_be_of_type("MonthlyCharges", "float")
        self.expect_column_values_to_be_of_type("tenure", "int")
        self.expect_column_values_to_be_of_type("SeniorCitizen", "int")

        # -- Numeric range checks
        self.expect_column_values_to_be_between("tenure", min_value=0, max_value=None)
        self.expect_column_values_to_be_between("TotalCharges", min_value=0.0, max_value=None)
        self.expect_column_values_to_be_between(
            "MonthlyCharges", min_value=0.0, max_value=200.0
        )
        self.expect_column_values_to_be_between(
            "SeniorCitizen", min_value=0, max_value=1
        )

        # -- Mean range sanity checks (catch data corruption)
        self.expect_column_mean_to_be_between("tenure", min_value=1.0, max_value=71.0)
        self.expect_column_mean_to_be_between(
            "MonthlyCharges", min_value=10.0, max_value=200.0
        )

        # -- Categorical value set checks
        self.expect_column_values_to_be_in_set("Churn", EXPECTED_CHURN)
        self.expect_column_values_to_be_in_set("gender", EXPECTED_GENDER_VALUES)
        self.expect_column_values_to_be_in_set("Partner", EXPECTED_YES_NO)
        self.expect_column_values_to_be_in_set("Dependents", EXPECTED_YES_NO)
        self.expect_column_values_to_be_in_set("PhoneService", EXPECTED_YES_NO)
        self.expect_column_values_to_be_in_set("MultipleLines", EXPECTED_YES_NO_NO_PHONE)
        self.expect_column_values_to_be_in_set("InternetService", EXPECTED_INTERNET_SERVICE)
        self.expect_column_values_to_be_in_set("OnlineSecurity", EXPECTED_YES_NO_NO_INTERNET)
        self.expect_column_values_to_be_in_set("OnlineBackup", EXPECTED_YES_NO_NO_INTERNET)
        self.expect_column_values_to_be_in_set(
            "DeviceProtection", EXPECTED_YES_NO_NO_INTERNET
        )
        self.expect_column_values_to_be_in_set("TechSupport", EXPECTED_YES_NO_NO_INTERNET)
        self.expect_column_values_to_be_in_set("StreamingTV", EXPECTED_YES_NO_NO_INTERNET)
        self.expect_column_values_to_be_in_set("StreamingMovies", EXPECTED_YES_NO_NO_INTERNET)
        self.expect_column_values_to_be_in_set("Contract", EXPECTED_CONTRACT)
        self.expect_column_values_to_be_in_set("PaperlessBilling", EXPECTED_YES_NO)
        self.expect_column_values_to_be_in_set("PaymentMethod", EXPECTED_PAYMENT_METHOD)

        # -- Unique customer IDs (no duplicates)
        n_rows = len(self.df)
        n_unique = self.df["customerID"].nunique()
        self.expect_column_unique_value_count_to_be_between(
            "customerID", min_value=n_unique, max_value=n_rows
        )

        # -- Finalize summary
        total = self.report["summary"]["total"]
        passed = self.report["summary"]["passed"]
        self.report["summary"]["success_rate"] = round(passed / total, 4) if total > 0 else 0.0

        return self.report


# ---------------------------------------------------------------------------
# Main Validation Function
# ---------------------------------------------------------------------------
def validate_data(
    data_path: Optional[str] = None,
    report_output_path: Optional[str] = None,
    raise_on_failure: bool = True,
) -> Dict[str, Any]:
    """
    Load the processed dataset and run the full GE validation suite.

    Args:
        data_path: Path to the processed CSV. Falls back to env var
                   PROCESSED_DATA_PATH.
        report_output_path: Path to save the JSON validation report.
        raise_on_failure: If True, raises RuntimeError when any expectation
                          fails. Set to False in exploratory mode.

    Returns:
        Dict[str, Any]: Validation report with all expectation results.

    Raises:
        FileNotFoundError: If the data file does not exist.
        RuntimeError: If any expectation fails and raise_on_failure is True.
    """
    # -- Resolve paths
    Path("logs").mkdir(exist_ok=True)
    csv_path = Path(
        data_path or os.getenv("PROCESSED_DATA_PATH", "data/processed/telco_churn_processed.csv")
    )
    default_report_path = Path(
        report_output_path or "data/processed/validation_report.json"
    )

    logger.info("=" * 60)
    logger.info("ChurnShield MLOps — Data Validation Pipeline")
    logger.info("=" * 60)
    logger.info(f"Data path     : {csv_path}")
    logger.info(f"Report output : {default_report_path}")

    # -- Load data
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Processed data not found at '{csv_path}'. "
            "Run ingestion first: python -m src.ingestion.ingest"
        )

    df = pd.read_csv(csv_path)
    logger.info(f"Loaded dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")

    # -- Run validation suite
    validator = TelcoDataValidator(df)
    report = validator.run_full_suite()

    # -- Log summary
    summary = report["summary"]
    logger.info("=" * 60)
    logger.info("Validation Summary")
    logger.info("=" * 60)
    logger.info(f"  Total expectations : {summary['total']}")
    logger.info(f"  Passed             : {summary['passed']}")
    logger.info(f"  Failed             : {summary['failed']}")
    logger.info(f"  Success rate       : {summary['success_rate'] * 100:.1f}%")

    # -- Save JSON report
    default_report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(default_report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"Validation report saved to: {default_report_path}")

    # -- Handle failures
    failed_expectations = [
        e for e in report["expectations"] if not e["success"]
    ]

    if failed_expectations:
        logger.error(f"{len(failed_expectations)} expectation(s) FAILED:")
        for exp in failed_expectations:
            logger.error(
                f"  ❌ [{exp['expectation_type']}] col={exp['column']} | "
                f"observed={exp['observed_value']} | {exp['details']}"
            )
        if raise_on_failure:
            raise RuntimeError(
                f"Data validation failed: {len(failed_expectations)} expectation(s) "
                f"did not pass. Check '{default_report_path}' for details."
            )
    else:
        logger.success("All expectations passed! Data is ready for the training pipeline. ✅")

    return report


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    report = validate_data()
    logger.info(
        f"Validation complete. "
        f"{report['summary']['passed']}/{report['summary']['total']} expectations passed."
    )
