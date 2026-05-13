"""
ChurnShield MLOps Platform - Drift Monitoring Module
=====================================================
Uses Evidently AI to compare a reference dataset vs current batch data.
Generates Data Drift, Target Drift, and Data Quality HTML + JSON reports.

Reports saved to: monitoring/reports/

Usage:
    python -m src.monitoring.monitor
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
    "logs/monitoring.log",
    rotation="10 MB",
    retention="30 days",
    level="DEBUG",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{line} - {message}",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
REPORTS_DIR: str = os.getenv("DRIFT_REPORT_PATH", "monitoring/reports")
DRIFT_THRESHOLD: float = float(os.getenv("DRIFT_THRESHOLD", "0.3"))
REFERENCE_PATH: str = os.getenv("REFERENCE_DATA_PATH", "data/reference/reference_dataset.csv")
CURRENT_PATH: str = os.getenv("PROCESSED_DATA_PATH", "data/processed/telco_churn_processed.csv")

# Features to monitor for drift (numerical + key categoricals)
NUMERICAL_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
CATEGORICAL_FEATURES = [
    "gender", "Partner", "Dependents", "Contract",
    "InternetService", "PaymentMethod",
]


# ---------------------------------------------------------------------------
# Helper: Load Dataset
# ---------------------------------------------------------------------------
def _load_dataset(path: str, label: str) -> pd.DataFrame:
    """
    Load a CSV dataset, log its shape, and return the DataFrame.

    Args:
        path: Path to the CSV file.
        label: Human-readable label for logging (e.g. 'reference', 'current').

    Returns:
        pd.DataFrame: Loaded dataset.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"{label.capitalize()} dataset not found at '{path}'. "
            "Run ingestion first: python -m src.ingestion.ingest"
        )
    df = pd.read_csv(p)
    logger.info(f"Loaded {label} dataset: {df.shape[0]:,} rows × {df.shape[1]} cols | {path}")
    return df


# ---------------------------------------------------------------------------
# Helper: Simple Drift Calculation (fallback without Evidently)
# ---------------------------------------------------------------------------
def _compute_simple_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    numerical_cols: list,
    categorical_cols: list,
) -> Dict[str, Any]:
    """
    Compute a simple drift score using mean shift for numerical features
    and distribution divergence for categoricals. Used as a fallback when
    Evidently AI is not installed.

    Args:
        reference: Reference dataset DataFrame.
        current: Current batch DataFrame.
        numerical_cols: List of numerical feature names to check.
        categorical_cols: List of categorical feature names to check.

    Returns:
        Dict with per-feature drift scores and an overall drift flag.
    """
    feature_drift: Dict[str, Dict[str, float]] = {}
    drifted_features = []

    # -- Numerical: normalized mean shift
    for col in numerical_cols:
        if col not in reference.columns or col not in current.columns:
            continue
        ref_mean = float(reference[col].mean())
        cur_mean = float(current[col].mean())
        ref_std = float(reference[col].std()) + 1e-9
        drift_score = abs(cur_mean - ref_mean) / ref_std
        feature_drift[col] = {
            "reference_mean": round(ref_mean, 4),
            "current_mean": round(cur_mean, 4),
            "drift_score": round(drift_score, 4),
            "drifted": drift_score > DRIFT_THRESHOLD,
        }
        if drift_score > DRIFT_THRESHOLD:
            drifted_features.append(col)
            logger.warning(
                f"Drift detected in '{col}': score={drift_score:.4f} "
                f"(ref_mean={ref_mean:.2f}, cur_mean={cur_mean:.2f})"
            )

    # -- Categorical: top-category share shift
    for col in categorical_cols:
        if col not in reference.columns or col not in current.columns:
            continue
        ref_dist = reference[col].value_counts(normalize=True)
        cur_dist = current[col].value_counts(normalize=True)
        all_cats = set(ref_dist.index) | set(cur_dist.index)
        tvd = sum(
            abs(ref_dist.get(c, 0) - cur_dist.get(c, 0)) for c in all_cats
        ) / 2
        feature_drift[col] = {
            "total_variation_distance": round(float(tvd), 4),
            "drift_score": round(float(tvd), 4),
            "drifted": tvd > DRIFT_THRESHOLD,
        }
        if tvd > DRIFT_THRESHOLD:
            drifted_features.append(col)
            logger.warning(f"Drift detected in '{col}': TVD={tvd:.4f}")

    overall_drift = len(drifted_features) > 0
    overall_score = round(
        sum(v["drift_score"] for v in feature_drift.values()) / max(len(feature_drift), 1),
        4,
    )

    return {
        "method": "simple_statistical",
        "feature_drift": feature_drift,
        "drifted_features": drifted_features,
        "n_drifted_features": len(drifted_features),
        "drift_score": overall_score,
        "drift_detected": overall_drift,
    }


# ---------------------------------------------------------------------------
# Evidently Drift Report
# ---------------------------------------------------------------------------
def _run_evidently_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    reports_dir: Path,
    timestamp: str,
) -> Dict[str, Any]:
    """
    Generate Data Drift, Target Drift, and Data Quality reports using Evidently AI.
    Saves HTML and JSON reports to the monitoring/reports/ directory.

    Args:
        reference: Reference dataset DataFrame (baseline).
        current: Current batch DataFrame to compare against.
        reports_dir: Directory to save report files.
        timestamp: Timestamp string for filenames.

    Returns:
        Dict with drift results, report paths, and drift_detected flag.
    """
    from evidently.report import Report
    from evidently.metric_suite import MetricSuite
    from evidently.metrics import (
        DatasetDriftMetric,
        DatasetMissingValuesMetric,
        ColumnDriftMetric,
    )
    from evidently.metrics.data_drift.dataset_drift_metric import DatasetDriftMetricResults

    logger.info("Running Evidently AI drift analysis...")

    # -- Select only feature columns (drop ID and target)
    drop_cols = ["customerID"]
    ref_features = reference.drop(columns=[c for c in drop_cols if c in reference.columns])
    cur_features = current.drop(columns=[c for c in drop_cols if c in current.columns])

    # -- Data Drift Report
    drift_report = Report(metrics=[
        DatasetDriftMetric(),
        DatasetMissingValuesMetric(),
    ])
    drift_report.run(reference_data=ref_features, current_data=cur_features)

    html_path = reports_dir / f"data_drift_{timestamp}.html"
    json_path = reports_dir / f"data_drift_{timestamp}.json"
    drift_report.save_html(str(html_path))
    drift_report.save_json(str(json_path))
    logger.info(f"Evidently HTML report saved: {html_path}")
    logger.info(f"Evidently JSON report saved: {json_path}")

    # -- Parse drift result from JSON
    with open(json_path) as f:
        drift_json = json.load(f)

    # Extract overall drift flag and share
    drift_detected = False
    drift_share = 0.0
    try:
        for metric in drift_json.get("metrics", []):
            if "DatasetDriftMetric" in metric.get("metric", ""):
                drift_detected = metric["result"].get("dataset_drift", False)
                drift_share = metric["result"].get("drift_share", 0.0)
                break
    except (KeyError, TypeError):
        pass

    return {
        "method": "evidently_ai",
        "drift_detected": drift_detected,
        "drift_share": round(drift_share, 4),
        "drift_score": round(drift_share, 4),
        "html_report": str(html_path),
        "json_report": str(json_path),
    }


# ---------------------------------------------------------------------------
# Main Drift Report Function
# ---------------------------------------------------------------------------
def generate_drift_report(
    reference_path: Optional[str] = None,
    current_path: Optional[str] = None,
    reports_dir: Optional[str] = None,
    trigger_retraining: bool = False,
) -> Dict[str, Any]:
    """
    Compare reference vs current dataset and generate Evidently drift reports.

    Tries Evidently AI first; falls back to simple statistical checks
    if Evidently is not installed or fails.

    Saves to monitoring/reports/:
      - data_drift_{timestamp}.html
      - data_drift_{timestamp}.json
      - drift_summary_{timestamp}.json

    Args:
        reference_path: Path to reference CSV. Falls back to env var
                        REFERENCE_DATA_PATH.
        current_path: Path to current batch CSV. Falls back to env var
                      PROCESSED_DATA_PATH.
        reports_dir: Directory to save reports. Falls back to env var
                     DRIFT_REPORT_PATH.
        trigger_retraining: If True and drift detected, log a critical
                            warning to trigger retraining workflows.

    Returns:
        Dict[str, Any]: Drift results including drift_detected flag,
                        drift_score, report paths, and method used.
    """
    Path("logs").mkdir(exist_ok=True)

    ref_path = reference_path or REFERENCE_PATH
    cur_path = current_path or CURRENT_PATH
    out_dir = Path(reports_dir or REPORTS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    logger.info("=" * 60)
    logger.info("ChurnShield MLOps — Drift Monitoring")
    logger.info("=" * 60)
    logger.info(f"Reference : {ref_path}")
    logger.info(f"Current   : {cur_path}")
    logger.info(f"Reports   : {out_dir}")
    logger.info(f"Threshold : {DRIFT_THRESHOLD}")

    # -- Load datasets
    reference = _load_dataset(ref_path, "reference")
    current = _load_dataset(cur_path, "current")

    # -- Run Evidently (with fallback)
    result: Dict[str, Any] = {}
    try:
        result = _run_evidently_report(
            reference=reference,
            current=current,
            reports_dir=out_dir,
            timestamp=timestamp,
        )
        logger.success("Evidently AI drift analysis complete.")
    except ImportError:
        logger.warning(
            "Evidently AI not installed or import failed. "
            "Falling back to simple statistical drift detection."
        )
        result = _compute_simple_drift(
            reference=reference,
            current=current,
            numerical_cols=NUMERICAL_FEATURES,
            categorical_cols=CATEGORICAL_FEATURES,
        )
    except Exception as exc:
        logger.error(f"Evidently analysis failed: {exc}. Falling back to simple detection.")
        result = _compute_simple_drift(
            reference=reference,
            current=current,
            numerical_cols=NUMERICAL_FEATURES,
            categorical_cols=CATEGORICAL_FEATURES,
        )

    # -- Enrich result
    result["timestamp"] = timestamp
    result["reference_rows"] = len(reference)
    result["current_rows"] = len(current)
    result["drift_threshold"] = DRIFT_THRESHOLD

    # -- Log drift outcome
    if result.get("drift_detected"):
        logger.warning(
            f"⚠️  DATA DRIFT DETECTED! "
            f"Score={result.get('drift_score', 'N/A')}, "
            f"Threshold={DRIFT_THRESHOLD}"
        )
        if trigger_retraining:
            logger.critical(
                "Retraining triggered due to data drift. "
                "Run: python pipelines/training_pipeline.py"
            )
    else:
        logger.success(
            f"✅ No significant drift detected. "
            f"Score={result.get('drift_score', 'N/A')}"
        )

    # -- Save summary JSON
    summary_path = out_dir / f"drift_summary_{timestamp}.json"
    with open(summary_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    result["summary_path"] = str(summary_path)
    logger.info(f"Drift summary saved to: {summary_path}")

    logger.info("=" * 60)
    return result


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    report = generate_drift_report(trigger_retraining=True)
    drift_status = "DRIFT DETECTED ⚠️" if report["drift_detected"] else "No Drift ✅"
    logger.info(f"Drift monitoring complete: {drift_status}")
