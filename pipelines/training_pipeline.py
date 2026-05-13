"""
ChurnShield MLOps Platform - Prefect Orchestration Pipeline
============================================================
Defines a Prefect flow connecting all pipeline stages in order:
  1. ingest_data
  2. validate_data
  3. preprocess_data
  4. train_models
  5. evaluate_and_register
  6. generate_drift_report

Each task has retries=2, retry_delay=30s, structured logging,
and proper error handling.

Usage:
    python pipelines/training_pipeline.py
    prefect deployment run 'churnshield-training-pipeline/weekly-training'
"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger
from prefect import flow, task, get_run_logger
from prefect.tasks import task_input_hash

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
    "logs/pipeline.log",
    rotation="10 MB",
    retention="30 days",
    level="DEBUG",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{line} - {message}",
)

# ---------------------------------------------------------------------------
# Pipeline Configuration
# ---------------------------------------------------------------------------
PROCESSED_DATA_PATH: str = os.getenv(
    "PROCESSED_DATA_PATH", "data/processed/telco_churn_processed.csv"
)
RAW_DATA_PATH: str = os.getenv("RAW_DATA_PATH", "data/raw/telco_churn.csv")


# ---------------------------------------------------------------------------
# Task 1: Data Ingestion
# ---------------------------------------------------------------------------
@task(
    name="ingest-data",
    description="Load raw Telco data or generate synthetic dataset. Save to data/processed/.",
    retries=2,
    retry_delay_seconds=30,
    tags=["data", "ingestion"],
)
def ingest_data_task(generate_synthetic: bool = False) -> Dict[str, Any]:
    """
    Prefect task wrapper for the data ingestion pipeline step.

    Args:
        generate_synthetic: Force synthetic data generation even if CSV exists.

    Returns:
        Dict with shape info and output path.

    Raises:
        Exception: Propagated from ingest_data() on critical failures.
    """
    prefect_logger = get_run_logger()
    prefect_logger.info("Starting data ingestion task...")

    try:
        from src.ingestion.ingest import ingest_data

        df = ingest_data(generate_synthetic=generate_synthetic)
        result = {
            "rows": len(df),
            "columns": len(df.columns),
            "churn_rate": float((df["Churn"] == "Yes").mean()),
            "output_path": PROCESSED_DATA_PATH,
            "status": "success",
        }
        prefect_logger.info(
            f"Ingestion complete: {result['rows']:,} rows, "
            f"churn rate={result['churn_rate']:.3f}"
        )
        return result

    except Exception as exc:
        prefect_logger.error(f"Ingestion failed: {exc}")
        raise


# ---------------------------------------------------------------------------
# Task 2: Data Validation
# ---------------------------------------------------------------------------
@task(
    name="validate-data",
    description="Run Great Expectations validation suite on processed dataset.",
    retries=2,
    retry_delay_seconds=30,
    tags=["data", "validation"],
)
def validate_data_task(raise_on_failure: bool = True) -> Dict[str, Any]:
    """
    Prefect task wrapper for the data validation pipeline step.

    Args:
        raise_on_failure: If True, raise RuntimeError on any failed expectation.

    Returns:
        Dict with validation summary (total, passed, failed, success_rate).

    Raises:
        RuntimeError: If raise_on_failure=True and any expectation fails.
    """
    prefect_logger = get_run_logger()
    prefect_logger.info("Starting data validation task...")

    try:
        from src.validation.validate import validate_data

        report = validate_data(raise_on_failure=raise_on_failure)
        summary = report["summary"]
        result = {
            "total_expectations": summary["total"],
            "passed": summary["passed"],
            "failed": summary["failed"],
            "success_rate": summary["success_rate"],
            "status": "success" if summary["failed"] == 0 else "warnings",
        }
        prefect_logger.info(
            f"Validation complete: {result['passed']}/{result['total_expectations']} "
            f"expectations passed ({result['success_rate'] * 100:.1f}%)"
        )
        return result

    except Exception as exc:
        prefect_logger.error(f"Validation failed: {exc}")
        raise


# ---------------------------------------------------------------------------
# Task 3: Data Preprocessing
# ---------------------------------------------------------------------------
@task(
    name="preprocess-data",
    description="Feature engineering + sklearn ColumnTransformer fit/transform. Save artifacts.",
    retries=2,
    retry_delay_seconds=30,
    tags=["data", "preprocessing"],
)
def preprocess_data_task(test_size: float = 0.2, random_state: int = 42) -> Dict[str, Any]:
    """
    Prefect task wrapper for the preprocessing pipeline step.

    Args:
        test_size: Fraction of data reserved for testing (default 0.2).
        random_state: Random seed for reproducibility (default 42).

    Returns:
        Dict with train/test shapes and feature counts.
    """
    prefect_logger = get_run_logger()
    prefect_logger.info("Starting preprocessing task...")

    try:
        from src.preprocessing.preprocess import preprocess_data

        result_data = preprocess_data(test_size=test_size, random_state=random_state)
        result = {
            "train_samples": len(result_data["X_train"]),
            "test_samples": len(result_data["X_test"]),
            "n_features": result_data["X_train"].shape[1],
            "churn_rate_train": float(result_data["y_train"].mean()),
            "churn_rate_test": float(result_data["y_test"].mean()),
            "status": "success",
        }
        prefect_logger.info(
            f"Preprocessing complete: {result['train_samples']:,} train | "
            f"{result['test_samples']:,} test | {result['n_features']} features"
        )
        return result

    except Exception as exc:
        prefect_logger.error(f"Preprocessing failed: {exc}")
        raise


# ---------------------------------------------------------------------------
# Task 4: Model Training
# ---------------------------------------------------------------------------
@task(
    name="train-models",
    description="Train LogisticRegression, RandomForest, XGBoost. Log all to MLflow.",
    retries=2,
    retry_delay_seconds=60,
    tags=["training", "mlflow"],
)
def train_models_task(
    tracking_uri: Optional[str] = None,
    experiment_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Prefect task wrapper for the model training pipeline step.

    Args:
        tracking_uri: MLflow tracking server URI.
        experiment_name: MLflow experiment name.

    Returns:
        Dict with per-model run IDs and metrics summary.
    """
    prefect_logger = get_run_logger()
    prefect_logger.info("Starting model training task (3 models)...")

    try:
        from src.training.train import train_models

        results = train_models(
            tracking_uri=tracking_uri,
            experiment_name=experiment_name,
        )
        summary = {
            "n_models_trained": len(results),
            "runs": [
                {
                    "model_name": r["model_name"],
                    "run_id": r["run_id"],
                    "roc_auc": r["metrics"]["roc_auc"],
                    "f1_score": r["metrics"]["f1_score"],
                }
                for r in results
            ],
            "best_model": max(results, key=lambda r: r["metrics"]["roc_auc"])["model_name"],
            "status": "success",
        }
        prefect_logger.info(
            f"Training complete: {summary['n_models_trained']} models. "
            f"Best: {summary['best_model']}"
        )
        return summary

    except Exception as exc:
        prefect_logger.error(f"Training failed: {exc}")
        raise


# ---------------------------------------------------------------------------
# Task 5: Model Evaluation & Registration
# ---------------------------------------------------------------------------
@task(
    name="evaluate-and-register",
    description="Select best model by ROC-AUC. Register in MLflow Model Registry → Production.",
    retries=2,
    retry_delay_seconds=30,
    tags=["evaluation", "registry", "mlflow"],
)
def evaluate_and_register_task(
    tracking_uri: Optional[str] = None,
    experiment_name: Optional[str] = None,
    registry_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Prefect task wrapper for model evaluation and registry promotion.

    Args:
        tracking_uri: MLflow tracking server URI.
        experiment_name: MLflow experiment name.
        registry_name: Model Registry name to register under.

    Returns:
        Dict with best model info, version, and stage.
    """
    prefect_logger = get_run_logger()
    prefect_logger.info("Starting model evaluation & registry task...")

    try:
        from src.evaluation.evaluate import evaluate_and_register

        result = evaluate_and_register(
            tracking_uri=tracking_uri,
            experiment_name=experiment_name,
            registry_name=registry_name,
        )
        prefect_logger.info(
            f"Model registered: {result['best_model_name']} "
            f"v{result['model_version']} → {result['stage']}"
        )
        return result

    except Exception as exc:
        prefect_logger.error(f"Evaluation/registration failed: {exc}")
        raise


# ---------------------------------------------------------------------------
# Task 6: Generate Drift Report
# ---------------------------------------------------------------------------
@task(
    name="generate-drift-report",
    description="Run Evidently AI drift detection. Alert if data drift detected.",
    retries=2,
    retry_delay_seconds=30,
    tags=["monitoring", "drift"],
)
def generate_drift_report_task(
    reference_path: Optional[str] = None,
    current_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Prefect task wrapper for Evidently drift monitoring.

    If drift is detected (drift_detected=True), logs a warning and the
    result is surfaced in Prefect UI for manual or automated remediation.

    Args:
        reference_path: Path to the reference dataset CSV.
        current_path: Path to the current batch dataset CSV.

    Returns:
        Dict with drift_detected flag and report paths.
    """
    prefect_logger = get_run_logger()
    prefect_logger.info("Starting drift report generation task...")

    try:
        from src.monitoring.monitor import generate_drift_report

        result = generate_drift_report(
            reference_path=reference_path,
            current_path=current_path,
        )
        if result.get("drift_detected"):
            prefect_logger.warning(
                "⚠️  DATA DRIFT DETECTED! "
                f"Drift score: {result.get('drift_score', 'N/A')}. "
                "Consider triggering retraining."
            )
        else:
            prefect_logger.info(
                f"No significant drift detected. "
                f"Drift score: {result.get('drift_score', 'N/A')}"
            )
        return result

    except Exception as exc:
        # Drift monitoring failure should not block the pipeline
        prefect_logger.warning(
            f"Drift report generation encountered an error: {exc}. "
            "Continuing pipeline — monitoring is non-blocking."
        )
        return {
            "drift_detected": False,
            "status": "error",
            "error": str(exc),
        }


# ---------------------------------------------------------------------------
# Main Prefect Flow
# ---------------------------------------------------------------------------
@flow(
    name="churnshield-training-pipeline",
    description=(
        "End-to-end ChurnShield MLOps pipeline: ingest → validate → "
        "preprocess → train → evaluate → monitor drift."
    ),
    version="1.0.0",
    retries=1,
    retry_delay_seconds=60,
)
def churnshield_pipeline(
    generate_synthetic: bool = False,
    test_size: float = 0.2,
    random_state: int = 42,
    raise_on_validation_failure: bool = True,
    tracking_uri: Optional[str] = None,
    experiment_name: Optional[str] = None,
    registry_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Full ChurnShield MLOps training pipeline as a Prefect flow.

    Executes all 6 pipeline stages sequentially. Each stage is a Prefect
    task with retries=2. The drift report task is non-blocking (failures
    are caught and logged as warnings, not pipeline failures).

    Args:
        generate_synthetic: Force synthetic data generation (default False).
        test_size: Train/test split fraction (default 0.2).
        random_state: Global random seed (default 42).
        raise_on_validation_failure: Halt pipeline on data validation failure.
        tracking_uri: MLflow tracking URI (falls back to env var).
        experiment_name: MLflow experiment name (falls back to env var).
        registry_name: Model Registry name (falls back to env var).

    Returns:
        Dict with results from all pipeline stages.
    """
    prefect_logger = get_run_logger()

    prefect_logger.info("=" * 60)
    prefect_logger.info("ChurnShield MLOps — Full Training Pipeline")
    prefect_logger.info(f"Started at: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    prefect_logger.info("=" * 60)

    # -- Create required directories
    for d in ["logs", "models", "data/processed", "data/raw", "data/reference"]:
        Path(d).mkdir(parents=True, exist_ok=True)

    # -- Stage 1: Ingest
    prefect_logger.info("[1/6] Ingesting data...")
    ingestion_result = ingest_data_task(generate_synthetic=generate_synthetic)

    # -- Stage 2: Validate
    prefect_logger.info("[2/6] Validating data...")
    validation_result = validate_data_task(
        raise_on_failure=raise_on_validation_failure
    )

    # -- Stage 3: Preprocess
    prefect_logger.info("[3/6] Preprocessing data...")
    preprocessing_result = preprocess_data_task(
        test_size=test_size,
        random_state=random_state,
    )

    # -- Stage 4: Train
    prefect_logger.info("[4/6] Training models...")
    training_result = train_models_task(
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
    )

    # -- Stage 5: Evaluate & Register
    prefect_logger.info("[5/6] Evaluating and registering best model...")
    evaluation_result = evaluate_and_register_task(
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
        registry_name=registry_name,
    )

    # -- Stage 6: Drift Report (non-blocking)
    prefect_logger.info("[6/6] Generating drift report...")
    drift_result = generate_drift_report_task()

    # -- Pipeline Summary
    pipeline_result = {
        "pipeline_version": "1.0.0",
        "run_timestamp": datetime.utcnow().isoformat(),
        "stages": {
            "ingestion": ingestion_result,
            "validation": validation_result,
            "preprocessing": preprocessing_result,
            "training": training_result,
            "evaluation": evaluation_result,
            "drift_monitoring": drift_result,
        },
    }

    prefect_logger.info("=" * 60)
    prefect_logger.info("Pipeline Complete!")
    prefect_logger.info(
        f"  Best model : {evaluation_result.get('best_model_name', 'N/A')} "
        f"v{evaluation_result.get('model_version', 'N/A')}"
    )
    prefect_logger.info(
        f"  Drift alert: {'YES ⚠️' if drift_result.get('drift_detected') else 'NO ✅'}"
    )
    prefect_logger.info("=" * 60)

    return pipeline_result


# ---------------------------------------------------------------------------
# Prefect Deployment Configuration (Weekly Schedule)
# ---------------------------------------------------------------------------
def create_deployment() -> None:
    """
    Create and apply a Prefect deployment with a weekly cron schedule.

    Schedule: Every Monday at 02:00 UTC.
    Run with: prefect deployment run 'churnshield-training-pipeline/weekly-training'
    """
    from prefect.deployments import Deployment
    from prefect.server.schemas.schedules import CronSchedule

    deployment = Deployment.build_from_flow(
        flow=churnshield_pipeline,
        name="weekly-training",
        version="1.0.0",
        work_queue_name="default",
        schedule=CronSchedule(cron="0 2 * * 1", timezone="UTC"),  # Monday 02:00 UTC
        parameters={
            "generate_synthetic": False,
            "test_size": 0.2,
            "random_state": 42,
            "raise_on_validation_failure": True,
        },
        description=(
            "Weekly automated ChurnShield retraining pipeline. "
            "Runs every Monday at 02:00 UTC."
        ),
        tags=["churnshield", "weekly", "automated"],
    )
    deployment.apply()
    logger.success(
        "Deployment 'weekly-training' applied successfully. "
        "Run with: prefect deployment run "
        "'churnshield-training-pipeline/weekly-training'"
    )


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ChurnShield MLOps Training Pipeline")
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Generate synthetic data instead of loading from CSV",
    )
    parser.add_argument(
        "--deploy",
        action="store_true",
        help="Create Prefect deployment instead of running the flow",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test set fraction (default: 0.2)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    args = parser.parse_args()

    if args.deploy:
        logger.info("Creating Prefect deployment...")
        create_deployment()
    else:
        logger.info("Running ChurnShield pipeline...")
        result = churnshield_pipeline(
            generate_synthetic=args.synthetic,
            test_size=args.test_size,
            random_state=args.seed,
        )
        logger.success("Pipeline finished successfully.")
