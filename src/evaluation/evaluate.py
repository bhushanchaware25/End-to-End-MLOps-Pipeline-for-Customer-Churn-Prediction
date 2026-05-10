"""
ChurnShield MLOps Platform - Model Evaluation & Registry Module
===============================================================
Compares all runs in the MLflow 'ChurnShield' experiment, selects the
best model by ROC-AUC score, registers it in the MLflow Model Registry
as 'ChurnShield-Model', and transitions it to 'Production' stage.

Also generates and logs a Model Card as a text artifact.

Usage:
    python -m src.evaluation.evaluate
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import mlflow
import mlflow.sklearn
import mlflow.xgboost
from mlflow.tracking import MlflowClient
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
    "logs/evaluation.log",
    rotation="10 MB",
    retention="30 days",
    level="DEBUG",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{line} - {message}",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EXPERIMENT_NAME: str = os.getenv("MLFLOW_EXPERIMENT_NAME", "ChurnShield")
MODEL_REGISTRY_NAME: str = os.getenv("MLFLOW_MODEL_NAME", "ChurnShield-Model")
MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
PRIMARY_METRIC: str = "roc_auc"


# ---------------------------------------------------------------------------
# Model Card Generator
# ---------------------------------------------------------------------------
def generate_model_card(
    model_name: str,
    run_id: str,
    metrics: Dict[str, float],
    params: Dict[str, Any],
    model_version: str,
) -> str:
    """
    Generate a structured Model Card as a markdown-formatted string.

    The model card documents the model's purpose, performance, limitations,
    training data characteristics, and intended use — following responsible AI
    documentation practices.

    Args:
        model_name: Name of the winning model architecture.
        run_id: MLflow run ID of the winning run.
        metrics: Dictionary of evaluation metrics (accuracy, f1, roc_auc, etc.).
        params: Dictionary of model hyperparameters.
        model_version: Registered model version string.

    Returns:
        str: Markdown-formatted model card text.
    """
    card = f"""# ChurnShield Model Card
## Model: {model_name} v{model_version}

**Generated**: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}
**MLflow Run ID**: `{run_id}`
**Registry Name**: `{MODEL_REGISTRY_NAME}`
**Stage**: Production

---

## Model Overview

| Field            | Value                                           |
|------------------|-------------------------------------------------|
| **Task**         | Binary Classification — Customer Churn          |
| **Dataset**      | Telco Customer Churn (Kaggle / Synthetic)       |
| **Target**       | `Churn` (1 = Churned, 0 = Retained)            |
| **Architecture** | {model_name}                                   |
| **Version**      | {model_version}                                |
| **Run ID**       | `{run_id[:8]}...`                              |

---

## Performance Metrics (Test Set)

| Metric     | Value   |
|------------|---------|
| Accuracy   | {metrics.get('accuracy', 0):.4f}  |
| Precision  | {metrics.get('precision', 0):.4f} |
| Recall     | {metrics.get('recall', 0):.4f}    |
| F1-Score   | {metrics.get('f1_score', 0):.4f}  |
| ROC-AUC    | {metrics.get('roc_auc', 0):.4f}   |

> **Selection Criterion**: Best ROC-AUC score across all trained models in the ChurnShield experiment.

---

## Hyperparameters

```json
{json.dumps(params, indent=2)}
```

---

## Training Data

- **Source**: Telco Customer Churn Dataset
- **Features**: 15 original + 5 engineered = 20 input features
- **Preprocessing**: Median imputation + StandardScaler (numerical), most_frequent imputation + OneHotEncoder (categorical)
- **Train/Test Split**: 80% / 20% stratified
- **Class Imbalance**: ~73% No Churn / ~27% Churn

---

## Feature Engineering

| Feature               | Description                                    |
|-----------------------|------------------------------------------------|
| tenure_months_squared | Captures non-linear retention effect           |
| avg_monthly_spend     | TotalCharges / (tenure + 1)                   |
| has_any_addon         | Binary: any internet add-on service active     |
| services_count        | Total count of active services (0–9)           |
| is_long_term_contract | Binary: 1 if One year or Two year contract     |

---

## Intended Use

- **Primary Use**: Predict probability of customer churn to enable proactive retention campaigns.
- **Users**: Marketing, Customer Success, and Data Science teams.
- **Deployment**: FastAPI REST API with real-time single-customer predictions.

---

## Limitations & Risks

- Model trained on historical data; performance may degrade as customer behavior evolves.
- Predictions are probabilistic, not deterministic — human review recommended for high-stakes decisions.
- Dataset may not represent all customer demographics equally.
- Drift monitoring (Evidently AI) should be checked weekly for data and target drift.

---

## Ethical Considerations

- Do not use churn predictions to discriminate against customer segments.
- Ensure retention offers are applied equitably across demographics.
- Model decisions should augment — not replace — human judgment.

---

## Monitoring

- **Drift Detection**: Evidently AI weekly batch monitoring
- **Retraining Trigger**: Data drift score > 0.3 or ROC-AUC degradation > 5%
- **Scheduled Retraining**: Weekly via Prefect orchestration

---

*This model card was auto-generated by the ChurnShield MLOps Platform.*
"""
    return card


# ---------------------------------------------------------------------------
# Get Best Run
# ---------------------------------------------------------------------------
def get_best_run(
    client: MlflowClient,
    experiment_name: str,
    metric: str = PRIMARY_METRIC,
) -> Dict[str, Any]:
    """
    Query all completed runs in the experiment and select the best by metric.

    Args:
        client: Initialized MlflowClient instance.
        experiment_name: Name of the MLflow experiment to search.
        metric: Metric name to rank runs by (higher is better).

    Returns:
        Dict[str, Any]: Best run info including run_id, metrics, params.

    Raises:
        ValueError: If no completed runs found in the experiment.
    """
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(
            f"Experiment '{experiment_name}' not found. "
            "Run training first: python -m src.training.train"
        )

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="status = 'FINISHED'",
        order_by=[f"metrics.{metric} DESC"],
        max_results=10,
    )

    if not runs:
        raise ValueError(
            f"No completed runs found in experiment '{experiment_name}'. "
            "Run training first: python -m src.training.train"
        )

    best_run = runs[0]
    logger.info(f"Found {len(runs)} completed runs in experiment '{experiment_name}'.")
    logger.info(f"Best run selected:")
    logger.info(f"  Run ID    : {best_run.info.run_id}")
    logger.info(f"  Model     : {best_run.data.tags.get('model_type', 'Unknown')}")
    logger.info(f"  {metric:<10}: {best_run.data.metrics.get(metric, 0):.4f}")

    return {
        "run_id": best_run.info.run_id,
        "metrics": dict(best_run.data.metrics),
        "params": dict(best_run.data.params),
        "tags": dict(best_run.data.tags),
        "model_name": best_run.data.tags.get("model_type", "Unknown"),
    }


# ---------------------------------------------------------------------------
# Register and Promote Model
# ---------------------------------------------------------------------------
def register_and_promote_model(
    client: MlflowClient,
    run_id: str,
    model_name_tag: str,
    registry_name: str,
) -> str:
    """
    Register the best model run in the MLflow Model Registry and
    transition it to the 'Production' stage.

    Any previously 'Production' versions are transitioned to 'Archived'.

    Args:
        client: Initialized MlflowClient instance.
        run_id: MLflow run ID of the model to register.
        model_name_tag: Architecture name tag (e.g., 'XGBoost').
        registry_name: Name under which to register in the Model Registry.

    Returns:
        str: Version string of the newly registered model.
    """
    # -- Determine artifact path from run tags / naming convention
    model_artifact_path = f"model_{model_name_tag}"
    model_uri = f"runs:/{run_id}/{model_artifact_path}"

    logger.info(f"Registering model from URI: {model_uri}")
    logger.info(f"Registry name: {registry_name}")

    # -- Register model
    model_version_info = mlflow.register_model(
        model_uri=model_uri,
        name=registry_name,
    )
    version = model_version_info.version
    logger.info(f"Model registered. Version: {version}")

    # -- Archive any existing Production versions
    try:
        existing_versions = client.search_model_versions(
            f"name='{registry_name}'"
        )
        for mv in existing_versions:
            if mv.current_stage == "Production" and mv.version != version:
                client.transition_model_version_stage(
                    name=registry_name,
                    version=mv.version,
                    stage="Archived",
                    archive_existing_versions=False,
                )
                logger.info(
                    f"Archived previous Production version: {mv.version}"
                )
    except Exception as exc:
        logger.warning(f"Could not archive previous versions: {exc}")

    # -- Transition new version to Production
    client.transition_model_version_stage(
        name=registry_name,
        version=version,
        stage="Production",
        archive_existing_versions=True,
    )
    logger.success(
        f"Model v{version} transitioned to 'Production' stage in registry '{registry_name}'."
    )

    # -- Add description to registered version
    client.update_model_version(
        name=registry_name,
        version=version,
        description=(
            f"Best performing {model_name_tag} model selected by ROC-AUC. "
            f"Auto-promoted by ChurnShield MLOps evaluation pipeline."
        ),
    )

    return str(version)


# ---------------------------------------------------------------------------
# Main Evaluation Function
# ---------------------------------------------------------------------------
def evaluate_and_register(
    tracking_uri: Optional[str] = None,
    experiment_name: Optional[str] = None,
    registry_name: Optional[str] = None,
    metric: str = PRIMARY_METRIC,
) -> Dict[str, Any]:
    """
    Compare all MLflow runs, select the best model, register it, and
    generate a Model Card artifact.

    Args:
        tracking_uri: MLflow tracking server URI. Falls back to env var
                      MLFLOW_TRACKING_URI.
        experiment_name: MLflow experiment name. Falls back to env var
                         MLFLOW_EXPERIMENT_NAME.
        registry_name: Model Registry name. Falls back to env var
                       MLFLOW_MODEL_NAME.
        metric: Primary metric for model selection (default: 'roc_auc').

    Returns:
        Dict[str, Any]: Evaluation result with best model info and version.
    """
    Path("logs").mkdir(exist_ok=True)

    # -- Resolve configuration
    uri = tracking_uri or MLFLOW_TRACKING_URI
    exp_name = experiment_name or EXPERIMENT_NAME
    reg_name = registry_name or MODEL_REGISTRY_NAME

    logger.info("=" * 60)
    logger.info("ChurnShield MLOps — Model Evaluation & Registry")
    logger.info("=" * 60)
    logger.info(f"Tracking URI    : {uri}")
    logger.info(f"Experiment      : {exp_name}")
    logger.info(f"Registry name   : {reg_name}")
    logger.info(f"Selection metric: {metric}")

    # -- Initialize MLflow
    mlflow.set_tracking_uri(uri)
    client = MlflowClient(tracking_uri=uri)

    # -- Find best run
    best_run = get_best_run(client=client, experiment_name=exp_name, metric=metric)
    run_id = best_run["run_id"]
    model_name_tag = best_run["model_name"]
    metrics = best_run["metrics"]
    params = best_run["params"]

    logger.info("=" * 60)
    logger.info("Best Model Results")
    logger.info("=" * 60)
    for k, v in metrics.items():
        if isinstance(v, float):
            logger.info(f"  {k:<20}: {v:.4f}")

    # -- Register model and promote to Production
    version = register_and_promote_model(
        client=client,
        run_id=run_id,
        model_name_tag=model_name_tag,
        registry_name=reg_name,
    )

    # -- Generate and log Model Card
    model_card_text = generate_model_card(
        model_name=model_name_tag,
        run_id=run_id,
        metrics=metrics,
        params=params,
        model_version=version,
    )

    # Save model card locally and log as MLflow artifact
    model_card_path = Path("models") / "model_card.md"
    model_card_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_card_path, "w", encoding="utf-8") as f:
        f.write(model_card_text)

    with mlflow.start_run(run_id=run_id):
        mlflow.log_artifact(str(model_card_path), artifact_path="model_card")
        mlflow.set_tag("model_version", version)
        mlflow.set_tag("registry_stage", "Production")

    logger.success(f"Model Card saved to: {model_card_path}")

    # -- Save evaluation summary
    eval_summary = {
        "timestamp": datetime.utcnow().isoformat(),
        "best_run_id": run_id,
        "best_model_name": model_name_tag,
        "model_version": version,
        "registry_name": reg_name,
        "stage": "Production",
        "selection_metric": metric,
        "metrics": metrics,
    }
    summary_path = Path("models") / "evaluation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(eval_summary, f, indent=2)
    logger.info(f"Evaluation summary saved to: {summary_path}")

    logger.info("=" * 60)
    logger.success(
        f"✅ Model '{model_name_tag}' v{version} promoted to Production in registry '{reg_name}'"
    )
    logger.info("=" * 60)

    return eval_summary


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    result = evaluate_and_register()
    logger.info(
        f"Evaluation complete. "
        f"Model: {result['best_model_name']} v{result['model_version']} → Production"
    )
