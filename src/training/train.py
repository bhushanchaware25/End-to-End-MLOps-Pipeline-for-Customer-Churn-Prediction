"""
ChurnShield MLOps Platform - Model Training Module
===================================================
Trains three models on the preprocessed Telco Customer Churn dataset
inside a single MLflow experiment called 'ChurnShield'.

Models trained:
  1. Logistic Regression  (baseline)
  2. Random Forest        (ensemble)
  3. XGBoost              (gradient boosting)

For each model, logs to MLflow:
  - All hyperparameters
  - Metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC
  - Confusion matrix PNG artifact
  - Feature importance plot PNG artifact
  - The serialized model itself (mlflow.sklearn / mlflow.xgboost)

Usage:
    python -m src.training.train
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server environments
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from loguru import logger
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    ConfusionMatrixDisplay,
)
from xgboost import XGBClassifier

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
    "logs/training.log",
    rotation="10 MB",
    retention="30 days",
    level="DEBUG",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{line} - {message}",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EXPERIMENT_NAME: str = os.getenv("MLFLOW_EXPERIMENT_NAME", "ChurnShield")
MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
ARTIFACT_DIR: str = "training_artifacts"


# ---------------------------------------------------------------------------
# Model Definitions
# ---------------------------------------------------------------------------
def get_model_configs() -> List[Dict[str, Any]]:
    """
    Return the list of model configurations to train.

    Each config contains:
        - name: Unique identifier string for MLflow run name
        - model: Instantiated sklearn-compatible estimator
        - params: Dict of hyperparameters to log to MLflow
        - flavor: MLflow logging flavor ('sklearn' or 'xgboost')

    Returns:
        List[Dict[str, Any]]: Ordered list of model configurations.
    """
    configs = [
        {
            "name": "LogisticRegression",
            "model": LogisticRegression(
                C=1.0,
                max_iter=1000,
                solver="lbfgs",
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ),
            "params": {
                "C": 1.0,
                "max_iter": 1000,
                "solver": "lbfgs",
                "class_weight": "balanced",
                "random_state": 42,
            },
            "flavor": "sklearn",
        },
        {
            "name": "RandomForest",
            "model": RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ),
            "params": {
                "n_estimators": 200,
                "max_depth": 10,
                "min_samples_split": 5,
                "min_samples_leaf": 2,
                "class_weight": "balanced",
                "random_state": 42,
            },
            "flavor": "sklearn",
        },
        {
            "name": "XGBoost",
            "model": XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=3,   # handles class imbalance (≈73/27 ratio)
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=42,
                n_jobs=-1,
                verbosity=0,
            ),
            "params": {
                "n_estimators": 300,
                "max_depth": 6,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "scale_pos_weight": 3,
                "random_state": 42,
            },
            "flavor": "xgboost",
        },
    ]
    return configs


# ---------------------------------------------------------------------------
# Metrics Computation
# ---------------------------------------------------------------------------
def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> Dict[str, float]:
    """
    Compute classification metrics for churn prediction.

    Args:
        y_true: Ground-truth binary labels (0/1).
        y_pred: Predicted binary labels (0/1).
        y_prob: Predicted positive-class probabilities.

    Returns:
        Dict[str, float]: Dictionary with metric names and values.
    """
    return {
        "accuracy": float(round(accuracy_score(y_true, y_pred), 6)),
        "precision": float(round(precision_score(y_true, y_pred, zero_division=0), 6)),
        "recall": float(round(recall_score(y_true, y_pred, zero_division=0), 6)),
        "f1_score": float(round(f1_score(y_true, y_pred, zero_division=0), 6)),
        "roc_auc": float(round(roc_auc_score(y_true, y_prob), 6)),
    }


# ---------------------------------------------------------------------------
# Plot: Confusion Matrix
# ---------------------------------------------------------------------------
def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    output_dir: Path,
) -> Path:
    """
    Generate and save a styled confusion matrix plot as PNG.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        model_name: Name of the model (used in title and filename).
        output_dir: Directory to save the PNG file.

    Returns:
        Path: Absolute path to the saved PNG file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["No Churn", "Churn"],
    )
    disp.plot(
        ax=ax,
        colorbar=True,
        cmap="Blues",
        values_format="d",
    )
    ax.set_title(
        f"Confusion Matrix — {model_name}\n"
        f"TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]}",
        fontsize=12,
        fontweight="bold",
        pad=15,
    )
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    plt.tight_layout()

    png_path = output_dir / f"confusion_matrix_{model_name}.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.debug(f"Confusion matrix saved: {png_path}")
    return png_path


# ---------------------------------------------------------------------------
# Plot: Feature Importance
# ---------------------------------------------------------------------------
def plot_feature_importance(
    model: Any,
    feature_names: List[str],
    model_name: str,
    output_dir: Path,
    top_n: int = 20,
) -> Optional[Path]:
    """
    Generate and save a horizontal bar chart of feature importances.

    Supports:
        - RandomForestClassifier / XGBClassifier via .feature_importances_
        - LogisticRegression via abs(coef_[0])

    Args:
        model: Trained estimator instance.
        feature_names: List of feature name strings (post-OHE).
        model_name: Name of the model (used in title and filename).
        output_dir: Directory to save the PNG file.
        top_n: Number of top features to display.

    Returns:
        Optional[Path]: Path to the saved PNG, or None if not applicable.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    importances: Optional[np.ndarray] = None

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        logger.warning(f"{model_name} does not expose feature importances. Skipping plot.")
        return None

    # Align feature names with importance array length
    n_features = len(importances)
    if len(feature_names) != n_features:
        logger.warning(
            f"Feature name count ({len(feature_names)}) != importance count ({n_features}). "
            "Using generic feature names."
        )
        feature_names = [f"feature_{i}" for i in range(n_features)]

    # Select top N features
    indices = np.argsort(importances)[::-1][:top_n]
    top_importances = importances[indices]
    top_names = [feature_names[i] for i in indices]

    # Clean up feature names (remove sklearn prefixes like 'numerical__', 'categorical__')
    top_names_clean = [
        name.replace("numerical__", "").replace("categorical__", "")
        for name in top_names
    ]

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.4)))
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(top_importances)))  # type: ignore
    bars = ax.barh(
        range(len(top_importances)),
        top_importances[::-1],
        color=colors[::-1],
        edgecolor="white",
        linewidth=0.5,
    )
    ax.set_yticks(range(len(top_importances)))
    ax.set_yticklabels(top_names_clean[::-1], fontsize=9)
    ax.set_xlabel("Feature Importance", fontsize=11)
    ax.set_title(
        f"Top {top_n} Feature Importances — {model_name}",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.3, linestyle="--")

    # Add value labels on bars
    for bar, val in zip(bars, top_importances[::-1]):
        ax.text(
            bar.get_width() + max(top_importances) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}",
            va="center",
            fontsize=8,
            color="dimgray",
        )

    plt.tight_layout()
    png_path = output_dir / f"feature_importance_{model_name}.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.debug(f"Feature importance plot saved: {png_path}")
    return png_path


# ---------------------------------------------------------------------------
# Single Model Training Run
# ---------------------------------------------------------------------------
def train_single_model(
    config: Dict[str, Any],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    feature_names: List[str],
    artifact_dir: Path,
) -> Dict[str, Any]:
    """
    Train a single model, compute metrics, generate plots, and log all to MLflow.

    Args:
        config: Model configuration dict (name, model, params, flavor).
        X_train: Training features DataFrame.
        X_test: Test features DataFrame.
        y_train: Training target Series.
        y_test: Test target Series.
        feature_names: List of feature name strings for importance plotting.
        artifact_dir: Local directory for temporary PNG artifacts.

    Returns:
        Dict[str, Any]: Run summary with run_id, model_name, metrics.
    """
    model_name = config["name"]
    model = config["model"]
    params = config["params"]
    flavor = config["flavor"]

    logger.info(f"{'─' * 50}")
    logger.info(f"Training: {model_name}")
    logger.info(f"{'─' * 50}")

    with mlflow.start_run(run_name=model_name) as run:
        run_id = run.info.run_id
        logger.info(f"MLflow run_id: {run_id}")

        # -- Tags
        mlflow.set_tags(
            {
                "model_type": model_name,
                "dataset": "TelcoCustomerChurn",
                "framework": "scikit-learn" if flavor == "sklearn" else "xgboost",
                "task": "binary_classification",
                "target": "Churn",
            }
        )

        # -- Log hyperparameters
        mlflow.log_params(params)
        logger.info(f"Logged {len(params)} hyperparameters.")

        # -- Train model
        logger.info(f"Fitting {model_name} on {len(X_train):,} samples...")
        if flavor == "xgboost":
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_test, y_test)],
                verbose=False,
            )
        else:
            model.fit(X_train, y_train)

        # -- Predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # -- Compute and log metrics
        metrics = compute_metrics(
            y_true=y_test.values,
            y_pred=y_pred,
            y_prob=y_prob,
        )
        mlflow.log_metrics(metrics)
        logger.info("Metrics logged:")
        for metric_name, metric_val in metrics.items():
            logger.info(f"  {metric_name:<15}: {metric_val:.4f}")

        # -- Confusion matrix artifact
        cm_path = plot_confusion_matrix(
            y_true=y_test.values,
            y_pred=y_pred,
            model_name=model_name,
            output_dir=artifact_dir,
        )
        mlflow.log_artifact(str(cm_path), artifact_path="plots")

        # -- Feature importance artifact
        fi_path = plot_feature_importance(
            model=model,
            feature_names=feature_names,
            model_name=model_name,
            output_dir=artifact_dir,
            top_n=20,
        )
        if fi_path:
            mlflow.log_artifact(str(fi_path), artifact_path="plots")

        # -- Log model
        model_artifact_path = f"model_{model_name}"
        if flavor == "xgboost":
            mlflow.xgboost.log_model(
                model,
                artifact_path=model_artifact_path,
                registered_model_name=None,  # Registration handled in evaluate.py
            )
        else:
            mlflow.sklearn.log_model(
                model,
                artifact_path=model_artifact_path,
                registered_model_name=None,
            )
        logger.info(f"Model logged to MLflow artifact path: {model_artifact_path}")

        # -- Log dataset info as params
        mlflow.log_params(
            {
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "n_features": X_train.shape[1],
                "churn_rate_train": round(float(y_train.mean()), 4),
                "churn_rate_test": round(float(y_test.mean()), 4),
            }
        )

        logger.success(
            f"{model_name} training complete. "
            f"ROC-AUC={metrics['roc_auc']:.4f}, "
            f"F1={metrics['f1_score']:.4f}"
        )

        return {
            "run_id": run_id,
            "model_name": model_name,
            "model": model,
            "metrics": metrics,
            "flavor": flavor,
        }


# ---------------------------------------------------------------------------
# Main Training Function
# ---------------------------------------------------------------------------
def train_models(
    data_dir: Optional[str] = None,
    tracking_uri: Optional[str] = None,
    experiment_name: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Train all three models inside a single MLflow experiment.

    Loads preprocessed X_train/test and y_train/test CSV files, trains
    Logistic Regression, Random Forest, and XGBoost, and logs all results
    to MLflow.

    Args:
        data_dir: Directory containing X_train.csv, X_test.csv, y_train.csv,
                  y_test.csv. Defaults to 'data/processed/'.
        tracking_uri: MLflow tracking server URI. Defaults to env var
                      MLFLOW_TRACKING_URI or 'http://localhost:5000'.
        experiment_name: MLflow experiment name. Defaults to env var
                         MLFLOW_EXPERIMENT_NAME or 'ChurnShield'.

    Returns:
        List[Dict[str, Any]]: One result dict per model with run_id, metrics, etc.

    Raises:
        FileNotFoundError: If any of the required CSV files are missing.
    """
    Path("logs").mkdir(exist_ok=True)

    # -- Resolve configuration
    processed_dir = Path(data_dir or "data/processed")
    uri = tracking_uri or MLFLOW_TRACKING_URI
    exp_name = experiment_name or EXPERIMENT_NAME

    logger.info("=" * 60)
    logger.info("ChurnShield MLOps — Model Training Pipeline")
    logger.info("=" * 60)
    logger.info(f"MLflow tracking URI: {uri}")
    logger.info(f"Experiment name    : {exp_name}")
    logger.info(f"Data directory     : {processed_dir}")

    # -- Load preprocessed data
    required_files = ["X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv"]
    for fname in required_files:
        fpath = processed_dir / fname
        if not fpath.exists():
            raise FileNotFoundError(
                f"Required file not found: '{fpath}'. "
                "Run preprocessing first: python -m src.preprocessing.preprocess"
            )

    X_train = pd.read_csv(processed_dir / "X_train.csv")
    X_test = pd.read_csv(processed_dir / "X_test.csv")
    y_train = pd.read_csv(processed_dir / "y_train.csv").squeeze("columns")
    y_test = pd.read_csv(processed_dir / "y_test.csv").squeeze("columns")
    feature_names = list(X_train.columns)

    logger.info(f"X_train shape: {X_train.shape}")
    logger.info(f"X_test shape : {X_test.shape}")
    logger.info(f"Features     : {len(feature_names)}")
    logger.info(f"Churn rate   : train={y_train.mean():.3f}, test={y_test.mean():.3f}")

    # -- Load feature metadata for richer logging
    meta_path = Path("models/feature_metadata.json")
    if meta_path.exists():
        with open(meta_path) as f:
            feature_meta = json.load(f)
        logger.info(f"Feature metadata loaded from: {meta_path}")
    else:
        feature_meta = {}

    # -- Configure MLflow
    mlflow.set_tracking_uri(uri)
    experiment = mlflow.set_experiment(exp_name)
    logger.info(f"MLflow experiment ID: {experiment.experiment_id}")

    # -- Create artifact temp directory
    artifact_dir = Path(ARTIFACT_DIR)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # -- Train all models
    model_configs = get_model_configs()
    results: List[Dict[str, Any]] = []

    for config in model_configs:
        try:
            result = train_single_model(
                config=config,
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                feature_names=feature_names,
                artifact_dir=artifact_dir,
            )
            results.append(result)
        except Exception as exc:
            logger.error(
                f"Failed to train {config['name']}: {exc}",
                exc_info=True,
            )
            raise

    # -- Training Summary Table
    logger.info("=" * 60)
    logger.info("Training Summary")
    logger.info("=" * 60)
    logger.info(
        f"{'Model':<25} {'Accuracy':>10} {'Precision':>10} "
        f"{'Recall':>10} {'F1':>10} {'ROC-AUC':>10}"
    )
    logger.info("─" * 75)
    for r in results:
        m = r["metrics"]
        logger.info(
            f"{r['model_name']:<25} "
            f"{m['accuracy']:>10.4f} "
            f"{m['precision']:>10.4f} "
            f"{m['recall']:>10.4f} "
            f"{m['f1_score']:>10.4f} "
            f"{m['roc_auc']:>10.4f}"
        )

    best = max(results, key=lambda r: r["metrics"]["roc_auc"])
    logger.success(
        f"Best model by ROC-AUC: {best['model_name']} "
        f"(ROC-AUC={best['metrics']['roc_auc']:.4f})"
    )
    logger.info("=" * 60)

    # -- Save summary JSON
    summary = {
        "experiment_name": exp_name,
        "n_models_trained": len(results),
        "best_model": best["model_name"],
        "best_roc_auc": best["metrics"]["roc_auc"],
        "runs": [
            {
                "run_id": r["run_id"],
                "model_name": r["model_name"],
                "metrics": r["metrics"],
            }
            for r in results
        ],
    }
    summary_path = Path("models") / "training_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Training summary saved to: {summary_path}")

    return results


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    results = train_models()
    logger.info(f"Training complete. {len(results)} models trained and logged to MLflow.")
