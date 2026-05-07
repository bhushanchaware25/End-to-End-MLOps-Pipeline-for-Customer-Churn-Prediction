"""
ChurnShield MLOps Platform - Data Preprocessing Module
=======================================================
Builds a production-grade sklearn Pipeline with ColumnTransformer for
feature engineering on the Telco Customer Churn dataset.

Pipeline Steps:
  Numerical : SimpleImputer(median) → StandardScaler
  Categorical: SimpleImputer(most_frequent) → OneHotEncoder(sparse=False)

Outputs:
  - data/processed/X_train.csv
  - data/processed/X_test.csv
  - data/processed/y_train.csv
  - data/processed/y_test.csv
  - models/preprocessor.pkl   (fitted ColumnTransformer)

Usage:
    python -m src.preprocessing.preprocess
"""

import json
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer

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
    "logs/preprocessing.log",
    rotation="10 MB",
    retention="30 days",
    level="DEBUG",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{line} - {message}",
)

# ---------------------------------------------------------------------------
# Feature Definitions
# ---------------------------------------------------------------------------
# Drop these columns before training (identifier + target)
DROP_COLS: List[str] = ["customerID"]
TARGET_COL: str = "Churn"

# Numerical features (will be imputed + scaled)
NUMERICAL_FEATURES: List[str] = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "SeniorCitizen",
]

# Categorical features (will be imputed + one-hot encoded)
CATEGORICAL_FEATURES: List[str] = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

# Engineered feature names (added before pipeline)
ENGINEERED_FEATURES: List[str] = [
    "tenure_months_squared",
    "avg_monthly_spend",
    "has_any_addon",
    "services_count",
    "is_long_term_contract",
]


# ---------------------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------------------
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create domain-specific derived features for churn prediction.

    New Features:
        - tenure_months_squared: Captures non-linear tenure effects
        - avg_monthly_spend: TotalCharges / (tenure + 1) to avoid div-by-zero
        - has_any_addon: 1 if customer has any internet add-on service
        - services_count: Total number of active services (0–8)
        - is_long_term_contract: 1 if contract is One year or Two year

    Args:
        df: Input DataFrame with raw Telco columns.

    Returns:
        pd.DataFrame: DataFrame with engineered columns appended.
    """
    df = df.copy()

    # -- Non-linear tenure
    df["tenure_months_squared"] = df["tenure"].astype(float) ** 2

    # -- Average spend per month (avoids high TotalCharges for long-tenure)
    df["avg_monthly_spend"] = df["TotalCharges"] / (df["tenure"].astype(float) + 1)

    # -- Has any internet add-on service
    addon_cols = [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]
    df["has_any_addon"] = (
        df[addon_cols]
        .isin(["Yes"])
        .any(axis=1)
        .astype(int)
    )

    # -- Count of active services
    service_cols = [
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]
    df["services_count"] = (
        df[service_cols]
        .isin(["Yes", "DSL", "Fiber optic"])
        .sum(axis=1)
        .astype(int)
    )

    # -- Is long-term contract
    df["is_long_term_contract"] = (
        df["Contract"].isin(["One year", "Two year"]).astype(int)
    )

    logger.debug(
        f"Engineered {len(ENGINEERED_FEATURES)} new features: {ENGINEERED_FEATURES}"
    )
    return df


# ---------------------------------------------------------------------------
# Build Preprocessing Pipeline
# ---------------------------------------------------------------------------
def build_preprocessor(
    numerical_features: Optional[List[str]] = None,
    categorical_features: Optional[List[str]] = None,
) -> ColumnTransformer:
    """
    Build a sklearn ColumnTransformer preprocessing pipeline.

    Numerical pipeline:
        1. SimpleImputer(strategy='median') — handles any residual nulls
        2. StandardScaler() — zero mean, unit variance

    Categorical pipeline:
        1. SimpleImputer(strategy='most_frequent') — handles nulls
        2. OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    Args:
        numerical_features: List of numerical column names. Defaults to
                            module-level NUMERICAL_FEATURES + ENGINEERED_FEATURES.
        categorical_features: List of categorical column names. Defaults to
                              module-level CATEGORICAL_FEATURES.

    Returns:
        ColumnTransformer: Unfitted preprocessing pipeline.
    """
    num_features = numerical_features or (NUMERICAL_FEATURES + ENGINEERED_FEATURES)
    cat_features = categorical_features or CATEGORICAL_FEATURES

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ],
        memory=None,
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                    drop=None,
                ),
            ),
        ],
        memory=None,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numerical", numeric_pipeline, num_features),
            ("categorical", categorical_pipeline, cat_features),
        ],
        remainder="drop",          # Drop customerID and any unlisted cols
        verbose_feature_names_out=True,
    )

    logger.info(
        f"Preprocessor built — {len(num_features)} numerical + "
        f"{len(cat_features)} categorical features."
    )
    return preprocessor


# ---------------------------------------------------------------------------
# Encode Target
# ---------------------------------------------------------------------------
def encode_target(series: pd.Series) -> pd.Series:
    """
    Encode the Churn target column from Yes/No to 1/0.

    Args:
        series: Raw Churn column with 'Yes'/'No' string values.

    Returns:
        pd.Series: Binary encoded target (1=Churn, 0=No Churn).
    """
    encoded = series.map({"Yes": 1, "No": 0}).astype(int)
    if encoded.isna().any():
        logger.warning(
            "Some Churn values could not be mapped to 0/1. "
            "Unexpected values will be set to 0."
        )
        encoded = encoded.fillna(0).astype(int)
    return encoded


# ---------------------------------------------------------------------------
# Split Data
# ---------------------------------------------------------------------------
def split_data(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Perform a stratified train/test split preserving churn class ratio.

    Args:
        df: Full processed DataFrame (including target column).
        target_col: Name of the target column.
        test_size: Fraction of data for the test set.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    X = df.drop(columns=DROP_COLS + [target_col], errors="ignore")
    y = encode_target(df[target_col])

    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )
    train_idx, test_idx = next(splitter.split(X, y))

    X_train = X.iloc[train_idx].reset_index(drop=True)
    X_test = X.iloc[test_idx].reset_index(drop=True)
    y_train = y.iloc[train_idx].reset_index(drop=True)
    y_test = y.iloc[test_idx].reset_index(drop=True)

    logger.info(
        f"Train/test split — Train: {len(X_train):,} | Test: {len(X_test):,} | "
        f"Churn rate (train): {y_train.mean():.3f} | "
        f"Churn rate (test): {y_test.mean():.3f}"
    )
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Get Feature Names After Transformation
# ---------------------------------------------------------------------------
def get_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    """
    Extract human-readable feature names from a fitted ColumnTransformer.

    Args:
        preprocessor: A fitted ColumnTransformer instance.

    Returns:
        List[str]: Ordered list of output feature names.
    """
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception as exc:
        logger.warning(f"Could not extract feature names automatically: {exc}")
        return [f"feature_{i}" for i in range(preprocessor.transform(
            pd.DataFrame()
        ).shape[1])]


# ---------------------------------------------------------------------------
# Save Artifacts
# ---------------------------------------------------------------------------
def save_artifact(obj: object, path: Path) -> None:
    """
    Serialize and save a Python object as a pickle file.

    Args:
        obj: Any picklable Python object.
        path: Destination file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    size_kb = path.stat().st_size / 1024
    logger.info(f"Saved artifact: {path} ({size_kb:.1f} KB)")


def load_preprocessor(path: Optional[str] = None) -> ColumnTransformer:
    """
    Load a previously saved preprocessor pickle artifact.

    Args:
        path: Path to the .pkl file. Defaults to 'models/preprocessor.pkl'.

    Returns:
        ColumnTransformer: The loaded, fitted preprocessor.

    Raises:
        FileNotFoundError: If the pickle file does not exist.
    """
    pkl_path = Path(path or "models/preprocessor.pkl")
    if not pkl_path.exists():
        raise FileNotFoundError(
            f"Preprocessor not found at '{pkl_path}'. "
            "Run preprocessing first: python -m src.preprocessing.preprocess"
        )
    with open(pkl_path, "rb") as f:
        preprocessor = pickle.load(f)
    logger.info(f"Loaded preprocessor from: {pkl_path}")
    return preprocessor


# ---------------------------------------------------------------------------
# Main Preprocessing Function
# ---------------------------------------------------------------------------
def preprocess_data(
    data_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    models_dir: Optional[str] = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, object]:
    """
    Full preprocessing pipeline: load → engineer features → split →
    fit/transform → save artifacts.

    Args:
        data_path: Path to processed CSV. Falls back to env var
                   PROCESSED_DATA_PATH.
        output_dir: Directory to save train/test CSVs. Defaults to
                    'data/processed/'.
        models_dir: Directory to save the preprocessor pickle. Defaults
                    to 'models/'.
        test_size: Test set fraction (default 0.2).
        random_state: Random seed (default 42).

    Returns:
        Dict with keys: X_train, X_test, y_train, y_test,
                        preprocessor, feature_names, report.

    Raises:
        FileNotFoundError: If the input CSV does not exist.
    """
    # -- Paths
    Path("logs").mkdir(exist_ok=True)
    csv_path = Path(
        data_path or os.getenv(
            "PROCESSED_DATA_PATH", "data/processed/telco_churn_processed.csv"
        )
    )
    out_dir = Path(output_dir or "data/processed")
    mdl_dir = Path(models_dir or "models")

    logger.info("=" * 60)
    logger.info("ChurnShield MLOps — Data Preprocessing Pipeline")
    logger.info("=" * 60)
    logger.info(f"Input data  : {csv_path}")
    logger.info(f"Output dir  : {out_dir}")
    logger.info(f"Models dir  : {mdl_dir}")
    logger.info(f"Test size   : {test_size}")
    logger.info(f"Random seed : {random_state}")

    # -- Load data
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Processed data not found at '{csv_path}'. "
            "Run ingestion first: python -m src.ingestion.ingest"
        )

    df = pd.read_csv(csv_path)
    logger.info(f"Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

    # -- Feature engineering
    logger.info("Engineering domain-specific features...")
    df = engineer_features(df)

    # -- Update feature lists with engineered features
    all_numerical = NUMERICAL_FEATURES + ENGINEERED_FEATURES
    logger.info(f"Total numerical features  : {len(all_numerical)}")
    logger.info(f"Total categorical features: {len(CATEGORICAL_FEATURES)}")
    logger.info(f"Total input features      : {len(all_numerical) + len(CATEGORICAL_FEATURES)}")

    # -- Split data
    logger.info("Splitting data into train/test sets...")
    X_train, X_test, y_train, y_test = split_data(
        df, target_col=TARGET_COL, test_size=test_size, random_state=random_state
    )

    # -- Build and fit preprocessor on training data ONLY
    logger.info("Building and fitting preprocessing pipeline...")
    preprocessor = build_preprocessor(
        numerical_features=all_numerical,
        categorical_features=CATEGORICAL_FEATURES,
    )
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    # -- Get feature names
    feature_names = get_feature_names(preprocessor)
    logger.info(
        f"Transformed feature space: {len(feature_names)} features "
        f"(from {len(all_numerical) + len(CATEGORICAL_FEATURES)} input features via OHE)"
    )

    # -- Convert to DataFrames for saving
    X_train_df = pd.DataFrame(X_train_transformed, columns=feature_names)
    X_test_df = pd.DataFrame(X_test_transformed, columns=feature_names)
    y_train_df = pd.DataFrame({"Churn": y_train})
    y_test_df = pd.DataFrame({"Churn": y_test})

    # -- Save train/test CSVs
    out_dir.mkdir(parents=True, exist_ok=True)
    X_train_df.to_csv(out_dir / "X_train.csv", index=False)
    X_test_df.to_csv(out_dir / "X_test.csv", index=False)
    y_train_df.to_csv(out_dir / "y_train.csv", index=False)
    y_test_df.to_csv(out_dir / "y_test.csv", index=False)
    logger.success(f"Train/test CSV files saved to: {out_dir}")

    # -- Save preprocessor pickle
    save_artifact(preprocessor, mdl_dir / "preprocessor.pkl")

    # -- Save feature names for downstream use
    feature_meta = {
        "numerical_features": all_numerical,
        "categorical_features": CATEGORICAL_FEATURES,
        "engineered_features": ENGINEERED_FEATURES,
        "output_feature_names": feature_names,
        "n_input_features": len(all_numerical) + len(CATEGORICAL_FEATURES),
        "n_output_features": len(feature_names),
        "train_samples": len(X_train_df),
        "test_samples": len(X_test_df),
        "target_col": TARGET_COL,
        "test_size": test_size,
        "random_state": random_state,
        "churn_rate_train": float(y_train.mean()),
        "churn_rate_test": float(y_test.mean()),
    }
    meta_path = mdl_dir / "feature_metadata.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w") as f:
        json.dump(feature_meta, f, indent=2)
    logger.info(f"Feature metadata saved to: {meta_path}")

    # -- Summary
    logger.info("=" * 60)
    logger.info("Preprocessing Summary")
    logger.info("=" * 60)
    logger.info(f"  Train samples     : {len(X_train_df):,}")
    logger.info(f"  Test samples      : {len(X_test_df):,}")
    logger.info(f"  Input features    : {feature_meta['n_input_features']}")
    logger.info(f"  Output features   : {feature_meta['n_output_features']}")
    logger.info(f"  Churn rate (train): {y_train.mean():.3f}")
    logger.info(f"  Churn rate (test) : {y_test.mean():.3f}")
    logger.info("=" * 60)
    logger.success("Preprocessing complete! ✅")

    return {
        "X_train": X_train_df,
        "X_test": X_test_df,
        "y_train": y_train,
        "y_test": y_test,
        "preprocessor": preprocessor,
        "feature_names": feature_names,
        "report": feature_meta,
    }


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    result = preprocess_data()
    logger.info(
        f"Preprocessing done. "
        f"X_train shape: {result['X_train'].shape}, "
        f"X_test shape: {result['X_test'].shape}"
    )
