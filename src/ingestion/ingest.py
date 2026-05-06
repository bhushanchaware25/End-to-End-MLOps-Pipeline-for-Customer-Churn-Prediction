"""
ChurnShield MLOps Platform - Data Ingestion Module
===================================================
Handles loading raw Telco Customer Churn data (from CSV or synthetic),
performs basic cleaning, validates schema, and saves to processed directory.

Usage:
    python -m src.ingestion.ingest
"""

import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
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
    "logs/ingestion.log",
    rotation="10 MB",
    retention="30 days",
    level="DEBUG",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{line} - {message}",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EXPECTED_COLUMNS = [
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
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
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
]

CATEGORICAL_COLS = [
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
    "Churn",
]

NUMERICAL_COLS = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]


# ---------------------------------------------------------------------------
# Synthetic Data Generator
# ---------------------------------------------------------------------------
def generate_synthetic_telco_data(n_samples: int = 7043, random_seed: int = 42) -> pd.DataFrame:
    """
    Generate a synthetic Telco Customer Churn dataset that mirrors the
    schema and statistical properties of the original Kaggle dataset.

    Args:
        n_samples: Number of customer records to generate.
        random_seed: Random seed for reproducibility.

    Returns:
        pd.DataFrame: Synthetic Telco churn dataset.
    """
    logger.info(f"Generating synthetic Telco dataset with {n_samples} samples...")
    rng = np.random.default_rng(random_seed)

    # -- Customer IDs
    customer_ids = [f"CUST-{str(i).zfill(6)}" for i in range(1, n_samples + 1)]

    # -- Demographics
    gender = rng.choice(["Male", "Female"], size=n_samples, p=[0.505, 0.495])
    senior_citizen = rng.choice([0, 1], size=n_samples, p=[0.838, 0.162])
    partner = rng.choice(["Yes", "No"], size=n_samples, p=[0.483, 0.517])
    dependents = rng.choice(["Yes", "No"], size=n_samples, p=[0.299, 0.701])

    # -- Tenure (months) — bimodal: new customers and long-term
    tenure_new = rng.integers(0, 6, size=n_samples // 3)
    tenure_old = rng.integers(24, 72, size=n_samples - n_samples // 3)
    tenure = np.concatenate([tenure_new, tenure_old])
    rng.shuffle(tenure)
    tenure = tenure.astype(int)

    # -- Phone Service
    phone_service = rng.choice(["Yes", "No"], size=n_samples, p=[0.904, 0.096])
    multiple_lines = np.where(
        phone_service == "No",
        "No phone service",
        rng.choice(["Yes", "No"], size=n_samples, p=[0.421, 0.579]),
    )

    # -- Internet Service
    internet_service = rng.choice(
        ["DSL", "Fiber optic", "No"], size=n_samples, p=[0.344, 0.44, 0.216]
    )

    def internet_addon(has_internet: np.ndarray, yes_prob: float = 0.3) -> np.ndarray:
        """Generate internet add-on field based on internet service."""
        result = np.where(
            has_internet == "No",
            "No internet service",
            rng.choice(["Yes", "No"], size=n_samples, p=[yes_prob, 1 - yes_prob]),
        )
        return result

    online_security = internet_addon(internet_service, 0.286)
    online_backup = internet_addon(internet_service, 0.344)
    device_protection = internet_addon(internet_service, 0.343)
    tech_support = internet_addon(internet_service, 0.289)
    streaming_tv = internet_addon(internet_service, 0.384)
    streaming_movies = internet_addon(internet_service, 0.389)

    # -- Contract & Billing
    contract = rng.choice(
        ["Month-to-month", "One year", "Two year"],
        size=n_samples,
        p=[0.551, 0.209, 0.240],
    )
    paperless_billing = rng.choice(["Yes", "No"], size=n_samples, p=[0.592, 0.408])
    payment_method = rng.choice(
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
        size=n_samples,
        p=[0.337, 0.228, 0.219, 0.216],
    )

    # -- Charges
    monthly_charges = np.round(rng.uniform(18.0, 118.75, size=n_samples), 2)
    # TotalCharges correlated with tenure and monthly charges
    total_charges = np.round(tenure * monthly_charges + rng.normal(0, 50, n_samples), 2)
    total_charges = np.clip(total_charges, 18.8, 8684.8)

    # -- Churn label (influenced by contract, tenure, internet service)
    churn_prob = np.where(contract == "Month-to-month", 0.42, 0.11)
    churn_prob = np.where(internet_service == "Fiber optic", churn_prob + 0.10, churn_prob)
    churn_prob = np.where(tenure < 12, churn_prob + 0.15, churn_prob - 0.05)
    churn_prob = np.clip(churn_prob, 0.02, 0.95)
    churn_raw = rng.random(size=n_samples) < churn_prob
    churn = np.where(churn_raw, "Yes", "No")

    # -- Introduce ~11 missing TotalCharges (mirrors real dataset)
    missing_indices = rng.choice(n_samples, size=11, replace=False)
    total_charges_str = total_charges.astype(str)
    total_charges_str[missing_indices] = " "

    df = pd.DataFrame(
        {
            "customerID": customer_ids,
            "gender": gender,
            "SeniorCitizen": senior_citizen,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges_str,
            "Churn": churn,
        }
    )

    logger.success(f"Synthetic dataset generated: {df.shape[0]} rows × {df.shape[1]} columns")
    return df


# ---------------------------------------------------------------------------
# Schema Validator
# ---------------------------------------------------------------------------
def validate_schema(df: pd.DataFrame) -> bool:
    """
    Validate that the DataFrame has all expected columns.

    Args:
        df: Input DataFrame to validate.

    Returns:
        bool: True if schema is valid.

    Raises:
        ValueError: If required columns are missing.
    """
    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Schema validation failed. Missing columns: {missing}")
    logger.info("Schema validation passed — all expected columns present.")
    return True


# ---------------------------------------------------------------------------
# Cleaner
# ---------------------------------------------------------------------------
def clean_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """
    Perform basic data cleaning on the raw Telco dataset.

    Steps:
        1. Convert TotalCharges from string to float (handles whitespace).
        2. Fill missing TotalCharges with tenure × MonthlyCharges estimate.
        3. Strip whitespace from string columns.
        4. Standardize Churn to Yes/No.
        5. Drop duplicates by customerID.

    Args:
        df: Raw input DataFrame.

    Returns:
        Tuple[pd.DataFrame, dict]: Cleaned DataFrame and cleaning report dict.
    """
    report: dict = {}
    original_shape = df.shape
    logger.info(f"Starting data cleaning. Input shape: {original_shape}")

    # -- Step 1: Convert TotalCharges to numeric
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    n_missing_tc = df["TotalCharges"].isna().sum()
    report["missing_total_charges_before"] = int(n_missing_tc)
    logger.debug(f"TotalCharges: {n_missing_tc} missing values found after coercion.")

    # -- Step 2: Impute missing TotalCharges
    mask = df["TotalCharges"].isna()
    df.loc[mask, "TotalCharges"] = df.loc[mask, "tenure"] * df.loc[mask, "MonthlyCharges"]
    report["missing_total_charges_imputed"] = int(mask.sum())
    logger.info(f"Imputed {mask.sum()} missing TotalCharges values using tenure × MonthlyCharges.")

    # -- Step 3: Strip whitespace from all string columns
    str_cols = df.select_dtypes(include="object").columns
    for col in str_cols:
        df[col] = df[col].str.strip()
    logger.debug(f"Stripped whitespace from {len(str_cols)} string columns.")

    # -- Step 4: Standardize Churn column
    df["Churn"] = df["Churn"].map({"Yes": "Yes", "No": "No", "1": "Yes", "0": "No"}).fillna("No")
    churn_dist = df["Churn"].value_counts().to_dict()
    report["churn_distribution"] = churn_dist
    logger.info(f"Churn distribution: {churn_dist}")

    # -- Step 5: Drop duplicates
    n_before = len(df)
    df = df.drop_duplicates(subset=["customerID"]).reset_index(drop=True)
    n_dupes = n_before - len(df)
    report["duplicates_removed"] = n_dupes
    if n_dupes > 0:
        logger.warning(f"Removed {n_dupes} duplicate customerID rows.")
    else:
        logger.info("No duplicate customerIDs found.")

    # -- Step 6: Ensure correct dtypes
    df["SeniorCitizen"] = df["SeniorCitizen"].astype(int)
    df["tenure"] = df["tenure"].astype(int)
    df["MonthlyCharges"] = df["MonthlyCharges"].astype(float)
    df["TotalCharges"] = df["TotalCharges"].astype(float)

    report["final_shape"] = df.shape
    report["null_counts"] = df.isnull().sum().to_dict()

    logger.success(
        f"Data cleaning complete. Shape: {original_shape} → {df.shape}. "
        f"Nulls remaining: {df.isnull().sum().sum()}"
    )
    return df, report


# ---------------------------------------------------------------------------
# Main Ingestion Function
# ---------------------------------------------------------------------------
def ingest_data(
    raw_data_path: Optional[str] = None,
    output_path: Optional[str] = None,
    generate_synthetic: bool = False,
) -> pd.DataFrame:
    """
    Main data ingestion entry point. Loads or generates raw Telco data,
    validates schema, cleans it, and saves processed output.

    Args:
        raw_data_path: Path to raw CSV file. Reads from env var RAW_DATA_PATH
                       if not provided.
        output_path: Path to save processed CSV. Reads from env var
                     PROCESSED_DATA_PATH if not provided.
        generate_synthetic: Force synthetic data generation even if CSV exists.

    Returns:
        pd.DataFrame: Cleaned, validated DataFrame ready for feature engineering.

    Raises:
        FileNotFoundError: If raw_data_path is given but file doesn't exist.
        ValueError: If schema validation fails.
    """
    # -- Resolve paths from env or args
    raw_path = Path(raw_data_path or os.getenv("RAW_DATA_PATH", "data/raw/telco_churn.csv"))
    out_path = Path(
        output_path or os.getenv("PROCESSED_DATA_PATH", "data/processed/telco_churn_processed.csv")
    )
    ref_path = Path(os.getenv("REFERENCE_DATA_PATH", "data/reference/reference_dataset.csv"))

    # -- Ensure output directories exist
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ref_path.parent.mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

    logger.info("=" * 60)
    logger.info("ChurnShield MLOps — Data Ingestion Pipeline")
    logger.info("=" * 60)

    # -- Load or generate data
    if generate_synthetic or not raw_path.exists():
        if not generate_synthetic:
            logger.warning(
                f"Raw data file not found at '{raw_path}'. "
                "Falling back to synthetic data generation."
            )
        df_raw = generate_synthetic_telco_data()
        # Save synthetic raw data
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        df_raw.to_csv(raw_path, index=False)
        logger.info(f"Synthetic raw data saved to: {raw_path}")
    else:
        logger.info(f"Loading raw data from: {raw_path}")
        df_raw = pd.read_csv(raw_path)
        logger.info(f"Loaded {len(df_raw):,} rows × {len(df_raw.columns)} columns")

    # -- Log raw data profile
    logger.info(f"Raw data shape: {df_raw.shape}")
    logger.info(f"Column dtypes:\n{df_raw.dtypes.to_string()}")
    logger.info(f"Null counts:\n{df_raw.isnull().sum().to_string()}")

    # -- Validate schema
    validate_schema(df_raw)

    # -- Clean data
    df_clean, cleaning_report = clean_data(df_raw)

    # -- Log cleaning report
    logger.info("Cleaning Report:")
    for key, value in cleaning_report.items():
        logger.info(f"  {key}: {value}")

    # -- Save processed data
    df_clean.to_csv(out_path, index=False)
    logger.success(f"Processed data saved to: {out_path}")

    # -- Save reference dataset (first 20% of clean data for drift detection)
    n_ref = max(1000, len(df_clean) // 5)
    df_ref = df_clean.iloc[:n_ref].copy()
    df_ref.to_csv(ref_path, index=False)
    logger.success(f"Reference dataset saved to: {ref_path} ({len(df_ref):,} rows)")

    # -- Final summary
    churn_rate = (df_clean["Churn"] == "Yes").mean() * 100
    logger.info("=" * 60)
    logger.info("Ingestion Summary")
    logger.info("=" * 60)
    logger.info(f"  Total customers : {len(df_clean):,}")
    logger.info(f"  Features        : {len(df_clean.columns) - 2}")  # excl. customerID, Churn
    logger.info(f"  Churn rate      : {churn_rate:.1f}%")
    logger.info(f"  Output path     : {out_path}")
    logger.info(f"  Reference path  : {ref_path}")
    logger.info("=" * 60)

    return df_clean


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    df = ingest_data()
    logger.info(f"Ingestion complete. DataFrame head:\n{df.head(3).to_string()}")
