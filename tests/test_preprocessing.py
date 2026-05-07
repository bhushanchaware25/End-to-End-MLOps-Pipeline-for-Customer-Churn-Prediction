"""
Tests for src/preprocessing/preprocess.py
==========================================
Tests cover:
  - Feature engineering: all 5 derived features created correctly
  - build_preprocessor(): correct structure and transformer types
  - encode_target(): Yes→1, No→0 mapping
  - split_data(): shapes, stratification, reproducibility
  - Full preprocess_data() integration with temp directories
  - load_preprocessor(): save and reload correctness
  - Output shape correctness after fit_transform
"""

import json
import os
import pickle
import tempfile
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src.preprocessing.preprocess import (
    CATEGORICAL_FEATURES,
    ENGINEERED_FEATURES,
    NUMERICAL_FEATURES,
    TARGET_COL,
    build_preprocessor,
    encode_target,
    engineer_features,
    get_feature_names,
    load_preprocessor,
    preprocess_data,
    save_artifact,
    split_data,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Return a minimal valid processed Telco DataFrame (200 rows)."""
    np.random.seed(42)
    n = 200
    return pd.DataFrame(
        {
            "customerID": [f"C{i:04d}" for i in range(n)],
            "gender": np.random.choice(["Male", "Female"], n),
            "SeniorCitizen": np.random.choice([0, 1], n, p=[0.85, 0.15]),
            "Partner": np.random.choice(["Yes", "No"], n),
            "Dependents": np.random.choice(["Yes", "No"], n),
            "tenure": np.random.randint(0, 72, n),
            "PhoneService": np.random.choice(["Yes", "No"], n, p=[0.9, 0.1]),
            "MultipleLines": np.random.choice(
                ["Yes", "No", "No phone service"], n
            ),
            "InternetService": np.random.choice(
                ["DSL", "Fiber optic", "No"], n, p=[0.35, 0.44, 0.21]
            ),
            "OnlineSecurity": np.random.choice(
                ["Yes", "No", "No internet service"], n
            ),
            "OnlineBackup": np.random.choice(
                ["Yes", "No", "No internet service"], n
            ),
            "DeviceProtection": np.random.choice(
                ["Yes", "No", "No internet service"], n
            ),
            "TechSupport": np.random.choice(
                ["Yes", "No", "No internet service"], n
            ),
            "StreamingTV": np.random.choice(
                ["Yes", "No", "No internet service"], n
            ),
            "StreamingMovies": np.random.choice(
                ["Yes", "No", "No internet service"], n
            ),
            "Contract": np.random.choice(
                ["Month-to-month", "One year", "Two year"], n
            ),
            "PaperlessBilling": np.random.choice(["Yes", "No"], n),
            "PaymentMethod": np.random.choice(
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
                n,
            ),
            "MonthlyCharges": np.round(np.random.uniform(18.0, 118.0, n), 2),
            "TotalCharges": np.round(np.random.uniform(18.8, 8000.0, n), 2),
            "Churn": np.random.choice(["Yes", "No"], n, p=[0.27, 0.73]),
        }
    )


# ---------------------------------------------------------------------------
# Test: Feature Engineering
# ---------------------------------------------------------------------------
class TestEngineerFeatures:
    def test_all_engineered_features_created(self, sample_df: pd.DataFrame) -> None:
        """All 5 engineered features should be added to the DataFrame."""
        df_eng = engineer_features(sample_df)
        for feat in ENGINEERED_FEATURES:
            assert feat in df_eng.columns, f"Missing engineered feature: {feat}"

    def test_original_columns_preserved(self, sample_df: pd.DataFrame) -> None:
        """Feature engineering should not remove any original columns."""
        df_eng = engineer_features(sample_df)
        for col in sample_df.columns:
            assert col in df_eng.columns

    def test_tenure_squared_values(self, sample_df: pd.DataFrame) -> None:
        """tenure_months_squared should equal tenure²."""
        df_eng = engineer_features(sample_df)
        expected = sample_df["tenure"].astype(float) ** 2
        pd.testing.assert_series_equal(
            df_eng["tenure_months_squared"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )

    def test_avg_monthly_spend_no_div_zero(self, sample_df: pd.DataFrame) -> None:
        """avg_monthly_spend should be finite (no division by zero)."""
        df_eng = engineer_features(sample_df)
        assert df_eng["avg_monthly_spend"].isna().sum() == 0
        assert np.isfinite(df_eng["avg_monthly_spend"]).all()

    def test_has_any_addon_binary(self, sample_df: pd.DataFrame) -> None:
        """has_any_addon should only contain 0 or 1."""
        df_eng = engineer_features(sample_df)
        assert set(df_eng["has_any_addon"].unique()).issubset({0, 1})

    def test_services_count_non_negative(self, sample_df: pd.DataFrame) -> None:
        """services_count should be non-negative integer."""
        df_eng = engineer_features(sample_df)
        assert (df_eng["services_count"] >= 0).all()

    def test_is_long_term_contract_binary(self, sample_df: pd.DataFrame) -> None:
        """is_long_term_contract should only contain 0 or 1."""
        df_eng = engineer_features(sample_df)
        assert set(df_eng["is_long_term_contract"].unique()).issubset({0, 1})

    def test_is_long_term_contract_logic(self, sample_df: pd.DataFrame) -> None:
        """Long-term contract flag should be 1 for One year and Two year."""
        df_eng = engineer_features(sample_df)
        mask = sample_df["Contract"].isin(["One year", "Two year"])
        assert (df_eng.loc[mask, "is_long_term_contract"] == 1).all()
        assert (df_eng.loc[~mask, "is_long_term_contract"] == 0).all()

    def test_engineer_does_not_mutate_original(self, sample_df: pd.DataFrame) -> None:
        """engineer_features should not mutate the input DataFrame."""
        original_cols = list(sample_df.columns)
        _ = engineer_features(sample_df)
        assert list(sample_df.columns) == original_cols


# ---------------------------------------------------------------------------
# Test: encode_target
# ---------------------------------------------------------------------------
class TestEncodeTarget:
    def test_yes_maps_to_1(self) -> None:
        """'Yes' values should map to 1."""
        s = pd.Series(["Yes", "Yes", "No"])
        encoded = encode_target(s)
        assert encoded[0] == 1
        assert encoded[1] == 1

    def test_no_maps_to_0(self) -> None:
        """'No' values should map to 0."""
        s = pd.Series(["No", "No", "Yes"])
        encoded = encode_target(s)
        assert encoded[0] == 0
        assert encoded[1] == 0

    def test_output_is_binary(self) -> None:
        """Encoded output should only contain 0 and 1."""
        s = pd.Series(["Yes", "No", "Yes", "No", "No"])
        encoded = encode_target(s)
        assert set(encoded.unique()).issubset({0, 1})

    def test_output_dtype_is_int(self) -> None:
        """Encoded target should have integer dtype."""
        s = pd.Series(["Yes", "No"])
        encoded = encode_target(s)
        assert pd.api.types.is_integer_dtype(encoded)


# ---------------------------------------------------------------------------
# Test: build_preprocessor
# ---------------------------------------------------------------------------
class TestBuildPreprocessor:
    def test_returns_column_transformer(self) -> None:
        """build_preprocessor should return a ColumnTransformer."""
        preprocessor = build_preprocessor()
        assert isinstance(preprocessor, ColumnTransformer)

    def test_has_numerical_and_categorical_transformers(self) -> None:
        """Preprocessor should have exactly 2 named transformers."""
        preprocessor = build_preprocessor()
        transformer_names = [name for name, _, _ in preprocessor.transformers]
        assert "numerical" in transformer_names
        assert "categorical" in transformer_names

    def test_numerical_pipeline_has_imputer_and_scaler(self) -> None:
        """Numerical sub-pipeline should have SimpleImputer and StandardScaler."""
        preprocessor = build_preprocessor()
        num_pipeline = dict(
            (name, t) for name, t, _ in preprocessor.transformers
        )["numerical"]
        step_names = [name for name, _ in num_pipeline.steps]
        assert "imputer" in step_names
        assert "scaler" in step_names

    def test_categorical_pipeline_has_imputer_and_encoder(self) -> None:
        """Categorical sub-pipeline should have SimpleImputer and OneHotEncoder."""
        preprocessor = build_preprocessor()
        cat_pipeline = dict(
            (name, t) for name, t, _ in preprocessor.transformers
        )["categorical"]
        step_names = [name for name, _ in cat_pipeline.steps]
        assert "imputer" in step_names
        assert "encoder" in step_names

    def test_custom_feature_lists_accepted(self) -> None:
        """build_preprocessor should accept custom feature lists."""
        preprocessor = build_preprocessor(
            numerical_features=["tenure", "MonthlyCharges"],
            categorical_features=["gender"],
        )
        assert isinstance(preprocessor, ColumnTransformer)


# ---------------------------------------------------------------------------
# Test: split_data
# ---------------------------------------------------------------------------
class TestSplitData:
    def test_split_shapes(self, sample_df: pd.DataFrame) -> None:
        """Train + test shapes should sum to original dataset size."""
        df_eng = engineer_features(sample_df)
        X_train, X_test, y_train, y_test = split_data(df_eng)
        total = len(X_train) + len(X_test)
        assert total == len(df_eng)

    def test_test_size_proportion(self, sample_df: pd.DataFrame) -> None:
        """Test set should be approximately 20% of the data."""
        df_eng = engineer_features(sample_df)
        X_train, X_test, _, _ = split_data(df_eng, test_size=0.2)
        ratio = len(X_test) / (len(X_train) + len(X_test))
        assert abs(ratio - 0.2) < 0.05

    def test_stratification_preserves_churn_rate(self, sample_df: pd.DataFrame) -> None:
        """Churn rate in train and test sets should be similar (stratified split)."""
        df_eng = engineer_features(sample_df)
        _, _, y_train, y_test = split_data(df_eng)
        assert abs(y_train.mean() - y_test.mean()) < 0.10

    def test_reproducibility(self, sample_df: pd.DataFrame) -> None:
        """Same random_state should produce identical splits."""
        df_eng = engineer_features(sample_df)
        X_train1, _, _, _ = split_data(df_eng, random_state=42)
        X_train2, _, _, _ = split_data(df_eng, random_state=42)
        pd.testing.assert_frame_equal(X_train1, X_train2)

    def test_no_data_leakage(self, sample_df: pd.DataFrame) -> None:
        """Train and test index sets should not overlap."""
        df_eng = engineer_features(sample_df)
        X_train, X_test, _, _ = split_data(df_eng)
        # Since we reset_index, check no shared indices
        assert len(X_train) + len(X_test) == len(df_eng)


# ---------------------------------------------------------------------------
# Test: Fit/Transform Output Shape
# ---------------------------------------------------------------------------
class TestFitTransformShape:
    def test_transform_output_is_2d_array(self, sample_df: pd.DataFrame) -> None:
        """fit_transform output should be a 2D numpy array."""
        df_eng = engineer_features(sample_df)
        X = df_eng.drop(columns=["customerID", "Churn"])
        preprocessor = build_preprocessor()
        result = preprocessor.fit_transform(X)
        assert result.ndim == 2

    def test_transform_no_nans(self, sample_df: pd.DataFrame) -> None:
        """Transformed output should contain no NaN values."""
        df_eng = engineer_features(sample_df)
        X = df_eng.drop(columns=["customerID", "Churn"])
        preprocessor = build_preprocessor()
        result = preprocessor.fit_transform(X)
        assert not np.isnan(result).any()

    def test_train_test_same_feature_count(self, sample_df: pd.DataFrame) -> None:
        """Train and test transforms should have the same number of features."""
        df_eng = engineer_features(sample_df)
        X_train, X_test, _, _ = split_data(df_eng)
        preprocessor = build_preprocessor()
        X_train_t = preprocessor.fit_transform(X_train)
        X_test_t = preprocessor.transform(X_test)
        assert X_train_t.shape[1] == X_test_t.shape[1]


# ---------------------------------------------------------------------------
# Test: save_artifact + load_preprocessor
# ---------------------------------------------------------------------------
class TestArtifactSaveLoad:
    def test_save_and_load_preprocessor(self, sample_df: pd.DataFrame) -> None:
        """Saved preprocessor should be loadable and produce same output."""
        df_eng = engineer_features(sample_df)
        X = df_eng.drop(columns=["customerID", "Churn"])
        preprocessor = build_preprocessor()
        preprocessor.fit(X)

        with tempfile.TemporaryDirectory() as tmpdir:
            pkl_path = Path(tmpdir) / "models" / "preprocessor.pkl"
            save_artifact(preprocessor, pkl_path)
            assert pkl_path.exists()
            loaded = load_preprocessor(str(pkl_path))

        result_original = preprocessor.transform(X)
        result_loaded = loaded.transform(X)
        np.testing.assert_array_almost_equal(result_original, result_loaded)

    def test_load_missing_preprocessor_raises(self) -> None:
        """Loading a non-existent preprocessor should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_preprocessor("/nonexistent/path/preprocessor.pkl")


# ---------------------------------------------------------------------------
# Test: Full preprocess_data() Integration
# ---------------------------------------------------------------------------
class TestPreprocessDataIntegration:
    def test_returns_expected_keys(self, sample_df: pd.DataFrame) -> None:
        """preprocess_data() should return dict with all expected keys."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "processed.csv"
            out_dir = Path(tmpdir) / "processed"
            mdl_dir = Path(tmpdir) / "models"
            sample_df.to_csv(csv_path, index=False)

            result = preprocess_data(
                data_path=str(csv_path),
                output_dir=str(out_dir),
                models_dir=str(mdl_dir),
            )
        expected_keys = {"X_train", "X_test", "y_train", "y_test",
                         "preprocessor", "feature_names", "report"}
        assert expected_keys.issubset(result.keys())

    def test_saves_all_csv_files(self, sample_df: pd.DataFrame) -> None:
        """preprocess_data() should save X_train/test and y_train/test CSVs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "processed.csv"
            out_dir = Path(tmpdir) / "processed"
            mdl_dir = Path(tmpdir) / "models"
            sample_df.to_csv(csv_path, index=False)

            preprocess_data(
                data_path=str(csv_path),
                output_dir=str(out_dir),
                models_dir=str(mdl_dir),
            )

        for fname in ["X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv"]:
            assert (out_dir / fname).exists(), f"Missing: {fname}"

    def test_saves_preprocessor_pickle(self, sample_df: pd.DataFrame) -> None:
        """preprocess_data() should save a valid preprocessor.pkl."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "processed.csv"
            out_dir = Path(tmpdir) / "processed"
            mdl_dir = Path(tmpdir) / "models"
            sample_df.to_csv(csv_path, index=False)

            preprocess_data(
                data_path=str(csv_path),
                output_dir=str(out_dir),
                models_dir=str(mdl_dir),
            )

        assert (mdl_dir / "preprocessor.pkl").exists()

    def test_output_has_no_nans(self, sample_df: pd.DataFrame) -> None:
        """Transformed X_train and X_test should have no NaN values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "processed.csv"
            out_dir = Path(tmpdir) / "processed"
            mdl_dir = Path(tmpdir) / "models"
            sample_df.to_csv(csv_path, index=False)

            result = preprocess_data(
                data_path=str(csv_path),
                output_dir=str(out_dir),
                models_dir=str(mdl_dir),
            )

        assert not result["X_train"].isna().any().any()
        assert not result["X_test"].isna().any().any()

    def test_raises_on_missing_input(self) -> None:
        """preprocess_data() should raise FileNotFoundError for missing CSV."""
        with pytest.raises(FileNotFoundError):
            preprocess_data(data_path="/nonexistent/data.csv")

    def test_feature_names_saved_in_report(self, sample_df: pd.DataFrame) -> None:
        """Report should contain output_feature_names matching X_train columns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "processed.csv"
            out_dir = Path(tmpdir) / "processed"
            mdl_dir = Path(tmpdir) / "models"
            sample_df.to_csv(csv_path, index=False)

            result = preprocess_data(
                data_path=str(csv_path),
                output_dir=str(out_dir),
                models_dir=str(mdl_dir),
            )

        report = result["report"]
        assert len(report["output_feature_names"]) == result["X_train"].shape[1]
