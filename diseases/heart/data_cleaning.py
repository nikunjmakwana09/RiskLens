from __future__ import annotations

import os
import json
import logging
from datetime import datetime
from typing import Any, Dict, List

import numpy as np
import pandas as pd


# ---------------------------------------------------
# Logging Configuration
# ---------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------
# Paths
# ---------------------------------------------------

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "heart.csv")
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "heart_clean.csv")

MODEL_REGISTRY_DIR = os.path.join(BASE_DIR, "model_registry", "heart", "data_clean")
SCHEMA_PATH = os.path.join(MODEL_REGISTRY_DIR, "schema.json")
FEATURE_INFO_PATH = os.path.join(MODEL_REGISTRY_DIR, "feature_info.json")
DATA_CLEANING_REPORT_PATH = os.path.join(MODEL_REGISTRY_DIR, "data_cleaning_report.json")


# ---------------------------------------------------
# Constants
# ---------------------------------------------------

DATASET_NAME = "heart"
TARGET_COLUMN = "target"

EXPECTED_COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "target",
]

CATEGORICAL_COLUMNS = [
    "sex",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "ca",
    "thal",
]

NUMERIC_COLUMNS = [
    "age",
    "trestbps",
    "chol",
    "thalach",
    "oldpeak",
]

INVALID_ZERO_COLUMNS = [
    "trestbps",
    "chol",
    "thalach",
]

COMMON_MISSING_MARKERS = ["", " ", "NA", "N/A", "null", "NULL", "?"]


# ---------------------------------------------------
# Metadata Tracker
# ---------------------------------------------------

def initialize_cleaning_report() -> Dict[str, Any]:
    return {
        "dataset_name": DATASET_NAME,
        "raw_data_path": RAW_DATA_PATH,
        "processed_data_path": PROCESSED_DATA_PATH,
        "target_column": TARGET_COLUMN,
        "pipeline_started_at": datetime.utcnow().isoformat() + "Z",
        "pipeline_completed_at": None,
        "initial_shape": None,
        "final_shape": None,
        "columns": [],
        "schema_validation": {
            "missing_columns": [],
            "extra_columns": [],
        },
        "steps": {
            "duplicates_removed": 0,
            "missing_markers_standardized": False,
            "invalid_zero_replacements": {},
            "datatype_enforcements": {},
            "missing_imputation": {},
        },
        "validation": {
            "target_unique_values": [],
            "missing_values_remaining": None,
            "negative_values_detected": None,
            "target_column_present": None,
            "categorical_range_warnings": {},
            "numerical_range_warnings": {},
        },
        "summary": {},
    }


def to_serializable(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if np.isnan(value):
            return None
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if pd.isna(value):
        return None
    return value


# ---------------------------------------------------
# Load Dataset
# ---------------------------------------------------

def load_raw_data(report: Dict[str, Any]) -> pd.DataFrame:
    logger.info("Loading heart disease dataset...")

    if not os.path.exists(RAW_DATA_PATH):
        raise FileNotFoundError(f"Dataset not found: {RAW_DATA_PATH}")

    df = pd.read_csv(RAW_DATA_PATH)

    logger.info("Dataset loaded successfully")
    logger.info(f"Initial Shape: {df.shape}")

    report["initial_shape"] = {"rows": int(df.shape[0]), "columns": int(df.shape[1])}
    report["columns"] = list(df.columns)

    return df


# ---------------------------------------------------
# Schema Validation
# ---------------------------------------------------

def validate_schema(df: pd.DataFrame, report: Dict[str, Any]) -> None:
    logger.info("Validating dataset schema...")

    missing_cols = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    extra_cols = [col for col in df.columns if col not in EXPECTED_COLUMNS]

    report["schema_validation"]["missing_columns"] = missing_cols
    report["schema_validation"]["extra_columns"] = extra_cols

    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    if extra_cols:
        logger.warning(f"Unexpected extra columns detected: {extra_cols}")

    logger.info("Schema validation successful")


# ---------------------------------------------------
# Remove Duplicates
# ---------------------------------------------------

def remove_duplicates(df: pd.DataFrame, report: Dict[str, Any]) -> pd.DataFrame:
    duplicate_count = int(df.duplicated().sum())

    logger.info(f"Duplicate rows detected: {duplicate_count}")

    if duplicate_count > 0:
        df = df.drop_duplicates().reset_index(drop=True)
        logger.info("Duplicate rows removed")

    report["steps"]["duplicates_removed"] = duplicate_count
    return df


# ---------------------------------------------------
# Standardize Missing Markers
# ---------------------------------------------------

def standardize_missing_markers(df: pd.DataFrame, report: Dict[str, Any]) -> pd.DataFrame:
    logger.info("Standardizing missing value markers...")
    df = df.replace(COMMON_MISSING_MARKERS, np.nan)
    report["steps"]["missing_markers_standardized"] = True
    return df


# ---------------------------------------------------
# Data Type Standardization
# ---------------------------------------------------

def enforce_data_types(df: pd.DataFrame, report: Dict[str, Any]) -> pd.DataFrame:
    logger.info("Standardizing feature data types...")

    dtype_report = {}

    for col in NUMERIC_COLUMNS:
        before_dtype = str(df[col].dtype)
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
        dtype_report[col] = {"before": before_dtype, "after": str(df[col].dtype)}

    for col in CATEGORICAL_COLUMNS:
        before_dtype = str(df[col].dtype)
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
        dtype_report[col] = {"before": before_dtype, "after": str(df[col].dtype)}

    before_target = str(df[TARGET_COLUMN].dtype)
    df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors="coerce").astype("Int64")
    dtype_report[TARGET_COLUMN] = {"before": before_target, "after": str(df[TARGET_COLUMN].dtype)}

    report["steps"]["datatype_enforcements"] = dtype_report

    logger.info("Data type standardization completed")
    return df


# ---------------------------------------------------
# Handle Invalid Medical Values
# ---------------------------------------------------

def handle_invalid_values(df: pd.DataFrame, report: Dict[str, Any]) -> pd.DataFrame:
    logger.info("Checking medically invalid values...")

    for col in INVALID_ZERO_COLUMNS:
        zero_count = int((df[col] == 0).sum())
        logger.info(f"{col}: {zero_count} invalid zero values")

        if zero_count > 0:
            valid_series = df.loc[df[col] != 0, col]
            median_val = valid_series.median()

            if pd.isna(median_val):
                raise ValueError(f"Cannot compute valid non-zero median for column: {col}")

            df.loc[df[col] == 0, col] = median_val

            report["steps"]["invalid_zero_replacements"][col] = {
                "replaced_count": zero_count,
                "replacement_strategy": "non_zero_median",
                "median_used": to_serializable(median_val),
            }
        else:
            report["steps"]["invalid_zero_replacements"][col] = {
                "replaced_count": 0,
                "replacement_strategy": "non_zero_median",
                "median_used": None,
            }

    return df


# ---------------------------------------------------
# Missing Value Handling
# ---------------------------------------------------

def impute_missing_values(df: pd.DataFrame, report: Dict[str, Any]) -> pd.DataFrame:
    logger.info("Handling missing values...")

    for col in NUMERIC_COLUMNS:
        missing_count = int(df[col].isnull().sum())
        if missing_count > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            report["steps"]["missing_imputation"][col] = {
                "strategy": "median",
                "imputed_count": missing_count,
                "fill_value": to_serializable(median_val),
            }
        else:
            report["steps"]["missing_imputation"][col] = {
                "strategy": "median",
                "imputed_count": 0,
                "fill_value": None,
            }

    for col in CATEGORICAL_COLUMNS:
        missing_count = int(df[col].isnull().sum())
        if missing_count > 0:
            mode_series = df[col].mode(dropna=True)
            if mode_series.empty:
                raise ValueError(f"Cannot compute mode for categorical column: {col}")
            mode_val = mode_series.iloc[0]
            df[col] = df[col].fillna(mode_val)
            report["steps"]["missing_imputation"][col] = {
                "strategy": "mode",
                "imputed_count": missing_count,
                "fill_value": to_serializable(mode_val),
            }
        else:
            report["steps"]["missing_imputation"][col] = {
                "strategy": "mode",
                "imputed_count": 0,
                "fill_value": None,
            }

    if df[TARGET_COLUMN].isnull().sum() > 0:
        raise ValueError("Missing values found in target column.")

    return df


# ---------------------------------------------------
# Target Validation
# ---------------------------------------------------

def validate_target(df: pd.DataFrame, report: Dict[str, Any]) -> pd.DataFrame:
    logger.info("Validating target column...")

    unique_values = sorted(df[TARGET_COLUMN].dropna().unique().tolist())
    report["validation"]["target_unique_values"] = [to_serializable(v) for v in unique_values]

    if unique_values != [0, 1]:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' must contain binary values [0, 1], found: {unique_values}"
        )

    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype("int64")
    return df


# ---------------------------------------------------
# Domain Validation
# ---------------------------------------------------

def validate_domain_ranges(df: pd.DataFrame, report: Dict[str, Any]) -> None:
    logger.info("Performing domain validation...")

    categorical_warnings = {
        "sex_not_in_{0,1}": int((~df["sex"].isin([0, 1])).sum()),
        "cp_outside_0_3": int((~df["cp"].isin([0, 1, 2, 3])).sum()),
        "fbs_not_in_{0,1}": int((~df["fbs"].isin([0, 1])).sum()),
        "restecg_outside_0_2": int((~df["restecg"].isin([0, 1, 2])).sum()),
        "exang_not_in_{0,1}": int((~df["exang"].isin([0, 1])).sum()),
        "slope_outside_0_2": int((~df["slope"].isin([0, 1, 2])).sum()),
        "ca_outside_0_4": int((~df["ca"].isin([0, 1, 2, 3, 4])).sum()),
        "thal_outside_known_set": int((~df["thal"].isin([0, 1, 2, 3])).sum()),
    }

    numerical_warnings = {
        "age <= 0": int((df["age"] <= 0).sum()),
        "trestbps <= 0": int((df["trestbps"] <= 0).sum()),
        "chol <= 0": int((df["chol"] <= 0).sum()),
        "thalach <= 0": int((df["thalach"] <= 0).sum()),
        "oldpeak < 0": int((df["oldpeak"] < 0).sum()),
        "age > 120": int((df["age"] > 120).sum()),
        "trestbps > 300": int((df["trestbps"] > 300).sum()),
        "chol > 700": int((df["chol"] > 700).sum()),
        "thalach > 250": int((df["thalach"] > 250).sum()),
        "oldpeak > 10": int((df["oldpeak"] > 10).sum()),
    }

    report["validation"]["categorical_range_warnings"] = categorical_warnings
    report["validation"]["numerical_range_warnings"] = numerical_warnings

    hard_fail_checks = {
        "age <= 0": numerical_warnings["age <= 0"],
        "trestbps <= 0": numerical_warnings["trestbps <= 0"],
        "chol <= 0": numerical_warnings["chol <= 0"],
        "thalach <= 0": numerical_warnings["thalach <= 0"],
        "oldpeak < 0": numerical_warnings["oldpeak < 0"],
    }

    for label, count in categorical_warnings.items():
        if count > 0:
            logger.warning(f"Categorical domain warning -> {label}: {count} rows")

    for label, count in numerical_warnings.items():
        if count > 0:
            logger.warning(f"Numerical domain warning -> {label}: {count} rows")

    for label, count in hard_fail_checks.items():
        if count > 0:
            raise ValueError(f"Invalid heart dataset values detected: {label} -> {count} rows")


# ---------------------------------------------------
# Final Data Validation
# ---------------------------------------------------

def validate_data(df: pd.DataFrame, report: Dict[str, Any]) -> None:
    logger.info("Performing final validation checks...")

    missing_values_remaining = int(df.isnull().sum().sum())
    negative_values_detected = bool((df[NUMERIC_COLUMNS] < 0).any().any())
    target_column_present = TARGET_COLUMN in df.columns

    report["validation"]["missing_values_remaining"] = missing_values_remaining
    report["validation"]["negative_values_detected"] = negative_values_detected
    report["validation"]["target_column_present"] = target_column_present
    report["final_shape"] = {"rows": int(df.shape[0]), "columns": int(df.shape[1])}

    if missing_values_remaining != 0:
        raise ValueError("Missing values still present in dataset.")

    if not target_column_present:
        raise ValueError("Target column missing.")

    validate_domain_ranges(df, report)

    logger.info("Validation successful")
    logger.info(f"Final Dataset Shape: {df.shape}")


# ---------------------------------------------------
# Save Clean Dataset
# ---------------------------------------------------

def save_cleaned_data(df: pd.DataFrame) -> None:
    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    df.to_csv(PROCESSED_DATA_PATH, index=False)

    logger.info("Cleaned dataset saved successfully")
    logger.info(f"Location: {PROCESSED_DATA_PATH}")


# ---------------------------------------------------
# Model Registry File Generators
# ---------------------------------------------------

def build_schema(df: pd.DataFrame) -> Dict[str, Any]:
    schema = {
        "dataset_name": DATASET_NAME,
        "target_column": TARGET_COLUMN,
        "task_type": "binary_classification",
        "row_count": int(df.shape[0]),
        "column_count": int(df.shape[1]),
        "feature_count": int(len(df.columns) - 1),
        "feature_columns": [col for col in df.columns if col != TARGET_COLUMN],
        "categorical_features": CATEGORICAL_COLUMNS,
        "numeric_features": NUMERIC_COLUMNS,
        "columns": [],
    }

    for col in df.columns:
        col_info = {
            "name": col,
            "dtype": str(df[col].dtype),
            "nullable": bool(df[col].isnull().any()),
            "unique_values": int(df[col].nunique(dropna=True)),
            "role": "target" if col == TARGET_COLUMN else "feature",
        }

        if pd.api.types.is_numeric_dtype(df[col]):
            col_info["statistics"] = {
                "min": to_serializable(df[col].min()),
                "max": to_serializable(df[col].max()),
                "mean": to_serializable(df[col].mean()),
                "median": to_serializable(df[col].median()),
                "std": to_serializable(df[col].std()),
            }

        schema["columns"].append(col_info)

    return schema


def build_feature_info(df: pd.DataFrame) -> Dict[str, Any]:
    feature_info = {
        "dataset_name": DATASET_NAME,
        "target_column": TARGET_COLUMN,
        "features": [],
    }

    for col in df.columns:
        if col == TARGET_COLUMN:
            continue

        feature_type = (
            "categorical" if col in CATEGORICAL_COLUMNS
            else "numeric" if col in NUMERIC_COLUMNS
            else "unknown"
        )

        info = {
            "name": col,
            "feature_type": feature_type,
            "dtype": str(df[col].dtype),
            "missing_count": int(df[col].isnull().sum()),
            "unique_values": int(df[col].nunique(dropna=True)),
        }

        if feature_type == "categorical":
            info["categories"] = sorted(
                [int(x) if pd.notnull(x) else x for x in df[col].dropna().unique().tolist()]
            )

        if feature_type == "numeric":
            info["summary_statistics"] = {
                "min": to_serializable(df[col].min()),
                "max": to_serializable(df[col].max()),
                "mean": to_serializable(df[col].mean()),
                "median": to_serializable(df[col].median()),
                "std": to_serializable(df[col].std()),
                "skewness": to_serializable(df[col].skew()),
            }

        feature_info["features"].append(info)

    return feature_info


def finalize_cleaning_report(report: Dict[str, Any], df: pd.DataFrame) -> Dict[str, Any]:
    report["pipeline_completed_at"] = datetime.utcnow().isoformat() + "Z"
    report["summary"] = {
        "row_count": int(df.shape[0]),
        "column_count": int(df.shape[1]),
        "total_missing_values": int(df.isnull().sum().sum()),
        "total_features": int(len(df.columns) - 1),
        "categorical_feature_count": int(len(CATEGORICAL_COLUMNS)),
        "numerical_feature_count": int(len(NUMERIC_COLUMNS)),
        "target_distribution": {
            str(k): int(v) for k, v in df[TARGET_COLUMN].value_counts().to_dict().items()
        },
    }
    return report


def save_json(data: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def generate_model_registry_files(df: pd.DataFrame, report: Dict[str, Any]) -> None:
    logger.info("Generating model registry files...")

    schema = build_schema(df)
    feature_info = build_feature_info(df)
    report = finalize_cleaning_report(report, df)

    save_json(schema, SCHEMA_PATH)
    save_json(feature_info, FEATURE_INFO_PATH)
    save_json(report, DATA_CLEANING_REPORT_PATH)

    logger.info("Model registry files generated successfully")


# ---------------------------------------------------
# Full Cleaning Pipeline
# ---------------------------------------------------

def run_data_cleaning_pipeline() -> pd.DataFrame:
    logger.info("Starting Heart Disease data cleaning pipeline...")

    report = initialize_cleaning_report()

    df = load_raw_data(report)
    validate_schema(df, report)
    df = remove_duplicates(df, report)
    df = standardize_missing_markers(df, report)
    df = enforce_data_types(df, report)
    df = handle_invalid_values(df, report)
    df = impute_missing_values(df, report)
    df = validate_target(df, report)
    validate_data(df, report)

    save_cleaned_data(df)
    generate_model_registry_files(df, report)

    logger.info("Heart Disease data cleaning completed successfully")
    return df


if __name__ == "__main__":
    run_data_cleaning_pipeline()
