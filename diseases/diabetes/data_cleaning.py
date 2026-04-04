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

RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "diabetes.csv")
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "diabetes_clean.csv")

MODEL_REGISTRY_DIR = os.path.join(BASE_DIR, "model_registry", "diabetes", "data_clean")
SCHEMA_PATH = os.path.join(MODEL_REGISTRY_DIR, "schema.json")
FEATURE_INFO_PATH = os.path.join(MODEL_REGISTRY_DIR, "feature_info.json")
DATA_CLEANING_REPORT_PATH = os.path.join(MODEL_REGISTRY_DIR, "data_cleaning_report.json")


# ---------------------------------------------------
# Constants
# ---------------------------------------------------

DATASET_NAME = "diabetes"
TARGET_COLUMN = "Outcome"

EXPECTED_COLUMNS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
    "Outcome",
]

INVALID_ZERO_COLUMNS = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
]

NUMERIC_FEATURE_COLUMNS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

COMMON_MISSING_MARKERS = ["", " ", "NA", "N/A", "null", "NULL", "?"]


# ---------------------------------------------------
# Metadata Helpers
# ---------------------------------------------------

def initialize_cleaning_metadata() -> Dict[str, Any]:
    return {
        "dataset_name": DATASET_NAME,
        "target_column": TARGET_COLUMN,
        "raw_data_path": RAW_DATA_PATH,
        "processed_data_path": PROCESSED_DATA_PATH,
        "pipeline_started_at": datetime.utcnow().isoformat() + "Z",
        "pipeline_completed_at": None,
        "initial_shape": None,
        "final_shape": None,
        "schema_validation": {
            "missing_columns": [],
            "extra_columns": [],
        },
        "rows_removed_duplicates": 0,
        "missing_markers_standardized": False,
        "missing_values_before": 0,
        "missing_values_after": 0,
        "invalid_zero_replacements": {},
        "invalid_zero_replacements_total": 0,
        "missing_imputation": {},
        "target_validation": {
            "unique_values": [],
        },
        "range_validation_warnings": {},
        "dtypes_after_cleaning": {},
        "final_null_count": 0,
        "final_validation": {},
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


def ensure_model_registry_dir() -> None:
    os.makedirs(MODEL_REGISTRY_DIR, exist_ok=True)


def save_json(data: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


# ---------------------------------------------------
# Load Dataset
# ---------------------------------------------------

def load_raw_data(metadata: Dict[str, Any]) -> pd.DataFrame:
    logger.info("Loading diabetes dataset...")

    if not os.path.exists(RAW_DATA_PATH):
        raise FileNotFoundError(f"Dataset not found: {RAW_DATA_PATH}")

    df = pd.read_csv(RAW_DATA_PATH)

    logger.info("Dataset loaded successfully")
    logger.info(f"Initial shape: {df.shape}")

    metadata["initial_shape"] = [int(df.shape[0]), int(df.shape[1])]
    return df


# ---------------------------------------------------
# Schema Validation
# ---------------------------------------------------

def validate_schema(df: pd.DataFrame, metadata: Dict[str, Any]) -> None:
    logger.info("Validating dataset schema...")

    missing_cols = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    extra_cols = [col for col in df.columns if col not in EXPECTED_COLUMNS]

    metadata["schema_validation"]["missing_columns"] = missing_cols
    metadata["schema_validation"]["extra_columns"] = extra_cols

    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    if extra_cols:
        logger.warning(f"Unexpected extra columns detected: {extra_cols}")

    logger.info("Schema validation successful")


# ---------------------------------------------------
# Remove Duplicates
# ---------------------------------------------------

def remove_duplicates(df: pd.DataFrame, metadata: Dict[str, Any]) -> pd.DataFrame:
    duplicate_count = int(df.duplicated().sum())

    logger.info(f"Duplicate rows detected: {duplicate_count}")
    metadata["rows_removed_duplicates"] = duplicate_count

    if duplicate_count > 0:
        df = df.drop_duplicates().reset_index(drop=True)
        logger.info("Duplicate rows removed")

    return df


# ---------------------------------------------------
# Standardize Missing Placeholders
# ---------------------------------------------------

def standardize_missing_markers(df: pd.DataFrame, metadata: Dict[str, Any]) -> pd.DataFrame:
    logger.info("Standardizing missing value markers...")
    df = df.replace(COMMON_MISSING_MARKERS, np.nan)
    metadata["missing_markers_standardized"] = True
    return df


# ---------------------------------------------------
# Data Type Standardization
# ---------------------------------------------------

def enforce_data_types(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Standardizing feature data types...")

    for col in EXPECTED_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    logger.info("Data type standardization completed")
    return df


# ---------------------------------------------------
# Handle Invalid Medical Values
# ---------------------------------------------------

def handle_invalid_medical_values(df: pd.DataFrame, metadata: Dict[str, Any]) -> pd.DataFrame:
    logger.info("Checking medically invalid zero values...")

    total_replacements = 0

    for col in INVALID_ZERO_COLUMNS:
        zero_count = int((df[col] == 0).sum())
        logger.info(f"{col}: {zero_count} invalid zero values detected")

        if zero_count > 0:
            valid_median = df.loc[df[col] != 0, col].median()

            if pd.isna(valid_median):
                raise ValueError(f"Cannot compute valid median for column: {col}")

            df.loc[df[col] == 0, col] = valid_median
            total_replacements += zero_count

            metadata["invalid_zero_replacements"][col] = {
                "replaced_count": zero_count,
                "replacement_strategy": "non_zero_median",
                "median_used": to_serializable(valid_median),
            }
        else:
            metadata["invalid_zero_replacements"][col] = {
                "replaced_count": 0,
                "replacement_strategy": "non_zero_median",
                "median_used": None,
            }

    metadata["invalid_zero_replacements_total"] = total_replacements
    logger.info(f"Total medically invalid values replaced: {total_replacements}")

    return df


# ---------------------------------------------------
# Missing Value Handling
# ---------------------------------------------------

def impute_missing_values(df: pd.DataFrame, metadata: Dict[str, Any]) -> pd.DataFrame:
    logger.info("Handling missing values...")

    missing_before = int(df.isnull().sum().sum())
    metadata["missing_values_before"] = missing_before

    for col in NUMERIC_FEATURE_COLUMNS:
        col_missing = int(df[col].isnull().sum())

        if col_missing > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)

            metadata["missing_imputation"][col] = {
                "strategy": "median",
                "imputed_count": col_missing,
                "fill_value": to_serializable(median_val),
            }

            logger.info(f"{col}: imputed {col_missing} missing values with median")
        else:
            metadata["missing_imputation"][col] = {
                "strategy": "median",
                "imputed_count": 0,
                "fill_value": None,
            }

    if df[TARGET_COLUMN].isnull().sum() > 0:
        raise ValueError("Missing values found in target column.")

    missing_after = int(df.isnull().sum().sum())
    metadata["missing_values_after"] = missing_after

    if missing_after > 0:
        raise ValueError(f"Missing values remain after imputation: {missing_after}")

    logger.info(f"Missing values before imputation: {missing_before}")
    logger.info(f"Missing values after imputation: {missing_after}")

    return df


# ---------------------------------------------------
# Target Validation
# ---------------------------------------------------

def validate_target(df: pd.DataFrame, metadata: Dict[str, Any]) -> pd.DataFrame:
    logger.info("Validating target column...")

    unique_values = sorted(df[TARGET_COLUMN].dropna().unique().tolist())
    metadata["target_validation"]["unique_values"] = [to_serializable(v) for v in unique_values]

    if unique_values != [0, 1]:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' must contain binary values [0, 1], found: {unique_values}"
        )

    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype("int64")

    logger.info("Target validation successful")
    return df


# ---------------------------------------------------
# Medical Range Validation
# ---------------------------------------------------

def validate_medical_ranges(df: pd.DataFrame, metadata: Dict[str, Any]) -> None:
    logger.info("Performing medical range validation...")

    if (df["Pregnancies"] < 0).any():
        raise ValueError("Negative values found in Pregnancies.")

    if (df["Age"] <= 0).any():
        raise ValueError("Non-positive values found in Age.")

    if (df["BMI"] <= 0).any():
        raise ValueError("Non-positive values found in BMI after cleaning.")

    if (df["Glucose"] <= 0).any():
        raise ValueError("Non-positive values found in Glucose after cleaning.")

    suspicious_checks = {
        "Glucose > 300": int((df["Glucose"] > 300).sum()),
        "BloodPressure > 200": int((df["BloodPressure"] > 200).sum()),
        "SkinThickness > 100": int((df["SkinThickness"] > 100).sum()),
        "Insulin > 900": int((df["Insulin"] > 900).sum()),
        "BMI > 70": int((df["BMI"] > 70).sum()),
        "Age > 120": int((df["Age"] > 120).sum()),
    }

    metadata["range_validation_warnings"] = suspicious_checks

    for label, count in suspicious_checks.items():
        if count > 0:
            logger.warning(f"Suspicious values detected -> {label}: {count} rows")


# ---------------------------------------------------
# Final Validation
# ---------------------------------------------------

def validate_data(df: pd.DataFrame, metadata: Dict[str, Any]) -> None:
    logger.info("Performing final validation checks...")

    if df.empty:
        raise ValueError("Processed dataset is empty.")

    if df.isnull().sum().sum() != 0:
        raise ValueError("Missing values still present in dataset.")

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column missing: {TARGET_COLUMN}")

    if not all(col in df.columns for col in EXPECTED_COLUMNS):
        raise ValueError("Processed dataset schema mismatch.")

    validate_medical_ranges(df, metadata)

    non_numeric_features = df[NUMERIC_FEATURE_COLUMNS].select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric_features:
        raise ValueError(f"Non-numeric feature columns detected: {non_numeric_features}")

    metadata["final_null_count"] = int(df.isnull().sum().sum())
    metadata["final_shape"] = [int(df.shape[0]), int(df.shape[1])]
    metadata["dtypes_after_cleaning"] = {col: str(dtype) for col, dtype in df.dtypes.items()}
    metadata["final_validation"] = {
        "all_expected_columns_present": True,
        "all_features_numeric": len(non_numeric_features) == 0,
        "target_binary": True,
        "row_count": int(df.shape[0]),
        "column_count": int(df.shape[1]),
    }

    logger.info(f"Validation successful. Final dataset shape: {df.shape}")


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

def generate_schema_json(df: pd.DataFrame) -> Dict[str, Any]:
    feature_columns = [col for col in df.columns if col != TARGET_COLUMN]

    schema = {
        "dataset_name": DATASET_NAME,
        "target_column": TARGET_COLUMN,
        "task_type": "binary_classification",
        "row_count": int(df.shape[0]),
        "column_count": int(df.shape[1]),
        "feature_count": int(len(feature_columns)),
        "feature_columns": feature_columns,
        "numeric_features": feature_columns,
        "categorical_features": [],
        "columns": [],
    }

    for col in df.columns:
        col_info = {
            "name": col,
            "dtype": str(df[col].dtype),
            "role": "target" if col == TARGET_COLUMN else "feature",
            "nullable": bool(df[col].isnull().any()),
            "unique_values": int(df[col].nunique(dropna=True)),
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


def generate_feature_info_json(df: pd.DataFrame) -> Dict[str, Any]:
    feature_info = {
        "dataset_name": DATASET_NAME,
        "target_column": TARGET_COLUMN,
        "features": [],
    }

    for col in df.columns:
        if col == TARGET_COLUMN:
            continue

        feature_record = {
            "name": col,
            "display_name": col,
            "dtype": str(df[col].dtype),
            "feature_type": "numeric",
            "input_required": True,
            "has_missing_after_cleaning": bool(df[col].isnull().any()),
            "description": f"{col} feature used for diabetes prediction.",
            "summary_statistics": {
                "min": to_serializable(df[col].min()),
                "max": to_serializable(df[col].max()),
                "mean": to_serializable(df[col].mean()),
                "median": to_serializable(df[col].median()),
                "std": to_serializable(df[col].std()),
            },
            "special_handling": {
                "invalid_zero_treated_as_missing": col in INVALID_ZERO_COLUMNS,
                "cleaning_rule": (
                    "zero_replaced_with_non_zero_median"
                    if col in INVALID_ZERO_COLUMNS
                    else "standard_numeric_cleaning"
                ),
            },
        }

        feature_info["features"].append(feature_record)

    return feature_info


def generate_data_cleaning_report_json(metadata: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "dataset_name": metadata["dataset_name"],
        "target_column": metadata["target_column"],
        "raw_data_path": metadata["raw_data_path"],
        "processed_data_path": metadata["processed_data_path"],
        "pipeline_started_at": metadata["pipeline_started_at"],
        "pipeline_completed_at": metadata["pipeline_completed_at"],
        "initial_shape": metadata["initial_shape"],
        "final_shape": metadata["final_shape"],
        "schema_validation": metadata["schema_validation"],
        "rows_removed_duplicates": metadata["rows_removed_duplicates"],
        "missing_markers_standardized": metadata["missing_markers_standardized"],
        "missing_values_before": metadata["missing_values_before"],
        "missing_values_after": metadata["missing_values_after"],
        "invalid_zero_replacements": metadata["invalid_zero_replacements"],
        "invalid_zero_replacements_total": metadata["invalid_zero_replacements_total"],
        "missing_imputation": metadata["missing_imputation"],
        "target_validation": metadata["target_validation"],
        "range_validation_warnings": metadata["range_validation_warnings"],
        "final_null_count": metadata["final_null_count"],
        "dtypes_after_cleaning": metadata["dtypes_after_cleaning"],
        "final_validation": metadata["final_validation"],
        "status": "success",
    }


def save_model_registry_files(df: pd.DataFrame, metadata: Dict[str, Any]) -> None:
    logger.info("Generating model registry files...")

    ensure_model_registry_dir()

    schema = generate_schema_json(df)
    feature_info = generate_feature_info_json(df)
    data_cleaning_report = generate_data_cleaning_report_json(metadata)

    save_json(schema, SCHEMA_PATH)
    save_json(feature_info, FEATURE_INFO_PATH)
    save_json(data_cleaning_report, DATA_CLEANING_REPORT_PATH)

    logger.info(f"schema.json saved at: {SCHEMA_PATH}")
    logger.info(f"feature_info.json saved at: {FEATURE_INFO_PATH}")
    logger.info(f"data_cleaning_report.json saved at: {DATA_CLEANING_REPORT_PATH}")


# ---------------------------------------------------
# Full Cleaning Pipeline
# ---------------------------------------------------

def run_data_cleaning_pipeline() -> pd.DataFrame:
    logger.info("Starting Diabetes data cleaning pipeline...")

    metadata = initialize_cleaning_metadata()

    df = load_raw_data(metadata)
    validate_schema(df, metadata)
    df = remove_duplicates(df, metadata)
    df = standardize_missing_markers(df, metadata)
    df = enforce_data_types(df)
    df = handle_invalid_medical_values(df, metadata)
    df = impute_missing_values(df, metadata)
    df = validate_target(df, metadata)
    validate_data(df, metadata)
    save_cleaned_data(df)

    metadata["pipeline_completed_at"] = datetime.utcnow().isoformat() + "Z"
    save_model_registry_files(df, metadata)

    logger.info("Diabetes data cleaning completed successfully")
    return df


if __name__ == "__main__":
    run_data_cleaning_pipeline()
