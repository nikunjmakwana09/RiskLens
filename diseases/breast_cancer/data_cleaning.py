from __future__ import annotations

import os
import re
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

RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "breast_cancer.csv")
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "breast_cancer_clean.csv")

MODEL_REGISTRY_DIR = os.path.join(BASE_DIR, "model_registry", "breast_cancer", "data_clean")
SCHEMA_PATH = os.path.join(MODEL_REGISTRY_DIR, "schema.json")
FEATURE_INFO_PATH = os.path.join(MODEL_REGISTRY_DIR, "feature_info.json")
DATA_CLEANING_REPORT_PATH = os.path.join(MODEL_REGISTRY_DIR, "data_cleaning_report.json")


# ---------------------------------------------------
# Constants
# ---------------------------------------------------

DATASET_NAME = "breast_cancer"
TARGET_SOURCE_COLUMN = "diagnosis"
TARGET_COLUMN = "target"

TARGET_MAPPING = {"M": 1, "B": 0}
EXPECTED_TARGET_VALUES = {0, 1}

COMMON_MISSING_MARKERS = ["", " ", "NA", "N/A", "na", "null", "NULL", "?"]

EXPECTED_RAW_COLUMNS = {
    "id",
    "diagnosis",
    "radius_mean",
    "texture_mean",
    "perimeter_mean",
    "area_mean",
    "smoothness_mean",
    "compactness_mean",
    "concavity_mean",
    "concave_points_mean",
    "symmetry_mean",
    "fractal_dimension_mean",
    "radius_se",
    "texture_se",
    "perimeter_se",
    "area_se",
    "smoothness_se",
    "compactness_se",
    "concavity_se",
    "concave_points_se",
    "symmetry_se",
    "fractal_dimension_se",
    "radius_worst",
    "texture_worst",
    "perimeter_worst",
    "area_worst",
    "smoothness_worst",
    "compactness_worst",
    "concavity_worst",
    "concave_points_worst",
    "symmetry_worst",
    "fractal_dimension_worst",
}


# ---------------------------------------------------
# JSON Utilities
# ---------------------------------------------------

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


def save_json(data: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


# ---------------------------------------------------
# Report Initialization
# ---------------------------------------------------

def initialize_report() -> Dict[str, Any]:
    return {
        "dataset_name": DATASET_NAME,
        "pipeline_name": "breast_cancer_data_cleaning",
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "raw_data_path": RAW_DATA_PATH,
        "processed_data_path": PROCESSED_DATA_PATH,
        "schema_validation": {
            "missing_columns": [],
            "extra_columns": []
        },
        "missing_markers_standardized": False,
        "dropped_columns": [],
        "duplicates_removed": 0,
        "missing_values_before": {},
        "missing_value_imputation": {},
        "missing_values_after": 0,
        "dtype_enforcement": {},
        "invalid_value_corrections": {},
        "total_invalid_values_replaced": 0,
        "final_validation": {}
    }


# ---------------------------------------------------
# Load Dataset
# ---------------------------------------------------

def load_raw_data() -> pd.DataFrame:
    logger.info("Loading breast cancer dataset...")

    if not os.path.exists(RAW_DATA_PATH):
        raise FileNotFoundError(f"Dataset not found: {RAW_DATA_PATH}")

    df = pd.read_csv(RAW_DATA_PATH)

    logger.info("Dataset loaded successfully")
    logger.info(f"Initial Shape: {df.shape}")
    return df


# ---------------------------------------------------
# Standardize Column Names
# ---------------------------------------------------

def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Standardizing column names...")

    cleaned_columns: List[str] = []

    for col in df.columns:
        col = str(col).strip()
        col = col.replace("\t", " ")
        col = re.sub(r"\s+", "_", col)
        col = col.lower()
        cleaned_columns.append(col)

    df.columns = cleaned_columns

    logger.info("Column names standardized")
    logger.info(f"Columns: {df.columns.tolist()}")

    return df


# ---------------------------------------------------
# Schema Validation
# ---------------------------------------------------

def validate_schema(df: pd.DataFrame, report: Dict[str, Any]) -> None:
    logger.info("Validating raw schema...")

    current_columns = set(df.columns)
    missing_columns = sorted(list(EXPECTED_RAW_COLUMNS - current_columns))
    extra_columns = sorted(list(current_columns - EXPECTED_RAW_COLUMNS))

    report["schema_validation"]["missing_columns"] = missing_columns
    report["schema_validation"]["extra_columns"] = extra_columns

    if missing_columns:
        raise ValueError(f"Missing required raw columns: {missing_columns}")

    if extra_columns:
        logger.warning(f"Unexpected extra columns found: {extra_columns}")

    logger.info("Schema validation completed")


# ---------------------------------------------------
# Missing Marker Standardization
# ---------------------------------------------------

def standardize_missing_markers(df: pd.DataFrame, report: Dict[str, Any]) -> pd.DataFrame:
    logger.info("Standardizing missing markers...")
    df = df.replace(COMMON_MISSING_MARKERS, np.nan)
    report["missing_markers_standardized"] = True
    return df


# ---------------------------------------------------
# Drop Irrelevant Columns
# ---------------------------------------------------

def drop_irrelevant_columns(df: pd.DataFrame, report: Dict[str, Any]) -> pd.DataFrame:
    logger.info("Dropping irrelevant columns...")

    columns_to_drop: List[str] = []

    unnamed_cols = [col for col in df.columns if col.startswith("unnamed")]
    columns_to_drop.extend(unnamed_cols)

    if "id" in df.columns:
        columns_to_drop.append("id")

    columns_to_drop = sorted(list(set(columns_to_drop)))

    if columns_to_drop:
        df = df.drop(columns=columns_to_drop, errors="ignore")
        logger.info(f"Dropped columns: {columns_to_drop}")
    else:
        logger.info("No irrelevant columns found")

    report["dropped_columns"] = columns_to_drop
    logger.info(f"Shape after dropping columns: {df.shape}")
    return df


# ---------------------------------------------------
# Remove Duplicates
# ---------------------------------------------------

def remove_duplicates(df: pd.DataFrame, report: Dict[str, Any]) -> pd.DataFrame:
    duplicate_count = int(df.duplicated().sum())

    logger.info(f"Duplicate rows detected: {duplicate_count}")

    if duplicate_count > 0:
        df = df.drop_duplicates().reset_index(drop=True)
        logger.info("Duplicate rows removed")

    report["duplicates_removed"] = duplicate_count
    return df


# ---------------------------------------------------
# Encode Target
# ---------------------------------------------------

def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Encoding target column...")

    if TARGET_SOURCE_COLUMN not in df.columns:
        raise ValueError(f"Required column '{TARGET_SOURCE_COLUMN}' not found.")

    unique_values = set(df[TARGET_SOURCE_COLUMN].dropna().astype(str).str.strip().unique())
    if not unique_values.issubset(set(TARGET_MAPPING.keys())):
        raise ValueError(
            f"Unexpected diagnosis values found: {unique_values}. "
            f"Expected only {set(TARGET_MAPPING.keys())}"
        )

    df[TARGET_COLUMN] = df[TARGET_SOURCE_COLUMN].astype(str).str.strip().map(TARGET_MAPPING)
    df = df.drop(columns=[TARGET_SOURCE_COLUMN])

    logger.info(f"Target encoding completed: {df[TARGET_COLUMN].value_counts().to_dict()}")
    return df


# ---------------------------------------------------
# Enforce Data Types
# ---------------------------------------------------

def enforce_data_types(df: pd.DataFrame, report: Dict[str, Any]) -> pd.DataFrame:
    logger.info("Enforcing data types...")

    dtype_report: Dict[str, Dict[str, str]] = {}

    for col in df.columns:
        before = str(df[col].dtype)

        if col == TARGET_COLUMN:
            df[col] = pd.to_numeric(df[col], errors="raise").astype("int64")
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

        dtype_report[col] = {
            "before": before,
            "after": str(df[col].dtype)
        }

    report["dtype_enforcement"] = dtype_report
    logger.info("Data type enforcement completed")
    return df


# ---------------------------------------------------
# Missing Value Handling
# ---------------------------------------------------

def handle_missing_values(df: pd.DataFrame, report: Dict[str, Any]) -> pd.DataFrame:
    logger.info("Checking missing values...")

    missing_summary = df.isnull().sum()
    missing_cols = missing_summary[missing_summary > 0]

    report["missing_values_before"] = {
        col: int(cnt) for col, cnt in missing_cols.to_dict().items()
    }
    report["missing_value_imputation"] = {}

    if missing_cols.empty:
        logger.info("No missing values detected")
        report["missing_values_after"] = 0
        return df

    feature_columns = [col for col in df.columns if col != TARGET_COLUMN]

    for col in feature_columns:
        missing_count = int(df[col].isnull().sum())
        if missing_count > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            report["missing_value_imputation"][col] = {
                "strategy": "median",
                "filled_count": missing_count,
                "median_used": to_serializable(median_val)
            }
            logger.info(f"{col}: filled {missing_count} missing values with median")

    if df[TARGET_COLUMN].isnull().sum() > 0:
        raise ValueError("Missing values detected in target column.")

    remaining_missing = int(df.isnull().sum().sum())
    report["missing_values_after"] = remaining_missing

    if remaining_missing > 0:
        raise ValueError(f"Missing values still present after imputation: {remaining_missing}")

    logger.info("Missing value handling completed")
    return df


# ---------------------------------------------------
# Invalid Medical Values
# ---------------------------------------------------

def handle_invalid_values(df: pd.DataFrame, report: Dict[str, Any]) -> pd.DataFrame:
    logger.info("Checking medically invalid values...")

    feature_columns = [col for col in df.columns if col != TARGET_COLUMN]
    total_replacements = 0
    report["invalid_value_corrections"] = {}

    for col in feature_columns:
        invalid_count = int((df[col] < 0).sum())

        if invalid_count > 0:
            valid_series = df.loc[df[col] >= 0, col]
            median_val = valid_series.median()

            if pd.isna(median_val):
                raise ValueError(f"Cannot compute replacement median for column '{col}'.")

            df.loc[df[col] < 0, col] = median_val
            total_replacements += invalid_count

            report["invalid_value_corrections"][col] = {
                "invalid_negative_count": invalid_count,
                "replacement_strategy": "median_of_non_negative_values",
                "median_used": to_serializable(median_val)
            }
            logger.warning(f"{col}: replaced {invalid_count} negative values with median")

    report["total_invalid_values_replaced"] = total_replacements
    logger.info(f"Total invalid values replaced: {total_replacements}")
    return df


# ---------------------------------------------------
# Final Validation
# ---------------------------------------------------

def validate_data(df: pd.DataFrame, report: Dict[str, Any]) -> None:
    logger.info("Performing final validation...")

    if df.empty:
        raise ValueError("Dataset is empty after preprocessing.")

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' is missing.")

    if df.isnull().sum().sum() != 0:
        raise ValueError("Missing values still present in dataset.")

    target_values = set(df[TARGET_COLUMN].unique())
    if not target_values.issubset(EXPECTED_TARGET_VALUES):
        raise ValueError(
            f"Unexpected target values found: {target_values}. "
            f"Expected only {EXPECTED_TARGET_VALUES}"
        )

    feature_columns = [col for col in df.columns if col != TARGET_COLUMN]
    if not feature_columns:
        raise ValueError("No feature columns available after preprocessing.")

    non_numeric = df[feature_columns].select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        raise ValueError(f"Non-numeric feature columns detected: {non_numeric}")

    negative_feature_value_count = int((df[feature_columns] < 0).sum().sum())

    report["final_validation"] = {
        "null_values_remaining": int(df.isnull().sum().sum()),
        "target_values": sorted([to_serializable(v) for v in df[TARGET_COLUMN].unique().tolist()]),
        "feature_count": int(len(feature_columns)),
        "all_features_numeric": len(non_numeric) == 0,
        "negative_feature_value_count": negative_feature_value_count,
        "final_shape": {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1])
        }
    }

    if negative_feature_value_count > 0:
        logger.warning(f"Negative feature values remain after cleaning: {negative_feature_value_count}")

    logger.info(f"Validation successful. Final shape: {df.shape}")


# ---------------------------------------------------
# Save Clean Dataset
# ---------------------------------------------------

def save_cleaned_data(df: pd.DataFrame) -> None:
    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    df.to_csv(PROCESSED_DATA_PATH, index=False)

    logger.info("Cleaned dataset saved successfully")
    logger.info(f"Location: {PROCESSED_DATA_PATH}")


# ---------------------------------------------------
# Registry Artifacts
# ---------------------------------------------------

def build_schema(df: pd.DataFrame) -> Dict[str, Any]:
    feature_columns = [col for col in df.columns if col != TARGET_COLUMN]
    numeric_columns = df[feature_columns].select_dtypes(include=[np.number]).columns.tolist()

    return {
        "dataset_name": DATASET_NAME,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "target_column": TARGET_COLUMN,
        "row_count": int(df.shape[0]),
        "column_count": int(df.shape[1]),
        "feature_count": int(len(feature_columns)),
        "feature_columns": feature_columns,
        "numeric_features": numeric_columns,
        "categorical_features": [],
        "target_dtype": str(df[TARGET_COLUMN].dtype),
        "feature_dtypes": {col: str(df[col].dtype) for col in feature_columns},
        "target_classes": sorted([to_serializable(v) for v in df[TARGET_COLUMN].unique().tolist()]),
        "target_distribution": {
            str(k): int(v) for k, v in df[TARGET_COLUMN].value_counts().to_dict().items()
        }
    }


def build_feature_info(df: pd.DataFrame, report: Dict[str, Any]) -> Dict[str, Any]:
    feature_columns = [col for col in df.columns if col != TARGET_COLUMN]
    invalid_corrected_columns = set(report.get("invalid_value_corrections", {}).keys())

    feature_info = {
        "dataset_name": DATASET_NAME,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "target_column": TARGET_COLUMN,
        "features": {}
    }

    for col in feature_columns:
        series = df[col]
        feature_info["features"][col] = {
            "dtype": str(series.dtype),
            "role": "feature",
            "missing_count": int(series.isnull().sum()),
            "unique_count": int(series.nunique(dropna=True)),
            "mean": to_serializable(series.mean()),
            "median": to_serializable(series.median()),
            "std": to_serializable(series.std()),
            "min": to_serializable(series.min()),
            "max": to_serializable(series.max()),
            "skewness": to_serializable(series.skew()),
            "is_numeric": bool(pd.api.types.is_numeric_dtype(series)),
            "transformed": False,
            "transformation_type": None,
            "outlier_capping_applied": False,
            "outliers_capped_count": 0,
            "invalid_values_corrected": col in invalid_corrected_columns,
            "invalid_values_corrected_count": int(
                report.get("invalid_value_corrections", {}).get(col, {}).get("invalid_negative_count", 0)
            )
        }

    return feature_info


def build_data_cleaning_report(report: Dict[str, Any], input_shape: tuple, output_df: pd.DataFrame) -> Dict[str, Any]:
    return {
        "dataset_name": DATASET_NAME,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "pipeline_name": "breast_cancer_data_cleaning",
        "status": "success",
        "input_shape": {
            "rows": int(input_shape[0]),
            "columns": int(input_shape[1])
        },
        "output_shape": {
            "rows": int(output_df.shape[0]),
            "columns": int(output_df.shape[1])
        },
        "target_column": TARGET_COLUMN,
        "steps_summary": report,
        "artifacts": {
            "cleaned_data_path": PROCESSED_DATA_PATH,
            "schema_path": SCHEMA_PATH,
            "feature_info_path": FEATURE_INFO_PATH,
            "data_cleaning_report_path": DATA_CLEANING_REPORT_PATH
        }
    }


def save_model_registry_artifacts(df: pd.DataFrame, report: Dict[str, Any], input_shape: tuple) -> None:
    logger.info("Generating model registry artifacts...")

    os.makedirs(MODEL_REGISTRY_DIR, exist_ok=True)

    schema = build_schema(df)
    feature_info = build_feature_info(df, report)
    cleaning_report = build_data_cleaning_report(report, input_shape, df)

    save_json(schema, SCHEMA_PATH)
    save_json(feature_info, FEATURE_INFO_PATH)
    save_json(cleaning_report, DATA_CLEANING_REPORT_PATH)

    logger.info(f"schema.json saved at: {SCHEMA_PATH}")
    logger.info(f"feature_info.json saved at: {FEATURE_INFO_PATH}")
    logger.info(f"data_cleaning_report.json saved at: {DATA_CLEANING_REPORT_PATH}")


# ---------------------------------------------------
# Full Cleaning Pipeline
# ---------------------------------------------------

def run_data_cleaning_pipeline() -> pd.DataFrame:
    logger.info("Starting Breast Cancer data cleaning pipeline...")

    report = initialize_report()

    df = load_raw_data()
    input_shape = df.shape

    df = standardize_column_names(df)
    validate_schema(df, report)
    df = standardize_missing_markers(df, report)
    df = drop_irrelevant_columns(df, report)
    df = remove_duplicates(df, report)
    df = encode_target(df)
    df = enforce_data_types(df, report)
    df = handle_missing_values(df, report)
    df = handle_invalid_values(df, report)

    validate_data(df, report)
    save_cleaned_data(df)
    save_model_registry_artifacts(df, report, input_shape)

    logger.info("Breast Cancer data cleaning completed successfully")
    return df


if __name__ == "__main__":
    run_data_cleaning_pipeline()
