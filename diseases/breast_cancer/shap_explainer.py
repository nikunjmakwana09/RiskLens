from __future__ import annotations

import os
import json
import pickle
import logging
import warnings
from datetime import datetime
from typing import Dict, Tuple, List, Any

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

from sklearn.model_selection import train_test_split
try:
    from services.preprocessing_utils import safe_log1p_array
except ModuleNotFoundError:
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
    from services.preprocessing_utils import safe_log1p_array


warnings.filterwarnings("ignore")


# ---------------------------------------------------
# Logging
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

DATASET_NAME = "breast_cancer"
TARGET_COLUMN = "target"

DATA_PATH = os.path.join(BASE_DIR, "data", "processed", f"{DATASET_NAME}_clean.csv")

TRAIN_REGISTRY_DIR = os.path.join(BASE_DIR, "model_registry", DATASET_NAME, "train")
SHAP_REGISTRY_DIR = os.path.join(BASE_DIR, "model_registry", DATASET_NAME, "shap")

MODEL_PATH = os.path.join(TRAIN_REGISTRY_DIR, "model.pkl")
PREPROCESSOR_PATH = os.path.join(TRAIN_REGISTRY_DIR, "preprocessor.pkl")
THRESHOLD_PATH = os.path.join(TRAIN_REGISTRY_DIR, "threshold.json")
SELECTED_FEATURES_PATH = os.path.join(TRAIN_REGISTRY_DIR, "selected_features.json")
MODEL_METADATA_PATH = os.path.join(TRAIN_REGISTRY_DIR, "model_metadata.json")
TRAINING_SUMMARY_PATH = os.path.join(TRAIN_REGISTRY_DIR, "training_summary.json")
TRAINING_CONFIG_PATH = os.path.join(TRAIN_REGISTRY_DIR, "training_config.json")

SHAP_SUMMARY_PNG_PATH = os.path.join(SHAP_REGISTRY_DIR, "shap_summary.png")
SHAP_BAR_PNG_PATH = os.path.join(SHAP_REGISTRY_DIR, "shap_bar.png")
GLOBAL_FEATURE_IMPACT_CSV_PATH = os.path.join(SHAP_REGISTRY_DIR, "global_feature_impact.csv")
LOCAL_EXPLANATIONS_JSON_PATH = os.path.join(SHAP_REGISTRY_DIR, "local_explanations.json")
EXPLAINABILITY_SUMMARY_JSON_PATH = os.path.join(SHAP_REGISTRY_DIR, "explainability_summary.json")

os.makedirs(SHAP_REGISTRY_DIR, exist_ok=True)


# ---------------------------------------------------
# Constants
# ---------------------------------------------------

RANDOM_STATE = 42
TEST_SIZE = 0.15
VALID_SIZE = 0.15

BACKGROUND_SAMPLE_SIZE = 80
EXPLANATION_SAMPLE_SIZE = 80
TOP_N_FEATURES = 20
LOCAL_EXPLANATION_SAMPLES = 5
LOCAL_TOP_FEATURES = 10


# ---------------------------------------------------
# Utility
# ---------------------------------------------------

def to_serializable(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        if np.isnan(value):
            return None
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    if pd.isna(value):
        return None
    return value


def make_json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_json_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return [make_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Series):
        return obj.to_dict()
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    return to_serializable(obj)


def save_json(data: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(make_json_safe(data), f, indent=4, ensure_ascii=False)
    logger.info(f"Saved JSON: {path}")


def load_json(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required JSON file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_json_if_exists(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_threshold(threshold_data: Dict[str, Any]) -> float:
    for key in ["threshold", "selected_threshold", "optimal_threshold"]:
        if key in threshold_data:
            return float(threshold_data[key])

    nested_paths = [
        ("selected_metrics", "threshold"),
        ("validation_metrics_at_optimal_threshold", "threshold")
    ]

    for outer, inner in nested_paths:
        if outer in threshold_data and isinstance(threshold_data[outer], dict) and inner in threshold_data[outer]:
            return float(threshold_data[outer][inner])

    raise ValueError("Threshold value not found in threshold.json")


# ---------------------------------------------------
# Feature engineering
# ---------------------------------------------------

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Must match latest train.py exactly.
    """
    df = df.copy()
    eps = 1e-6

    if {"radius_mean", "perimeter_mean"}.issubset(df.columns):
        df["radius_perimeter_ratio_mean"] = df["radius_mean"] / (df["perimeter_mean"] + eps)

    if {"area_mean", "radius_mean"}.issubset(df.columns):
        df["area_radius_ratio_mean"] = df["area_mean"] / (df["radius_mean"] + eps)

    if {"radius_worst", "radius_mean"}.issubset(df.columns):
        df["radius_worst_to_mean"] = df["radius_worst"] / (df["radius_mean"] + eps)

    if {"area_worst", "area_mean"}.issubset(df.columns):
        df["area_worst_to_mean"] = df["area_worst"] / (df["area_mean"] + eps)

    if {"compactness_mean", "concavity_mean"}.issubset(df.columns):
        df["compactness_concavity_interaction"] = df["compactness_mean"] * df["concavity_mean"]

    return df


# ---------------------------------------------------
# Load data
# ---------------------------------------------------

def load_data() -> Tuple[pd.DataFrame, pd.Series]:
    logger.info("Loading processed breast cancer dataset...")

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in dataset.")

    if df.empty:
        raise ValueError("Dataset is empty.")

    if df.isnull().sum().sum() != 0:
        raise ValueError("Processed dataset contains missing values.")

    X = df.drop(columns=[TARGET_COLUMN]).copy()
    y = df[TARGET_COLUMN].astype(int).copy()

    X = feature_engineering(X)

    logger.info(f"Dataset loaded successfully. Shape: {df.shape}")
    logger.info(f"Feature shape after engineering: {X.shape}")

    return X, y


def recreate_split(
    X: pd.DataFrame,
    y: pd.Series
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    logger.info("Recreating train/validation/test split...")

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE
    )

    val_relative_size = VALID_SIZE / (1 - TEST_SIZE)

    X_train, X_val, X_test_dummy, y_train, y_val, y_test_dummy = (
        X_train_full, None, None, y_train_full, None, None
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=val_relative_size,
        stratify=y_train_full,
        random_state=RANDOM_STATE
    )

    logger.info(f"Train shape: {X_train.shape}")
    logger.info(f"Validation shape: {X_val.shape}")
    logger.info(f"Test shape: {X_test.shape}")

    return X_train, X_val, X_test, y_train, y_val, y_test


# ---------------------------------------------------
# Artifact loading
# ---------------------------------------------------

def load_registry_artifacts() -> Tuple[Any, Dict[str, Any], float, List[str], Dict[str, Any]]:
    logger.info("Loading training registry artifacts...")

    required_paths = [
        MODEL_PATH,
        PREPROCESSOR_PATH,
        THRESHOLD_PATH,
        SELECTED_FEATURES_PATH
    ]
    for path in required_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required artifact not found: {path}")

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    with open(PREPROCESSOR_PATH, "rb") as f:
        preprocessor_artifact = pickle.load(f)

    threshold_data = load_json(THRESHOLD_PATH)
    selected_features_payload = load_json(SELECTED_FEATURES_PATH)

    if not isinstance(preprocessor_artifact, dict):
        raise ValueError("preprocessor.pkl must be a packaged dictionary artifact.")

    if "fitted_feature_pipeline" not in preprocessor_artifact:
        raise ValueError("Missing 'fitted_feature_pipeline' in preprocessor artifact.")

    selected_features = selected_features_payload.get("selected_features", [])
    if not selected_features:
        raise ValueError("selected_features.json does not contain selected_features.")

    metadata = {
        "selected_features": selected_features_payload,
        "training_config": load_json_if_exists(TRAINING_CONFIG_PATH),
        "training_summary": load_json_if_exists(TRAINING_SUMMARY_PATH),
        "model_metadata": load_json_if_exists(MODEL_METADATA_PATH),
        "threshold_data": threshold_data
    }

    threshold = extract_threshold(threshold_data)

    logger.info(f"Loaded threshold: {threshold:.4f}")
    logger.info(f"Loaded selected feature count: {len(selected_features)}")

    return model, preprocessor_artifact, threshold, selected_features, metadata


# ---------------------------------------------------
# Preprocessing
# ---------------------------------------------------

def transform_features(
    X: pd.DataFrame,
    preprocessor_artifact: Dict[str, Any],
    selected_features: List[str]
) -> pd.DataFrame:
    logger.info("Applying saved preprocessing contract...")

    fitted_pipeline = preprocessor_artifact["fitted_feature_pipeline"]
    X_ready = fitted_pipeline.transform(X)
    X_ready = np.asarray(X_ready)

    if X_ready.shape[1] != len(selected_features):
        raise ValueError(
            f"Transformed feature count mismatch: got {X_ready.shape[1]}, "
            f"expected {len(selected_features)}"
        )

    X_ready_df = pd.DataFrame(X_ready, columns=selected_features, index=X.index)
    return X_ready_df


# ---------------------------------------------------
# Sampling
# ---------------------------------------------------

def sample_background_data(X_train_ready: pd.DataFrame) -> pd.DataFrame:
    sample_size = min(BACKGROUND_SAMPLE_SIZE, len(X_train_ready))
    background = X_train_ready.sample(n=sample_size, random_state=RANDOM_STATE)
    logger.info(f"Background sample shape: {background.shape}")
    return background


def sample_explanation_data(X_test_ready: pd.DataFrame) -> pd.DataFrame:
    sample_size = min(EXPLANATION_SAMPLE_SIZE, len(X_test_ready))
    explain_data = X_test_ready.sample(n=sample_size, random_state=RANDOM_STATE)
    logger.info(f"Explanation sample shape: {explain_data.shape}")
    return explain_data


# ---------------------------------------------------
# SHAP helpers
# ---------------------------------------------------

def positive_class_predict_fn(model: Any):
    def _predict_fn(X_array):
        X_array = np.asarray(X_array, dtype=float)
        return model.predict_proba(X_array)[:, 1]
    return _predict_fn


def build_shap_explainer(model: Any, background_data: pd.DataFrame):
    logger.info("Building SHAP explainer...")

    predict_fn = positive_class_predict_fn(model)

    try:
        explainer = shap.Explainer(
            predict_fn,
            background_data,
            feature_names=list(background_data.columns)
        )
        logger.info("Using shap.Explainer on positive-class probability")
        return explainer
    except Exception as e:
        logger.warning(f"shap.Explainer failed: {e}")

    try:
        explainer = shap.KernelExplainer(
            predict_fn,
            background_data
        )
        logger.info("Using shap.KernelExplainer fallback")
        return explainer
    except Exception as e:
        logger.error(f"KernelExplainer failed: {e}")
        raise RuntimeError("Unable to build SHAP explainer for breast cancer model.")


def compute_shap_matrix(explainer, explain_data: pd.DataFrame) -> np.ndarray:
    logger.info("Computing SHAP values...")

    try:
        shap_values = explainer(explain_data)
        values = getattr(shap_values, "values", shap_values)
    except Exception:
        shap_values = explainer.shap_values(explain_data)
        values = shap_values

    values = np.asarray(values)

    if values.ndim == 3:
        # unlikely here, but kept for safety
        values = values[:, :, 1]
    elif values.ndim == 1:
        values = values.reshape(1, -1)

    if values.ndim != 2:
        raise ValueError(f"Unexpected SHAP value shape: {values.shape}")

    if values.shape[1] != explain_data.shape[1]:
        raise ValueError(
            f"SHAP feature count mismatch: shap={values.shape[1]}, data={explain_data.shape[1]}"
        )

    logger.info(f"Computed SHAP matrix shape: {values.shape}")
    return values


# ---------------------------------------------------
# Global importance
# ---------------------------------------------------

def compute_global_shap_importance(
    shap_matrix: np.ndarray,
    feature_names: List[str]
) -> pd.DataFrame:
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": np.abs(shap_matrix).mean(axis=0),
        "mean_shap": shap_matrix.mean(axis=0)
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    importance_df.to_csv(GLOBAL_FEATURE_IMPACT_CSV_PATH, index=False)
    logger.info(f"Saved global SHAP impact CSV: {GLOBAL_FEATURE_IMPACT_CSV_PATH}")

    return importance_df


# ---------------------------------------------------
# Plots
# ---------------------------------------------------

def save_shap_summary_plot(shap_matrix: np.ndarray, explain_data: pd.DataFrame) -> None:
    plt.figure()
    shap.summary_plot(
        shap_matrix,
        explain_data,
        show=False,
        max_display=TOP_N_FEATURES
    )
    plt.tight_layout()
    plt.savefig(SHAP_SUMMARY_PNG_PATH, dpi=300, bbox_inches="tight")
    plt.close("all")
    logger.info(f"Saved SHAP summary plot: {SHAP_SUMMARY_PNG_PATH}")


def save_shap_bar_plot(shap_matrix: np.ndarray, explain_data: pd.DataFrame) -> None:
    plt.figure()
    shap.summary_plot(
        shap_matrix,
        explain_data,
        plot_type="bar",
        show=False,
        max_display=TOP_N_FEATURES
    )
    plt.tight_layout()
    plt.savefig(SHAP_BAR_PNG_PATH, dpi=300, bbox_inches="tight")
    plt.close("all")
    logger.info(f"Saved SHAP bar plot: {SHAP_BAR_PNG_PATH}")


# ---------------------------------------------------
# Local explanations
# ---------------------------------------------------

def build_local_explanations_json(
    shap_matrix: np.ndarray,
    explain_data: pd.DataFrame,
    model: Any,
    threshold: float
) -> Dict[str, Any]:
    probabilities = model.predict_proba(explain_data)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    records = []
    num_samples = min(LOCAL_EXPLANATION_SAMPLES, len(explain_data))

    for i in range(num_samples):
        row_values = shap_matrix[i]
        row_features = explain_data.iloc[i]

        local_df = pd.DataFrame({
            "feature": explain_data.columns,
            "feature_value": row_features.values,
            "shap_value": row_values,
            "abs_shap_value": np.abs(row_values)
        }).sort_values("abs_shap_value", ascending=False).head(LOCAL_TOP_FEATURES)

        top_feature_rows = []
        for _, row in local_df.iterrows():
            top_feature_rows.append({
                "feature": str(row["feature"]),
                "feature_value": to_serializable(row["feature_value"]),
                "shap_value": to_serializable(row["shap_value"]),
                "abs_shap_value": to_serializable(row["abs_shap_value"]),
                "direction": "increase_malignancy_risk" if row["shap_value"] >= 0 else "decrease_malignancy_risk"
            })

        records.append({
            "sample_index": int(explain_data.index[i]),
            "predicted_probability": float(probabilities[i]),
            "predicted_class": int(predictions[i]),
            "top_feature_contributions": top_feature_rows
        })

    return {
        "dataset_name": DATASET_NAME,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "threshold": float(threshold),
        "explained_sample_count": int(num_samples),
        "top_features_per_sample": LOCAL_TOP_FEATURES,
        "samples": records
    }


# ---------------------------------------------------
# Explainability summary
# ---------------------------------------------------

def build_explainability_summary_json(
    global_importance_df: pd.DataFrame,
    explain_data: pd.DataFrame,
    threshold: float,
    metadata: Dict[str, Any]
) -> Dict[str, Any]:
    top_global_features = []
    for _, row in global_importance_df.head(TOP_N_FEATURES).iterrows():
        top_global_features.append({
            "feature": str(row["feature"]),
            "mean_abs_shap": to_serializable(row["mean_abs_shap"]),
            "mean_shap": to_serializable(row["mean_shap"])
        })

    return {
        "dataset_name": DATASET_NAME,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "analysis_type": "shap_explainability",
        "target_column": TARGET_COLUMN,
        "threshold": float(threshold),
        "background_sample_size": int(min(BACKGROUND_SAMPLE_SIZE, len(explain_data))),
        "explanation_sample_size": int(len(explain_data)),
        "top_n_features_visualized": TOP_N_FEATURES,
        "global_feature_count": int(len(global_importance_df)),
        "selected_feature_count": int(metadata.get("selected_features", {}).get("selected_feature_count", len(global_importance_df))),
        "model_name": metadata.get("model_metadata", {}).get("model_name", "Unknown"),
        "top_global_features": top_global_features,
        "artifacts": {
            "shap_summary_png": SHAP_SUMMARY_PNG_PATH,
            "shap_bar_png": SHAP_BAR_PNG_PATH,
            "global_feature_impact_csv": GLOBAL_FEATURE_IMPACT_CSV_PATH,
            "local_explanations_json": LOCAL_EXPLANATIONS_JSON_PATH,
            "explainability_summary_json": EXPLAINABILITY_SUMMARY_JSON_PATH
        },
        "notes": {
            "interpretation": "Positive SHAP values push prediction toward malignant class, while negative SHAP values push toward benign class.",
            "clinical_warning": "These explanations improve transparency but must not be treated as standalone clinical evidence."
        }
    }


# ---------------------------------------------------
# Main pipeline
# ---------------------------------------------------

def run_shap_pipeline() -> None:
    logger.info("Starting breast cancer SHAP explainability pipeline...")

    X, y = load_data()
    X_train, _, X_test, _, _, _ = recreate_split(X, y)

    model, preprocessor_artifact, threshold, selected_features, metadata = load_registry_artifacts()

    X_train_ready = transform_features(X_train, preprocessor_artifact, selected_features)
    X_test_ready = transform_features(X_test, preprocessor_artifact, selected_features)

    background_data = sample_background_data(X_train_ready)
    explain_data = sample_explanation_data(X_test_ready)

    explainer = build_shap_explainer(model, background_data)
    shap_matrix = compute_shap_matrix(explainer, explain_data)

    global_importance_df = compute_global_shap_importance(
        shap_matrix=shap_matrix,
        feature_names=list(explain_data.columns)
    )

    save_shap_summary_plot(shap_matrix, explain_data)
    save_shap_bar_plot(shap_matrix, explain_data)

    local_explanations_json = build_local_explanations_json(
        shap_matrix=shap_matrix,
        explain_data=explain_data,
        model=model,
        threshold=threshold
    )
    save_json(local_explanations_json, LOCAL_EXPLANATIONS_JSON_PATH)

    explainability_summary_json = build_explainability_summary_json(
        global_importance_df=global_importance_df,
        explain_data=explain_data,
        threshold=threshold,
        metadata=metadata
    )
    save_json(explainability_summary_json, EXPLAINABILITY_SUMMARY_JSON_PATH)

    logger.info("Top SHAP global features:")
    for idx, row in enumerate(global_importance_df.head(10).itertuples(index=False), start=1):
        logger.info(
            f"{idx}. {row.feature} | mean_abs_shap={row.mean_abs_shap:.6f} | mean_shap={row.mean_shap:.6f}"
        )

    logger.info(f"Saved SHAP summary plot: {SHAP_SUMMARY_PNG_PATH}")
    logger.info(f"Saved SHAP bar plot: {SHAP_BAR_PNG_PATH}")
    logger.info(f"Saved global impact CSV: {GLOBAL_FEATURE_IMPACT_CSV_PATH}")
    logger.info(f"Saved local explanations JSON: {LOCAL_EXPLANATIONS_JSON_PATH}")
    logger.info(f"Saved explainability summary JSON: {EXPLAINABILITY_SUMMARY_JSON_PATH}")
    logger.info("Breast cancer SHAP explainability pipeline completed successfully.")


if __name__ == "__main__":
    run_shap_pipeline()
