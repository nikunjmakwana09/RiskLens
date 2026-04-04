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

from sklearn.model_selection import train_test_split
try:
    from services.preprocessing_utils import safe_log1p_array
except ModuleNotFoundError:
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
    from services.preprocessing_utils import safe_log1p_array

warnings.filterwarnings("ignore")


# ---------------------------------------------------
# Optional SHAP Import
# ---------------------------------------------------

try:
    import shap
except ImportError as exc:
    raise ImportError(
        "The 'shap' package is required for shap_explainer.py. "
        "Install it using: pip install shap"
    ) from exc


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

DATASET_NAME = "heart"
TARGET_COLUMN = "target"
RANDOM_STATE = 42
TEST_SIZE = 0.15
VALID_SIZE = 0.15

DATA_PATH = os.path.join(BASE_DIR, "data", "processed", f"{DATASET_NAME}_clean.csv")

TRAIN_REGISTRY_DIR = os.path.join(BASE_DIR, "model_registry", DATASET_NAME, "train")
SHAP_REGISTRY_DIR = os.path.join(BASE_DIR, "model_registry", DATASET_NAME, "shap")

MODEL_PATH = os.path.join(TRAIN_REGISTRY_DIR, "model.pkl")
PREPROCESSOR_PATH = os.path.join(TRAIN_REGISTRY_DIR, "preprocessor.pkl")
SELECTED_FEATURES_PATH = os.path.join(TRAIN_REGISTRY_DIR, "selected_features.json")
THRESHOLD_PATH = os.path.join(TRAIN_REGISTRY_DIR, "threshold.json")
TRAINING_CONFIG_PATH = os.path.join(TRAIN_REGISTRY_DIR, "training_config.json")
TRAINING_SUMMARY_PATH = os.path.join(TRAIN_REGISTRY_DIR, "training_summary.json")
MODEL_METADATA_PATH = os.path.join(TRAIN_REGISTRY_DIR, "model_metadata.json")

SHAP_SUMMARY_PATH = os.path.join(SHAP_REGISTRY_DIR, "shap_summary.png")
SHAP_BAR_PATH = os.path.join(SHAP_REGISTRY_DIR, "shap_bar.png")
GLOBAL_FEATURE_IMPACT_PATH = os.path.join(SHAP_REGISTRY_DIR, "global_feature_impact.csv")
LOCAL_EXPLANATIONS_PATH = os.path.join(SHAP_REGISTRY_DIR, "local_explanations.json")
EXPLAINABILITY_SUMMARY_PATH = os.path.join(SHAP_REGISTRY_DIR, "explainability_summary.json")

os.makedirs(SHAP_REGISTRY_DIR, exist_ok=True)


# ---------------------------------------------------
# Base feature groups
# ---------------------------------------------------

BASE_NUMERICAL_FEATURES = [
    "age",
    "trestbps",
    "chol",
    "thalach",
    "oldpeak"
]

BASE_CATEGORICAL_FEATURES = [
    "sex",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "ca",
    "thal"
]

REQUIRED_COLUMNS = BASE_NUMERICAL_FEATURES + BASE_CATEGORICAL_FEATURES + [TARGET_COLUMN]


# ---------------------------------------------------
# SHAP constants
# ---------------------------------------------------

BACKGROUND_SAMPLE_SIZE = 80
EXPLANATION_SAMPLE_SIZE = 100
TOP_N_FEATURES = 20
LOCAL_EXPLANATION_SAMPLES = 5
LOCAL_TOP_FEATURES = 10


# ---------------------------------------------------
# Helpers
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


def extract_threshold(payload: Dict[str, Any]) -> float:
    candidate_keys = ["selected_threshold", "threshold", "optimal_threshold"]
    for key in candidate_keys:
        if key in payload and payload[key] is not None:
            return float(payload[key])

    nested_candidates = [
        ("selected_metrics", "threshold"),
        ("validation_metrics_at_optimal_threshold", "threshold")
    ]
    for outer, inner in nested_candidates:
        if outer in payload and isinstance(payload[outer], dict) and inner in payload[outer]:
            return float(payload[outer][inner])

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

    df["age_thalach_ratio"] = df["age"] / (df["thalach"] + eps)
    df["chol_age_ratio"] = df["chol"] / (df["age"] + eps)
    df["bp_age_ratio"] = df["trestbps"] / (df["age"] + eps)
    df["oldpeak_thalach_ratio"] = df["oldpeak"] / (df["thalach"] + eps)

    return df


# ---------------------------------------------------
# Data loading
# ---------------------------------------------------

def load_dataset() -> Tuple[pd.DataFrame, pd.Series]:
    logger.info("Loading cleaned heart dataset...")

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    if df.empty:
        raise ValueError("Dataset is empty.")

    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    if df.isnull().sum().sum() != 0:
        raise ValueError("Processed dataset contains missing values.")

    X = df.drop(columns=[TARGET_COLUMN]).copy()
    y = df[TARGET_COLUMN].astype(int).copy()

    X = feature_engineering(X)

    logger.info(f"Dataset loaded successfully. Shape: {df.shape}")
    logger.info(f"Feature shape after engineering: {X.shape}")
    logger.info(f"Target distribution: {y.value_counts().to_dict()}")

    return X, y


# ---------------------------------------------------
# Recreate exact split
# ---------------------------------------------------

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
# Load registry artifacts
# ---------------------------------------------------

def load_registry_artifacts() -> Tuple[Any, Dict[str, Any], float, List[str], Dict[str, Any]]:
    logger.info("Loading heart training registry artifacts...")

    required_paths = [
        MODEL_PATH,
        PREPROCESSOR_PATH,
        SELECTED_FEATURES_PATH,
        THRESHOLD_PATH
    ]
    for path in required_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required artifact not found: {path}")

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    with open(PREPROCESSOR_PATH, "rb") as f:
        preprocessor_artifact = pickle.load(f)

    if not isinstance(preprocessor_artifact, dict):
        raise ValueError("preprocessor.pkl must be a packaged dictionary artifact.")

    if "fitted_feature_pipeline" not in preprocessor_artifact:
        raise ValueError("Missing 'fitted_feature_pipeline' in preprocessor artifact.")

    selected_payload = load_json(SELECTED_FEATURES_PATH)
    selected_features = selected_payload.get("selected_features", [])
    if not selected_features:
        raise ValueError("selected_features.json does not contain selected_features.")

    threshold_payload = load_json(THRESHOLD_PATH)
    threshold = extract_threshold(threshold_payload)

    metadata = {
        "selected_features_payload": selected_payload,
        "threshold_payload": threshold_payload,
        "training_config": load_json_if_exists(TRAINING_CONFIG_PATH),
        "training_summary": load_json_if_exists(TRAINING_SUMMARY_PATH),
        "model_metadata": load_json_if_exists(MODEL_METADATA_PATH)
    }

    logger.info(f"Loaded threshold: {threshold:.4f}")
    logger.info(f"Loaded selected feature count: {len(selected_features)}")

    return model, preprocessor_artifact, threshold, selected_features, metadata


# ---------------------------------------------------
# Transform features using saved fitted pipeline
# ---------------------------------------------------

def transform_features(
    X: pd.DataFrame,
    preprocessor_artifact: Dict[str, Any],
    selected_features: List[str]
) -> pd.DataFrame:
    logger.info("Applying saved preprocessing contract...")

    fitted_pipeline = preprocessor_artifact["fitted_feature_pipeline"]
    Xt = fitted_pipeline.transform(X)
    Xt = np.asarray(Xt)

    if Xt.shape[1] != len(selected_features):
        raise ValueError(
            f"Transformed feature count mismatch: got {Xt.shape[1]}, expected {len(selected_features)}"
        )

    return pd.DataFrame(Xt, columns=selected_features, index=X.index)


# ---------------------------------------------------
# SHAP explainers
# ---------------------------------------------------

def positive_class_predict_fn(model: Any):
    def _predict_fn(X_array):
        X_array = np.asarray(X_array, dtype=float)
        return model.predict_proba(X_array)[:, 1]
    return _predict_fn


def build_shap_explainer(model: Any, background_df: pd.DataFrame):
    logger.info("Building SHAP explainer...")

    predict_fn = positive_class_predict_fn(model)

    try:
        explainer = shap.Explainer(
            predict_fn,
            background_df,
            feature_names=list(background_df.columns)
        )
        logger.info("Using shap.Explainer on positive-class probability.")
        return explainer
    except Exception as exc:
        logger.warning(f"shap.Explainer failed: {exc}")

    try:
        explainer = shap.KernelExplainer(
            predict_fn,
            background_df
        )
        logger.info("Using SHAP KernelExplainer fallback.")
        return explainer
    except Exception as exc:
        raise RuntimeError(f"Failed to build SHAP explainer: {exc}") from exc


def compute_shap_matrix(explainer: Any, explain_df: pd.DataFrame) -> np.ndarray:
    logger.info("Computing SHAP values...")

    try:
        shap_values = explainer(explain_df)
        values = getattr(shap_values, "values", shap_values)
    except Exception:
        shap_values = explainer.shap_values(explain_df)
        values = shap_values

    values = np.asarray(values)

    if values.ndim == 3:
        values = values[:, :, 1]
    elif values.ndim == 1:
        values = values.reshape(1, -1)

    if values.ndim != 2:
        raise ValueError(f"Unexpected SHAP value shape: {values.shape}")

    if values.shape[1] != explain_df.shape[1]:
        raise ValueError(
            f"SHAP feature mismatch: shap={values.shape[1]}, data={explain_df.shape[1]}"
        )

    logger.info(f"Computed SHAP matrix shape: {values.shape}")
    return values


# ---------------------------------------------------
# Sampling
# ---------------------------------------------------

def sample_background_and_explain_data(
    X_train_ready: pd.DataFrame,
    X_test_ready: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    background_size = min(BACKGROUND_SAMPLE_SIZE, len(X_train_ready))
    explain_size = min(EXPLANATION_SAMPLE_SIZE, len(X_test_ready))

    background_df = X_train_ready.sample(
        n=background_size,
        random_state=RANDOM_STATE
    )

    explain_df = X_test_ready.sample(
        n=explain_size,
        random_state=RANDOM_STATE
    )

    logger.info(f"Background sample shape: {background_df.shape}")
    logger.info(f"Explanation sample shape: {explain_df.shape}")

    return background_df, explain_df


# ---------------------------------------------------
# Global feature impact
# ---------------------------------------------------

def compute_global_feature_impact(
    shap_matrix: np.ndarray,
    explain_df: pd.DataFrame
) -> pd.DataFrame:
    impact_df = pd.DataFrame({
        "feature": explain_df.columns,
        "mean_abs_shap": np.abs(shap_matrix).mean(axis=0),
        "mean_shap": shap_matrix.mean(axis=0)
    }).sort_values(by="mean_abs_shap", ascending=False).reset_index(drop=True)

    total_impact = float(impact_df["mean_abs_shap"].sum())
    if total_impact > 0:
        impact_df["impact_percent"] = (impact_df["mean_abs_shap"] / total_impact * 100.0).round(4)
    else:
        impact_df["impact_percent"] = 0.0

    impact_df.to_csv(GLOBAL_FEATURE_IMPACT_PATH, index=False)
    logger.info(f"Global feature impact saved to: {GLOBAL_FEATURE_IMPACT_PATH}")
    return impact_df


# ---------------------------------------------------
# Plots
# ---------------------------------------------------

def save_shap_summary_plot(shap_matrix: np.ndarray, explain_df: pd.DataFrame) -> None:
    plt.figure()
    shap.summary_plot(
        shap_matrix,
        explain_df,
        show=False,
        max_display=TOP_N_FEATURES
    )
    plt.tight_layout()
    plt.savefig(SHAP_SUMMARY_PATH, dpi=300, bbox_inches="tight")
    plt.close("all")
    logger.info(f"SHAP summary plot saved to: {SHAP_SUMMARY_PATH}")


def save_shap_bar_plot(shap_matrix: np.ndarray, explain_df: pd.DataFrame) -> None:
    plt.figure()
    shap.summary_plot(
        shap_matrix,
        explain_df,
        plot_type="bar",
        show=False,
        max_display=TOP_N_FEATURES
    )
    plt.tight_layout()
    plt.savefig(SHAP_BAR_PATH, dpi=300, bbox_inches="tight")
    plt.close("all")
    logger.info(f"SHAP bar plot saved to: {SHAP_BAR_PATH}")


# ---------------------------------------------------
# Local explanations
# ---------------------------------------------------

def build_local_explanations(
    shap_matrix: np.ndarray,
    explain_df: pd.DataFrame,
    model: Any,
    threshold: float
) -> Dict[str, Any]:
    probabilities = model.predict_proba(explain_df)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    records = []
    num_samples = min(LOCAL_EXPLANATION_SAMPLES, len(explain_df))

    for i in range(num_samples):
        row_values = shap_matrix[i]
        row_features = explain_df.iloc[i]

        contrib_df = pd.DataFrame({
            "feature": explain_df.columns,
            "feature_value": row_features.values,
            "shap_value": row_values,
            "abs_shap_value": np.abs(row_values)
        }).sort_values(by="abs_shap_value", ascending=False).head(LOCAL_TOP_FEATURES)

        top_rows = []
        for _, row in contrib_df.iterrows():
            top_rows.append({
                "feature": str(row["feature"]),
                "feature_value": to_serializable(row["feature_value"]),
                "shap_value": to_serializable(row["shap_value"]),
                "abs_shap_value": to_serializable(row["abs_shap_value"]),
                "direction": "increase_heart_disease_risk" if row["shap_value"] >= 0 else "decrease_heart_disease_risk"
            })

        records.append({
            "sample_index": int(explain_df.index[i]),
            "predicted_probability": float(probabilities[i]),
            "predicted_class": int(predictions[i]),
            "top_feature_contributions": top_rows
        })

    payload = {
        "dataset_name": DATASET_NAME,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "threshold": float(threshold),
        "explained_sample_count": int(num_samples),
        "top_features_per_sample": LOCAL_TOP_FEATURES,
        "samples": records
    }

    save_json(payload, LOCAL_EXPLANATIONS_PATH)
    logger.info(f"Local explanations saved to: {LOCAL_EXPLANATIONS_PATH}")
    return payload


# ---------------------------------------------------
# Summary artifact
# ---------------------------------------------------

def build_explainability_summary(
    model: Any,
    global_impact_df: pd.DataFrame,
    explain_df: pd.DataFrame,
    threshold: float,
    metadata: Dict[str, Any]
) -> Dict[str, Any]:
    summary = {
        "dataset_name": DATASET_NAME,
        "analysis_type": "shap_explainability",
        "model_name": metadata.get("model_metadata", {}).get("model_name", model.__class__.__name__),
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "threshold": float(threshold),
        "explanation_sample_size": int(explain_df.shape[0]),
        "feature_count": int(explain_df.shape[1]),
        "top_global_features": make_json_safe(global_impact_df.head(15).to_dict(orient="records")),
        "local_explanations_file": os.path.basename(LOCAL_EXPLANATIONS_PATH),
        "global_feature_impact_file": os.path.basename(GLOBAL_FEATURE_IMPACT_PATH),
        "plots": {
            "shap_summary": os.path.basename(SHAP_SUMMARY_PATH),
            "shap_bar": os.path.basename(SHAP_BAR_PATH)
        },
        "notes": {
            "interpretation": "Positive SHAP values push prediction toward heart disease class, while negative SHAP values push prediction toward no-disease class.",
            "clinical_warning": "These explanations improve transparency but must not be used as standalone medical evidence."
        }
    }

    save_json(summary, EXPLAINABILITY_SUMMARY_PATH)
    logger.info(f"Explainability summary saved to: {EXPLAINABILITY_SUMMARY_PATH}")
    return summary


# ---------------------------------------------------
# Main pipeline
# ---------------------------------------------------

def run_shap_pipeline() -> None:
    logger.info("Starting heart SHAP explanation pipeline...")

    X, y = load_dataset()
    X_train, _, X_test, _, _, _ = recreate_split(X, y)

    model, preprocessor_artifact, threshold, selected_features, metadata = load_registry_artifacts()

    X_train_ready = transform_features(X_train, preprocessor_artifact, selected_features)
    X_test_ready = transform_features(X_test, preprocessor_artifact, selected_features)

    background_df, explain_df = sample_background_and_explain_data(X_train_ready, X_test_ready)

    explainer = build_shap_explainer(model, background_df)
    shap_matrix = compute_shap_matrix(explainer, explain_df)

    global_impact_df = compute_global_feature_impact(shap_matrix, explain_df)

    save_shap_summary_plot(shap_matrix, explain_df)
    save_shap_bar_plot(shap_matrix, explain_df)

    build_local_explanations(
        shap_matrix=shap_matrix,
        explain_df=explain_df,
        model=model,
        threshold=threshold
    )

    build_explainability_summary(
        model=model,
        global_impact_df=global_impact_df,
        explain_df=explain_df,
        threshold=threshold,
        metadata=metadata
    )

    logger.info("Top SHAP global features:")
    for idx, row in enumerate(global_impact_df.head(10).itertuples(index=False), start=1):
        logger.info(
            f"{idx}. {row.feature} | mean_abs_shap={row.mean_abs_shap:.6f} | mean_shap={row.mean_shap:.6f}"
        )

    logger.info("Heart SHAP explanation pipeline completed successfully.")


if __name__ == "__main__":
    run_shap_pipeline()
