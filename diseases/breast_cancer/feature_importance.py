from __future__ import annotations

import os
import json
import pickle
import logging
import warnings
from datetime import datetime
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.inspection import permutation_importance
from sklearn.ensemble import ExtraTreesClassifier
try:
    from services.preprocessing_utils import safe_log1p_array
except ModuleNotFoundError:
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
    from services.preprocessing_utils import safe_log1p_array


warnings.filterwarnings("ignore")


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

DATASET_NAME = "breast_cancer"
TARGET_COLUMN = "target"
RANDOM_STATE = 42
TOP_N_FEATURES = 20

DATA_PATH = os.path.join(BASE_DIR, "data", "processed", f"{DATASET_NAME}_clean.csv")

TRAIN_REGISTRY_DIR = os.path.join(BASE_DIR, "model_registry", DATASET_NAME, "train")
FEATURE_IMPORTANCE_DIR = os.path.join(BASE_DIR, "model_registry", DATASET_NAME, "feature_importance")

MODEL_PATH = os.path.join(TRAIN_REGISTRY_DIR, "model.pkl")
PREPROCESSOR_PATH = os.path.join(TRAIN_REGISTRY_DIR, "preprocessor.pkl")
SELECTED_FEATURES_PATH = os.path.join(TRAIN_REGISTRY_DIR, "selected_features.json")
MODEL_METADATA_PATH = os.path.join(TRAIN_REGISTRY_DIR, "model_metadata.json")
TRAINING_SUMMARY_PATH = os.path.join(TRAIN_REGISTRY_DIR, "training_summary.json")
TRAINING_CONFIG_PATH = os.path.join(TRAIN_REGISTRY_DIR, "training_config.json")

FEATURE_IMPORTANCE_CSV_PATH = os.path.join(FEATURE_IMPORTANCE_DIR, "feature_importance.csv")
FEATURE_IMPORTANCE_JSON_PATH = os.path.join(FEATURE_IMPORTANCE_DIR, "feature_importance.json")
FEATURE_IMPORTANCE_PNG_PATH = os.path.join(FEATURE_IMPORTANCE_DIR, "feature_importance.png")

os.makedirs(FEATURE_IMPORTANCE_DIR, exist_ok=True)


# ---------------------------------------------------
# Serialization Helpers
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


def load_json_if_exists(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------
# Feature Engineering
# ---------------------------------------------------

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Must match train.py exactly.
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
# Dataset Loading
# ---------------------------------------------------

def validate_dataset(df: pd.DataFrame) -> None:
    if df.empty:
        raise ValueError("Dataset is empty.")

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Missing target column: {TARGET_COLUMN}")

    if df.isnull().sum().sum() != 0:
        raise ValueError("Processed dataset contains missing values.")

    unique_targets = sorted(df[TARGET_COLUMN].dropna().unique().tolist())
    if unique_targets != [0, 1]:
        raise ValueError(f"Unexpected target values: {unique_targets}")


def load_data() -> Tuple[pd.DataFrame, pd.Series]:
    logger.info("Loading processed breast cancer dataset...")

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    validate_dataset(df)

    X = df.drop(columns=[TARGET_COLUMN]).copy()
    y = df[TARGET_COLUMN].astype(int).copy()

    X = feature_engineering(X)

    logger.info(f"Dataset loaded successfully. Raw shape: {df.shape}")
    logger.info(f"Feature matrix shape after feature engineering: {X.shape}")

    return X, y


# ---------------------------------------------------
# Registry Artifact Loading
# ---------------------------------------------------

def load_registry_artifacts() -> Tuple[Any, Dict[str, Any], List[str], Dict[str, Any]]:
    logger.info("Loading model registry artifacts...")

    for path in [MODEL_PATH, PREPROCESSOR_PATH, SELECTED_FEATURES_PATH]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required artifact not found: {path}")

    with open(MODEL_PATH, "rb") as f:
        final_model = pickle.load(f)

    with open(PREPROCESSOR_PATH, "rb") as f:
        preprocessor_bundle = pickle.load(f)

    with open(SELECTED_FEATURES_PATH, "r", encoding="utf-8") as f:
        selected_features_payload = json.load(f)

    selected_features = selected_features_payload.get("selected_features", [])
    if not isinstance(selected_features, list) or len(selected_features) == 0:
        raise ValueError("selected_features.json does not contain a valid selected_features list.")

    metadata = {
        "model_metadata": load_json_if_exists(MODEL_METADATA_PATH),
        "training_summary": load_json_if_exists(TRAINING_SUMMARY_PATH),
        "training_config": load_json_if_exists(TRAINING_CONFIG_PATH),
    }

    logger.info(f"Loaded selected feature count: {len(selected_features)}")
    return final_model, preprocessor_bundle, selected_features, metadata


# ---------------------------------------------------
# Transform with Saved Preprocessing
# ---------------------------------------------------

def transform_with_saved_preprocessing(
    X: pd.DataFrame,
    preprocessor_bundle: Dict[str, Any],
    selected_features: List[str]
) -> pd.DataFrame:
    logger.info("Applying saved preprocessing and feature selection...")

    if not isinstance(preprocessor_bundle, dict):
        raise ValueError("preprocessor.pkl must be a packaged dictionary artifact.")

    if "fitted_feature_pipeline" in preprocessor_bundle:
        fitted_pipeline = preprocessor_bundle["fitted_feature_pipeline"]
        X_selected = fitted_pipeline.transform(X)
    elif "sklearn_preprocessor" in preprocessor_bundle and "feature_selector" in preprocessor_bundle:
        sklearn_preprocessor = preprocessor_bundle["sklearn_preprocessor"]
        feature_selector = preprocessor_bundle["feature_selector"]
        X_processed = sklearn_preprocessor.transform(X)
        X_selected = feature_selector.transform(X_processed)
    else:
        raise ValueError(
            "Unsupported preprocessor artifact format. "
            "Expected 'fitted_feature_pipeline' or "
            "'sklearn_preprocessor' + 'feature_selector'."
        )

    X_selected = np.asarray(X_selected)
    if X_selected.ndim != 2:
        raise ValueError(f"Expected 2D transformed feature matrix, got shape: {X_selected.shape}")

    if X_selected.shape[1] != len(selected_features):
        raise ValueError(
            f"Mismatch between transformed columns ({X_selected.shape[1]}) "
            f"and selected feature names ({len(selected_features)})."
        )

    X_selected_df = pd.DataFrame(X_selected, columns=selected_features, index=X.index)

    logger.info(f"Preprocessing complete. Selected feature matrix shape: {X_selected_df.shape}")
    return X_selected_df


# ---------------------------------------------------
# Importance Models
# ---------------------------------------------------

def train_extra_trees_importance_model(
    X: pd.DataFrame,
    y: pd.Series
) -> ExtraTreesClassifier:
    logger.info("Training ExtraTrees model for impurity-based importance...")

    model = ExtraTreesClassifier(
        n_estimators=500,
        max_depth=8,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    model.fit(X, y)

    logger.info("ExtraTrees importance model trained successfully.")
    return model


def get_impurity_importance(
    model: ExtraTreesClassifier,
    feature_names: List[str]
) -> pd.DataFrame:
    df = pd.DataFrame({
        "feature": feature_names,
        "impurity_importance": model.feature_importances_
    })
    return df.sort_values("impurity_importance", ascending=False).reset_index(drop=True)


def get_permutation_importance(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    feature_names: List[str]
) -> pd.DataFrame:
    logger.info("Computing permutation importance using final trained model...")

    perm_result = permutation_importance(
        estimator=model,
        X=X,
        y=y,
        n_repeats=10,
        random_state=RANDOM_STATE,
        scoring="roc_auc",
        n_jobs=-1
    )

    df = pd.DataFrame({
        "feature": feature_names,
        "permutation_importance_mean": perm_result.importances_mean,
        "permutation_importance_std": perm_result.importances_std
    })

    logger.info("Permutation importance computed successfully.")
    return df.sort_values("permutation_importance_mean", ascending=False).reset_index(drop=True)


def merge_importance_reports(
    impurity_df: pd.DataFrame,
    permutation_df: pd.DataFrame
) -> pd.DataFrame:
    merged_df = impurity_df.merge(permutation_df, on="feature", how="inner")

    merged_df["impurity_rank"] = merged_df["impurity_importance"].rank(
        ascending=False,
        method="dense"
    )
    merged_df["permutation_rank"] = merged_df["permutation_importance_mean"].rank(
        ascending=False,
        method="dense"
    )
    merged_df["average_rank"] = (
        merged_df["impurity_rank"] + merged_df["permutation_rank"]
    ) / 2.0

    merged_df["importance_direction"] = np.where(
        merged_df["permutation_importance_mean"] < 0,
        "negative_or_unstable",
        "positive_or_neutral"
    )

    merged_df = merged_df.sort_values(
        by=["average_rank", "permutation_importance_mean", "impurity_importance"],
        ascending=[True, False, False]
    ).reset_index(drop=True)

    return merged_df


# ---------------------------------------------------
# Output Writers
# ---------------------------------------------------

def save_feature_importance_csv(merged_df: pd.DataFrame) -> None:
    merged_df.to_csv(FEATURE_IMPORTANCE_CSV_PATH, index=False)
    logger.info(f"Feature importance CSV saved: {FEATURE_IMPORTANCE_CSV_PATH}")


def save_feature_importance_plot(
    merged_df: pd.DataFrame,
    top_n: int = TOP_N_FEATURES
) -> None:
    plot_df = merged_df.head(top_n).copy()
    plot_df = plot_df.sort_values("permutation_importance_mean", ascending=True)

    plt.figure(figsize=(10, 7))
    plt.barh(plot_df["feature"], plot_df["permutation_importance_mean"])
    plt.xlabel("Permutation Importance Mean")
    plt.ylabel("Feature")
    plt.title(f"Top {min(top_n, len(plot_df))} Feature Importances - Breast Cancer")
    plt.tight_layout()
    plt.savefig(FEATURE_IMPORTANCE_PNG_PATH, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Feature importance plot saved: {FEATURE_IMPORTANCE_PNG_PATH}")


def build_feature_importance_json(
    merged_df: pd.DataFrame,
    metadata: Dict[str, Any]
) -> Dict[str, Any]:
    top_features = []
    for _, row in merged_df.iterrows():
        top_features.append({
            "feature": row["feature"],
            "impurity_importance": to_serializable(row["impurity_importance"]),
            "permutation_importance_mean": to_serializable(row["permutation_importance_mean"]),
            "permutation_importance_std": to_serializable(row["permutation_importance_std"]),
            "impurity_rank": int(row["impurity_rank"]),
            "permutation_rank": int(row["permutation_rank"]),
            "average_rank": to_serializable(row["average_rank"]),
            "importance_direction": row["importance_direction"]
        })

    return {
        "dataset_name": DATASET_NAME,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "target_column": TARGET_COLUMN,
        "feature_count": int(len(merged_df)),
        "top_n_plot_features": int(min(TOP_N_FEATURES, len(merged_df))),
        "ranking_method": "average_rank_of_impurity_and_permutation_importance",
        "primary_plot_metric": "permutation_importance_mean",
        "model_name": metadata.get("model_metadata", {}).get("model_name", "Unknown"),
        "model_family": metadata.get("model_metadata", {}).get("model_family", "Unknown"),
        "selected_feature_count": metadata.get("model_metadata", {}).get("selected_features_count", len(merged_df)),
        "summary": {
            "top_5_features_by_average_rank": merged_df.head(5)["feature"].tolist(),
            "negative_or_unstable_permutation_features": merged_df.loc[
                merged_df["permutation_importance_mean"] < 0, "feature"
            ].tolist()
        },
        "features": top_features,
        "artifacts": {
            "feature_importance_csv": FEATURE_IMPORTANCE_CSV_PATH,
            "feature_importance_png": FEATURE_IMPORTANCE_PNG_PATH,
            "feature_importance_json": FEATURE_IMPORTANCE_JSON_PATH
        }
    }


# ---------------------------------------------------
# Pipeline
# ---------------------------------------------------

def run_feature_importance_pipeline() -> None:
    logger.info("Starting breast cancer feature importance pipeline...")

    X, y = load_data()

    final_model, preprocessor_bundle, selected_features, metadata = load_registry_artifacts()

    X_selected = transform_with_saved_preprocessing(
        X=X,
        preprocessor_bundle=preprocessor_bundle,
        selected_features=selected_features
    )

    extra_trees_model = train_extra_trees_importance_model(X_selected, y)

    impurity_df = get_impurity_importance(
        model=extra_trees_model,
        feature_names=list(X_selected.columns)
    )

    permutation_df = get_permutation_importance(
        model=final_model,
        X=X_selected,
        y=y,
        feature_names=list(X_selected.columns)
    )

    merged_df = merge_importance_reports(
        impurity_df=impurity_df,
        permutation_df=permutation_df
    )

    save_feature_importance_csv(merged_df)
    save_feature_importance_plot(merged_df)

    feature_importance_payload = build_feature_importance_json(
        merged_df=merged_df,
        metadata=metadata
    )
    save_json(feature_importance_payload, FEATURE_IMPORTANCE_JSON_PATH)

    logger.info(f"Feature importance JSON saved: {FEATURE_IMPORTANCE_JSON_PATH}")
    logger.info("Breast cancer feature importance pipeline completed successfully.")


# ---------------------------------------------------
# Main
# ---------------------------------------------------

if __name__ == "__main__":
    run_feature_importance_pipeline()
