from __future__ import annotations

import os
import json
import pickle
import logging
import warnings
from datetime import datetime
from typing import Dict, Tuple, Any, List

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.inspection import permutation_importance
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
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

DATASET_NAME = "diabetes"
TARGET_COLUMN = "Outcome"
RANDOM_STATE = 42
TEST_SIZE = 0.15
TOP_N_FEATURES = 15

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
# Constants
# ---------------------------------------------------

BASE_NUMERICAL_FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age"
]

REQUIRED_COLUMNS = BASE_NUMERICAL_FEATURES + [TARGET_COLUMN]


# ---------------------------------------------------
# Utility Helpers
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
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(make_json_safe(data), f, indent=4, ensure_ascii=False)
    logger.info(f"Saved JSON: {path}")


def load_json_if_exists(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_importance(values: pd.Series) -> pd.Series:
    values = values.copy().fillna(0.0)
    values = values.clip(lower=0.0)

    total = values.sum()
    if total <= 0:
        return pd.Series(np.zeros(len(values)), index=values.index)

    return (values / total) * 100.0


# ---------------------------------------------------
# Feature Engineering
# ---------------------------------------------------

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Must match train.py exactly.
    """
    df = df.copy()
    eps = 1e-6

    # Existing
    df["Glucose_BMI"] = df["Glucose"] * df["BMI"]
    df["Glucose_Age"] = df["Glucose"] * df["Age"]
    df["BMI_Age"] = df["BMI"] * df["Age"]
    df["Insulin_Glucose_Ratio"] = df["Insulin"] / (df["Glucose"] + eps)
    df["DPF_Age_Interaction"] = df["DiabetesPedigreeFunction"] * df["Age"]
    df["Metabolic_Load"] = (df["Glucose"] + df["BMI"] + df["Insulin"]) / 3.0

    # 🔥 NEW POWERFUL FEATURES

    df["High_Glucose"] = (df["Glucose"] > 140).astype(int)
    df["Very_High_Glucose"] = (df["Glucose"] > 180).astype(int)

    df["Obese"] = (df["BMI"] > 30).astype(int)
    df["Severely_Obese"] = (df["BMI"] > 35).astype(int)

    df["High_Risk_Age"] = (df["Age"] > 45).astype(int)

    df["Glucose_Insulin_Product"] = df["Glucose"] * df["Insulin"]

    return df


# ---------------------------------------------------
# Dataset Loading
# ---------------------------------------------------

def validate_dataset(df: pd.DataFrame) -> None:
    if df.empty:
        raise ValueError("Dataset is empty.")

    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    if df.isnull().sum().sum() != 0:
        raise ValueError("Processed dataset contains missing values.")

    unique_targets = sorted(df[TARGET_COLUMN].dropna().unique().tolist())
    if unique_targets != [0, 1]:
        raise ValueError(f"Unexpected target values: {unique_targets}")


def load_dataset() -> pd.DataFrame:
    logger.info("Loading cleaned diabetes dataset...")

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    validate_dataset(df)

    df = df.copy()
    features = df.drop(columns=[TARGET_COLUMN])
    features = feature_engineering(features)
    df = pd.concat([features, df[[TARGET_COLUMN]].copy()], axis=1)

    logger.info(f"Dataset loaded successfully. Shape after engineering: {df.shape}")
    return df


# ---------------------------------------------------
# Registry Loaders
# ---------------------------------------------------

def load_registry_artifacts() -> Tuple[Any, Dict[str, Any], List[str], Dict[str, Any]]:
    logger.info("Loading model registry artifacts...")

    for path in [MODEL_PATH, PREPROCESSOR_PATH, SELECTED_FEATURES_PATH]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required artifact not found: {path}")

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    with open(PREPROCESSOR_PATH, "rb") as f:
        preprocessor = pickle.load(f)

    with open(SELECTED_FEATURES_PATH, "r", encoding="utf-8") as f:
        selected_payload = json.load(f)

    selected_features = selected_payload.get("selected_features", [])
    if not isinstance(selected_features, list) or len(selected_features) == 0:
        raise ValueError("selected_features.json does not contain a valid selected_features list.")

    metadata = {
        "model_metadata": load_json_if_exists(MODEL_METADATA_PATH),
        "training_summary": load_json_if_exists(TRAINING_SUMMARY_PATH),
        "training_config": load_json_if_exists(TRAINING_CONFIG_PATH),
    }

    logger.info(f"Loaded selected feature count: {len(selected_features)}")
    return model, preprocessor, selected_features, metadata


# ---------------------------------------------------
# Recreate Holdout Split
# ---------------------------------------------------

def create_holdout_split(
    df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE
    )

    logger.info(f"Recreated train shape: {X_train.shape}")
    logger.info(f"Recreated test shape: {X_test.shape}")

    return X_train, X_test, y_train, y_test


# ---------------------------------------------------
# Saved Preprocessing Transform
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

    logger.info(f"Selected feature matrix shape: {X_selected_df.shape}")
    return X_selected_df


# ---------------------------------------------------
# Importance Computation
# ---------------------------------------------------

def train_extra_trees_importance_model(
    X: pd.DataFrame,
    y: pd.Series
) -> ExtraTreesClassifier:
    logger.info("Training ExtraTrees model for impurity-based importance...")

    model = ExtraTreesClassifier(
        n_estimators=500,
        max_depth=8,
        min_samples_split=6,
        min_samples_leaf=3,
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
    logger.info("Computing permutation importance on transformed holdout features...")

    result = permutation_importance(
        estimator=model,
        X=X,
        y=y,
        scoring="roc_auc",
        n_repeats=20,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    df = pd.DataFrame({
        "feature": feature_names,
        "permutation_importance_mean": result.importances_mean,
        "permutation_importance_std": result.importances_std
    })
    df["importance_percent"] = normalize_importance(df["permutation_importance_mean"]).round(4)

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
    plot_df = merged_df.head(top_n).copy().sort_values("permutation_importance_mean", ascending=True)

    plt.figure(figsize=(10, 6))
    plt.barh(plot_df["feature"], plot_df["permutation_importance_mean"])
    plt.xlabel("Permutation Importance Mean (ROC-AUC)")
    plt.ylabel("Feature")
    plt.title(f"Top {min(top_n, len(plot_df))} Feature Importances - Diabetes")
    plt.tight_layout()
    plt.savefig(FEATURE_IMPORTANCE_PNG_PATH, dpi=300, bbox_inches="tight")
    plt.close("all")

    logger.info(f"Feature importance plot saved: {FEATURE_IMPORTANCE_PNG_PATH}")


def build_feature_importance_json(
    merged_df: pd.DataFrame,
    metadata: Dict[str, Any]
) -> Dict[str, Any]:
    features_payload = []
    for _, row in merged_df.iterrows():
        features_payload.append({
            "feature": row["feature"],
            "impurity_importance": to_serializable(row["impurity_importance"]),
            "permutation_importance_mean": to_serializable(row["permutation_importance_mean"]),
            "permutation_importance_std": to_serializable(row["permutation_importance_std"]),
            "importance_percent": to_serializable(row["importance_percent"]),
            "impurity_rank": int(row["impurity_rank"]),
            "permutation_rank": int(row["permutation_rank"]),
            "average_rank": to_serializable(row["average_rank"]),
            "importance_direction": row["importance_direction"]
        })

    payload = {
        "dataset_name": DATASET_NAME,
        "analysis_type": "feature_importance",
        "method": {
            "primary": "permutation_importance",
            "secondary": "extra_trees_impurity_importance",
            "scoring_metric": "roc_auc",
            "n_repeats": 20
        },
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "target_column": TARGET_COLUMN,
        "model_name": metadata.get("model_metadata", {}).get("model_name", "Unknown"),
        "model_family": metadata.get("model_metadata", {}).get("model_family", "Unknown"),
        "selected_feature_count": metadata.get("model_metadata", {}).get("selected_features_count", len(merged_df)),
        "feature_count": int(len(merged_df)),
        "top_n_plot_features": int(min(TOP_N_FEATURES, len(merged_df))),
        "ranking_method": "average_rank_of_impurity_and_permutation_importance",
        "summary": {
            "top_5_features_by_average_rank": merged_df.head(5)["feature"].tolist(),
            "negative_or_unstable_permutation_features": merged_df.loc[
                merged_df["permutation_importance_mean"] < 0, "feature"
            ].tolist()
        },
        "features": features_payload,
        "artifacts": {
            "feature_importance_csv": FEATURE_IMPORTANCE_CSV_PATH,
            "feature_importance_json": FEATURE_IMPORTANCE_JSON_PATH,
            "feature_importance_png": FEATURE_IMPORTANCE_PNG_PATH
        }
    }

    return payload


# ---------------------------------------------------
# Main Pipeline
# ---------------------------------------------------

def run_feature_importance_pipeline() -> None:
    logger.info("Starting diabetes feature importance pipeline...")

    df = load_dataset()
    model, preprocessor, selected_features, metadata = load_registry_artifacts()

    _, X_test, _, y_test = create_holdout_split(df)

    X_test_selected = transform_with_saved_preprocessing(
        X=X_test,
        preprocessor_bundle=preprocessor,
        selected_features=selected_features
    )

    impurity_model = train_extra_trees_importance_model(X_test_selected, y_test)

    impurity_df = get_impurity_importance(
        model=impurity_model,
        feature_names=list(X_test_selected.columns)
    )

    permutation_df = get_permutation_importance(
        model=model,
        X=X_test_selected,
        y=y_test,
        feature_names=list(X_test_selected.columns)
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

    logger.info("Top feature importance results:")
    for idx, row in enumerate(merged_df.head(10).itertuples(index=False), start=1):
        logger.info(
            f"{idx}. {row.feature} | perm_mean={row.permutation_importance_mean:.6f} | "
            f"perm_std={row.permutation_importance_std:.6f} | "
            f"impurity={row.impurity_importance:.6f} | avg_rank={row.average_rank:.2f}"
        )

    logger.info(f"Feature importance artifacts saved in: {FEATURE_IMPORTANCE_DIR}")
    logger.info("Diabetes feature importance pipeline completed successfully.")


if __name__ == "__main__":
    run_feature_importance_pipeline()
