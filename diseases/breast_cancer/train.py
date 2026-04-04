from __future__ import annotations

import os
import json
import pickle
import logging
import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Tuple, List, Any

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.calibration import CalibratedClassifierCV
try:
    from services.preprocessing_utils import safe_log1p_array
except ModuleNotFoundError:
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
    from services.preprocessing_utils import safe_log1p_array

warnings.filterwarnings("ignore")


# ---------------------------------------------------
# Optional External Models
# ---------------------------------------------------

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except Exception:
    CATBOOST_AVAILABLE = False


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
MODEL_REGISTRY_DIR = os.path.join(BASE_DIR, "model_registry", DATASET_NAME, "train")

MODEL_PKL_PATH = os.path.join(MODEL_REGISTRY_DIR, "model.pkl")
PREPROCESSOR_PKL_PATH = os.path.join(MODEL_REGISTRY_DIR, "preprocessor.pkl")
SELECTED_FEATURES_PATH = os.path.join(MODEL_REGISTRY_DIR, "selected_features.json")
THRESHOLD_PATH = os.path.join(MODEL_REGISTRY_DIR, "threshold.json")
TRAINING_CONFIG_PATH = os.path.join(MODEL_REGISTRY_DIR, "training_config.json")
TRAINING_SUMMARY_PATH = os.path.join(MODEL_REGISTRY_DIR, "training_summary.json")
MODEL_METADATA_PATH = os.path.join(MODEL_REGISTRY_DIR, "model_metadata.json")
TRAIN_METRICS_PATH = os.path.join(MODEL_REGISTRY_DIR, "train_metrics.json")

os.makedirs(MODEL_REGISTRY_DIR, exist_ok=True)


# ---------------------------------------------------
# Constants
# ---------------------------------------------------

RANDOM_STATE = 42
TEST_SIZE = 0.15
VALID_SIZE = 0.15
CV_FOLDS = 5

MIN_FEATURES_TO_KEEP = 10
MAX_FEATURES_TO_KEEP = 20

CALIBRATION_METHOD = "sigmoid"
CALIBRATION_CV = 5

THRESHOLD_MIN = 0.10
THRESHOLD_MAX = 0.90
THRESHOLD_STEP = 0.01


# ---------------------------------------------------
# Data Classes
# ---------------------------------------------------

@dataclass
class CandidateResult:
    model_name: str
    cv_roc_auc_mean: float
    cv_roc_auc_std: float
    cv_pr_auc_mean: float
    cv_f1_mean: float
    cv_recall_mean: float
    cv_balanced_accuracy_mean: float
    composite_score: float


# ---------------------------------------------------
# Utilities
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


def compute_class_distribution(y: pd.Series) -> Dict[str, int]:
    return {str(k): int(v) for k, v in y.value_counts().sort_index().to_dict().items()}


def validate_dataset(df: pd.DataFrame) -> None:
    if df.empty:
        raise ValueError("Dataset is empty.")

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Missing target column: {TARGET_COLUMN}")

    if df.isnull().sum().sum() != 0:
        raise ValueError("Processed dataset contains missing values. Training expects cleaned data.")

    unique_targets = sorted(df[TARGET_COLUMN].dropna().unique().tolist())
    if unique_targets != [0, 1]:
        raise ValueError(f"Unexpected target values: {unique_targets}")

    if df.duplicated().sum() > 0:
        logger.warning("Duplicate rows detected in processed dataset. Proceeding, but review data cleaning.")

    feature_df = df.drop(columns=[TARGET_COLUMN])

    if not all(np.issubdtype(dtype, np.number) for dtype in feature_df.dtypes):
        raise ValueError("Breast cancer training expects all feature columns to be numeric.")


# ---------------------------------------------------
# Load Data
# ---------------------------------------------------

def load_data() -> Tuple[pd.DataFrame, pd.Series]:
    logger.info("Loading processed breast cancer dataset...")

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    validate_dataset(df)

    X = df.drop(columns=[TARGET_COLUMN]).copy()
    y = df[TARGET_COLUMN].astype(int).copy()

    logger.info(f"Dataset loaded successfully. Shape: {df.shape}")
    logger.info(f"Feature matrix shape: {X.shape}")
    logger.info(f"Target distribution: {y.value_counts().to_dict()}")

    return X, y


# ---------------------------------------------------
# Feature Engineering
# ---------------------------------------------------

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Row-wise only features. No target usage, no leakage.
    Keep this conservative because the dataset already has strong signal.
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
# Split
# ---------------------------------------------------

def split_data(
    X: pd.DataFrame,
    y: pd.Series
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    logger.info("Performing stratified train/validation/test split...")

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
# Preprocessing
# ---------------------------------------------------

def identify_high_skew_features(X: pd.DataFrame, threshold: float = 1.0) -> List[str]:
    skewness = X.skew(numeric_only=True)
    high_skew = skewness[skewness.abs() > threshold].index.tolist()

    # Only log-transform strictly non-negative columns
    safe_high_skew = [col for col in high_skew if (X[col] >= 0).all()]
    return safe_high_skew


def build_preprocessor(
    feature_columns: List[str],
    skewed_columns: List[str],
    k_features: int,
) -> Pipeline:
    non_skewed_columns = [col for col in feature_columns if col not in skewed_columns]

    transformers = []

    if skewed_columns:
        skewed_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("log1p", FunctionTransformer(safe_log1p_array, validate=False)),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("skewed_num", skewed_pipeline, skewed_columns))

    if non_skewed_columns:
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("num", numeric_pipeline, non_skewed_columns))

    column_transformer = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False
    )

    pipeline = Pipeline([
        ("preprocessor", column_transformer),
        ("selector", SelectKBest(score_func=f_classif, k=k_features))
    ])

    return pipeline


# ---------------------------------------------------
# Model Builders
# ---------------------------------------------------

def compute_scale_pos_weight(y: pd.Series) -> float:
    negative_count = int((y == 0).sum())
    positive_count = int((y == 1).sum())
    return 1.0 if positive_count == 0 else negative_count / positive_count


def build_models(y_train: pd.Series) -> Dict[str, Any]:
    scale_pos_weight = compute_scale_pos_weight(y_train)

    models: Dict[str, Any] = {
        "LogisticRegression": LogisticRegression(
            C=1.0,
            max_iter=5000,
            class_weight="balanced",
            solver="liblinear",
            random_state=RANDOM_STATE,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=500,
            max_depth=8,
            min_samples_split=4,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=500,
            max_depth=8,
            min_samples_split=4,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBClassifier(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.2,
            reg_lambda=1.5,
            min_child_weight=2,
            gamma=0.1,
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_STATE,
        )

    if CATBOOST_AVAILABLE:
        models["CatBoost"] = CatBoostClassifier(
            iterations=500,
            depth=5,
            learning_rate=0.03,
            l2_leaf_reg=5,
            loss_function="Logloss",
            eval_metric="AUC",
            random_state=RANDOM_STATE,
            verbose=0,
        )

    return models


# ---------------------------------------------------
# Candidate Evaluation
# ---------------------------------------------------

def build_candidate_pipeline(preprocessor: Pipeline, estimator: Any) -> Pipeline:
    return Pipeline([
        ("features", clone(preprocessor)),
        ("model", clone(estimator))
    ])


def compare_models_cv(
    models: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor: Pipeline,
) -> pd.DataFrame:
    """
    Full-pipeline CV to avoid preprocessing/selection leakage across folds.
    """
    logger.info("Running full-pipeline cross-validation comparison...")

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    scoring = {
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision",
        "f1": "f1",
        "recall": "recall",
        "balanced_accuracy": "balanced_accuracy",
    }

    rows: List[Dict[str, Any]] = []

    for model_name, estimator in models.items():
        logger.info(f"Evaluating candidate: {model_name}")

        pipeline = build_candidate_pipeline(preprocessor, estimator)

        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            return_train_score=False,
        )

        roc_auc_mean = float(np.mean(scores["test_roc_auc"]))
        roc_auc_std = float(np.std(scores["test_roc_auc"]))
        pr_auc_mean = float(np.mean(scores["test_pr_auc"]))
        f1_mean = float(np.mean(scores["test_f1"]))
        recall_mean = float(np.mean(scores["test_recall"]))
        balanced_accuracy_mean = float(np.mean(scores["test_balanced_accuracy"]))

        composite_score = (
            0.35 * roc_auc_mean +
            0.20 * pr_auc_mean +
            0.20 * f1_mean +
            0.15 * recall_mean +
            0.10 * balanced_accuracy_mean
        )

        rows.append({
            "model": model_name,
            "cv_roc_auc_mean": roc_auc_mean,
            "cv_roc_auc_std": roc_auc_std,
            "cv_pr_auc_mean": pr_auc_mean,
            "cv_f1_mean": f1_mean,
            "cv_recall_mean": recall_mean,
            "cv_balanced_accuracy_mean": balanced_accuracy_mean,
            "composite_score": composite_score,
        })

        logger.info(
            f"{model_name} | ROC-AUC={roc_auc_mean:.4f}, PR-AUC={pr_auc_mean:.4f}, "
            f"F1={f1_mean:.4f}, Recall={recall_mean:.4f}, "
            f"BalancedAcc={balanced_accuracy_mean:.4f}, Composite={composite_score:.4f}"
        )

    results_df = pd.DataFrame(rows).sort_values(
        by=["composite_score", "cv_roc_auc_mean", "cv_pr_auc_mean", "cv_f1_mean", "cv_recall_mean"],
        ascending=False
    ).reset_index(drop=True)

    return results_df


# ---------------------------------------------------
# Fitting Helpers
# ---------------------------------------------------

def fit_feature_pipeline(
    preprocessor: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Pipeline:
    fitted = clone(preprocessor)
    fitted.fit(X_train, y_train)
    return fitted


def transform_features(
    fitted_preprocessor: Pipeline,
    X: pd.DataFrame
) -> np.ndarray:
    return fitted_preprocessor.transform(X)


def get_selected_feature_names(
    fitted_preprocessor: Pipeline,
    original_columns: List[str]
) -> List[str]:
    """
    Recover selected feature names safely without relying on
    ColumnTransformer.get_feature_names_out(), because FunctionTransformer
    in the skewed-feature pipeline may not expose feature names correctly.
    """
    feature_step = fitted_preprocessor.named_steps["preprocessor"]
    selector = fitted_preprocessor.named_steps["selector"]

    transformed_names: List[str] = []

    for _, _, cols in feature_step.transformers_:
        if cols is None:
            continue
        if isinstance(cols, slice):
            transformed_names.extend(original_columns[cols])
        elif isinstance(cols, (list, tuple, np.ndarray, pd.Index)):
            transformed_names.extend([str(col) for col in cols])

    selected_mask = selector.get_support()

    if len(transformed_names) != len(selected_mask):
        raise ValueError(
            f"Mismatch between transformed feature names ({len(transformed_names)}) "
            f"and selector mask ({len(selected_mask)})."
        )

    selected_names = [
        name for name, keep in zip(transformed_names, selected_mask) if keep
    ]
    return selected_names


def find_best_threshold(
    y_true: pd.Series,
    y_prob: np.ndarray
) -> Tuple[float, Dict[str, float], List[Dict[str, float]]]:
    """
    Recall-aware threshold search for healthcare use.
    """
    logger.info("Tuning threshold on validation set...")

    rows: List[Dict[str, float]] = []
    best_threshold = 0.3
    best_score = -np.inf
    best_metrics: Dict[str, float] = {}

    for threshold in np.arange(THRESHOLD_MIN, THRESHOLD_MAX + 1e-9, THRESHOLD_STEP):
        preds = (y_prob >= threshold).astype(int)

        accuracy = float(accuracy_score(y_true, preds))
        precision = float(precision_score(y_true, preds, zero_division=0))
        recall = float(recall_score(y_true, preds, zero_division=0))
        f1 = float(f1_score(y_true, preds, zero_division=0))
        balanced_acc = float(balanced_accuracy_score(y_true, preds))

        selection_score = (
            0.45 * f1 +
            0.30 * recall +
            0.15 * balanced_acc +
            0.10 * precision
        )

        row = {
            "threshold": round(float(threshold), 2),
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "balanced_accuracy": balanced_acc,
            "selection_score": float(selection_score),
        }
        rows.append(row)

        if selection_score > best_score:
            best_score = selection_score
            best_threshold = float(threshold)
            best_metrics = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "balanced_accuracy": balanced_acc,
            }

    logger.info(f"Best threshold: {best_threshold:.2f}")
    logger.info(f"Validation metrics at best threshold: {best_metrics}")

    return best_threshold, best_metrics, rows


# ---------------------------------------------------
# Evaluation
# ---------------------------------------------------

def evaluate_with_threshold(
    model: Any,
    X: np.ndarray,
    y: pd.Series,
    threshold: float
) -> Dict[str, float]:
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    return {
        "accuracy": float(accuracy_score(y, y_pred)),
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "recall": float(recall_score(y, y_pred, zero_division=0)),
        "f1": float(f1_score(y, y_pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, y_pred)),
        "roc_auc": float(roc_auc_score(y, y_prob)),
        "pr_auc": float(average_precision_score(y, y_prob)),
        "brier_score": float(brier_score_loss(y, y_prob)),
    }


# ---------------------------------------------------
# Registry Builders
# ---------------------------------------------------

def build_training_config(
    raw_feature_count: int,
    engineered_feature_count: int,
    selected_feature_count: int,
    skewed_columns: List[str],
    best_model_name: str,
) -> Dict[str, Any]:
    return {
        "dataset_name": DATASET_NAME,
        "target_column": TARGET_COLUMN,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "valid_size": VALID_SIZE,
        "cv_folds": CV_FOLDS,
        "feature_selection": {
            "method": "SelectKBest(f_classif)",
            "min_features_to_keep": MIN_FEATURES_TO_KEEP,
            "max_features_to_keep": MAX_FEATURES_TO_KEEP,
            "actual_selected_feature_count": selected_feature_count,
        },
        "preprocessing": {
            "imputer": "median",
            "scaler": "standard_scaler",
            "log_transform_for_high_skew_features": True,
            "high_skew_feature_threshold_abs_skew": 1.0,
            "skewed_columns_detected": skewed_columns,
        },
        "candidate_models": {
            "LogisticRegression": True,
            "RandomForest": True,
            "ExtraTrees": True,
            "XGBoost": XGBOOST_AVAILABLE,
            "CatBoost": CATBOOST_AVAILABLE,
        },
        "final_model_selection": {
            "best_model_name": best_model_name,
            "selection_strategy": "highest composite full-pipeline CV score",
            "composite_score_formula": "0.35*roc_auc + 0.20*pr_auc + 0.20*f1 + 0.15*recall + 0.10*balanced_accuracy",
        },
        "calibration": {
            "enabled": True,
            "method": CALIBRATION_METHOD,
            "cv": CALIBRATION_CV,
        },
        "input_feature_count": raw_feature_count,
        "engineered_feature_count": engineered_feature_count,
        "selected_feature_count": selected_feature_count,
    }


def build_training_summary(
    X_shape: Tuple[int, int],
    train_shape: Tuple[int, int],
    val_shape: Tuple[int, int],
    test_shape: Tuple[int, int],
    class_distribution: Dict[str, int],
    cv_results: pd.DataFrame,
    best_model_name: str,
    best_threshold: float,
) -> Dict[str, Any]:
    best_cv_row = cv_results.iloc[0].to_dict() if not cv_results.empty else {}

    return {
        "dataset_name": DATASET_NAME,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "input_dataset_shape": {
            "rows": int(X_shape[0]),
            "features": int(X_shape[1]),
        },
        "split_summary": {
            "train_shape": {"rows": int(train_shape[0]), "features": int(train_shape[1])},
            "validation_shape": {"rows": int(val_shape[0]), "features": int(val_shape[1])},
            "test_shape": {"rows": int(test_shape[0]), "features": int(test_shape[1])},
        },
        "target_distribution": class_distribution,
        "best_cv_model": {k: to_serializable(v) for k, v in best_cv_row.items()},
        "final_selected_model": best_model_name,
        "decision_threshold": float(best_threshold),
    }


def build_model_metadata(
    selected_features: List[str],
    best_model_name: str,
    train_metrics: Dict[str, float],
    val_metrics: Dict[str, float],
    test_metrics: Dict[str, float],
) -> Dict[str, Any]:
    return {
        "dataset_name": DATASET_NAME,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "model_name": f"Calibrated{best_model_name}",
        "model_family": "binary_classification",
        "target_column": TARGET_COLUMN,
        "selected_features_count": int(len(selected_features)),
        "selected_features": selected_features,
        "base_model": best_model_name,
        "calibration": {
            "enabled": True,
            "method": CALIBRATION_METHOD,
            "cv": CALIBRATION_CV,
        },
        "thresholding": {
            "enabled": True,
            "selection_strategy": "validation-set recall-aware threshold tuning",
        },
        "performance_snapshot": {
            "train_f1": to_serializable(train_metrics.get("f1")),
            "validation_f1": to_serializable(val_metrics.get("f1")),
            "test_f1": to_serializable(test_metrics.get("f1")),
            "train_roc_auc": to_serializable(train_metrics.get("roc_auc")),
            "validation_roc_auc": to_serializable(val_metrics.get("roc_auc")),
            "test_roc_auc": to_serializable(test_metrics.get("roc_auc")),
            "test_pr_auc": to_serializable(test_metrics.get("pr_auc")),
            "test_brier_score": to_serializable(test_metrics.get("brier_score")),
        },
        "artifact_files": [
            "model.pkl",
            "preprocessor.pkl",
            "selected_features.json",
            "threshold.json",
            "training_config.json",
            "training_summary.json",
            "model_metadata.json",
            "train_metrics.json",
        ],
    }


def build_train_metrics(
    cv_results: pd.DataFrame,
    threshold_search_rows: List[Dict[str, float]],
    train_metrics: Dict[str, float],
    val_metrics: Dict[str, float],
    test_metrics: Dict[str, float],
) -> Dict[str, Any]:
    return {
        "dataset_name": DATASET_NAME,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "cross_validation_results": cv_results.to_dict(orient="records"),
        "threshold_search": threshold_search_rows,
        "train_metrics_at_best_threshold": train_metrics,
        "validation_metrics_at_best_threshold": val_metrics,
        "test_metrics_at_best_threshold": test_metrics,
        "generalization_gap": {
            "f1_gap_train_vs_validation": float(train_metrics["f1"] - val_metrics["f1"]),
            "f1_gap_train_vs_test": float(train_metrics["f1"] - test_metrics["f1"]),
            "roc_auc_gap_train_vs_validation": float(train_metrics["roc_auc"] - val_metrics["roc_auc"]),
            "roc_auc_gap_train_vs_test": float(train_metrics["roc_auc"] - test_metrics["roc_auc"]),
            "brier_gap_train_vs_test": float(train_metrics["brier_score"] - test_metrics["brier_score"]),
        },
    }


# ---------------------------------------------------
# Save Artifacts
# ---------------------------------------------------

def save_registry_artifacts(
    model: Any,
    preprocessor: Any,
    selected_features: List[str],
    threshold: float,
    training_config: Dict[str, Any],
    training_summary: Dict[str, Any],
    model_metadata: Dict[str, Any],
    train_metrics: Dict[str, Any],
) -> None:
    with open(MODEL_PKL_PATH, "wb") as f:
        pickle.dump(model, f)

    with open(PREPROCESSOR_PKL_PATH, "wb") as f:
        pickle.dump(preprocessor, f)

    save_json(
        {
            "dataset_name": DATASET_NAME,
            "selected_features": selected_features,
            "selected_feature_count": len(selected_features),
        },
        SELECTED_FEATURES_PATH
    )

    save_json(
        {
            "dataset_name": DATASET_NAME,
            "threshold": float(threshold),
            "threshold_selection_strategy": "max(0.45*f1 + 0.30*recall + 0.15*balanced_accuracy + 0.10*precision) on validation set",
        },
        THRESHOLD_PATH
    )

    save_json(training_config, TRAINING_CONFIG_PATH)
    save_json(training_summary, TRAINING_SUMMARY_PATH)
    save_json(model_metadata, MODEL_METADATA_PATH)
    save_json(train_metrics, TRAIN_METRICS_PATH)

    logger.info(f"Saved model: {MODEL_PKL_PATH}")
    logger.info(f"Saved preprocessor: {PREPROCESSOR_PKL_PATH}")
    logger.info(f"Saved selected features: {SELECTED_FEATURES_PATH}")
    logger.info(f"Saved threshold: {THRESHOLD_PATH}")
    logger.info(f"Saved training config: {TRAINING_CONFIG_PATH}")
    logger.info(f"Saved training summary: {TRAINING_SUMMARY_PATH}")
    logger.info(f"Saved model metadata: {MODEL_METADATA_PATH}")
    logger.info(f"Saved train metrics: {TRAIN_METRICS_PATH}")


# ---------------------------------------------------
# Main Training Pipeline
# ---------------------------------------------------

def train() -> None:
    logger.info("Starting premium breast cancer training pipeline...")

    X_raw, y = load_data()
    raw_feature_count = X_raw.shape[1]

    X = feature_engineering(X_raw)
    engineered_feature_count = X.shape[1]

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    feature_columns = list(X_train.columns)
    skewed_columns = identify_high_skew_features(X_train)

    k_features = min(MAX_FEATURES_TO_KEEP, max(MIN_FEATURES_TO_KEEP, int(len(feature_columns) * 0.6)))
    preprocessor = build_preprocessor(
        feature_columns=feature_columns,
        skewed_columns=skewed_columns,
        k_features=k_features,
    )

    models = build_models(y_train)
    cv_results = compare_models_cv(
        models=models,
        X_train=X_train,
        y_train=y_train,
        preprocessor=preprocessor,
    )

    best_model_name = str(cv_results.iloc[0]["model"])
    best_estimator = models[best_model_name]

    logger.info(f"Selected best model: {best_model_name}")

    fitted_preprocessor = fit_feature_pipeline(preprocessor, X_train, y_train)

    X_train_selected = transform_features(fitted_preprocessor, X_train)
    X_val_selected = transform_features(fitted_preprocessor, X_val)
    X_test_selected = transform_features(fitted_preprocessor, X_test)

    selected_features = get_selected_feature_names(fitted_preprocessor, feature_columns)
    logger.info(f"Selected feature count: {len(selected_features)}")

    model = clone(best_estimator)
    model.fit(X_train_selected, y_train)

    val_probs = model.predict_proba(X_val_selected)[:, 1]
    best_threshold, val_threshold_metrics, threshold_search_rows = find_best_threshold(y_val, val_probs)

    train_metrics = evaluate_with_threshold(model, X_train_selected, y_train, best_threshold)
    val_metrics = evaluate_with_threshold(model, X_val_selected, y_val, best_threshold)
    test_metrics = evaluate_with_threshold(model, X_test_selected, y_test, best_threshold)

    packaged_preprocessor = {
        "feature_engineering_applied": True,
        "feature_engineering_version": "v2_conservative_breast_cancer",
        "input_feature_columns": list(X_raw.columns),
        "engineered_feature_columns": feature_columns,
        "detected_high_skew_columns": skewed_columns,
        "fitted_feature_pipeline": fitted_preprocessor,
    }

    training_config = build_training_config(
        raw_feature_count=raw_feature_count,
        engineered_feature_count=engineered_feature_count,
        selected_feature_count=len(selected_features),
        skewed_columns=skewed_columns,
        best_model_name=best_model_name,
    )

    training_summary = build_training_summary(
        X_shape=X.shape,
        train_shape=X_train.shape,
        val_shape=X_val.shape,
        test_shape=X_test.shape,
        class_distribution=compute_class_distribution(y),
        cv_results=cv_results,
        best_model_name=best_model_name,
        best_threshold=best_threshold,
    )

    model_metadata = build_model_metadata(
        selected_features=selected_features,
        best_model_name=best_model_name,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
    )

    metrics_artifact = build_train_metrics(
        cv_results=cv_results,
        threshold_search_rows=threshold_search_rows,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
    )

    save_registry_artifacts(
        model=model,
        preprocessor=packaged_preprocessor,
        selected_features=selected_features,
        threshold=best_threshold,
        training_config=training_config,
        training_summary=training_summary,
        model_metadata=model_metadata,
        train_metrics=metrics_artifact,
    )

    logger.info("Breast cancer training pipeline completed successfully.")
    logger.info(f"Best model: {best_model_name}")
    logger.info(f"Best threshold: {best_threshold:.2f}")
    logger.info(f"Train metrics: {train_metrics}")
    logger.info(f"Validation metrics: {val_metrics}")
    logger.info(f"Test metrics: {test_metrics}")


if __name__ == "__main__":
    train()
