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
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
    brier_score_loss
)
from sklearn.calibration import calibration_curve
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
EVAL_REGISTRY_DIR = os.path.join(BASE_DIR, "model_registry", DATASET_NAME, "evaluate")

MODEL_PATH = os.path.join(TRAIN_REGISTRY_DIR, "model.pkl")
PREPROCESSOR_PATH = os.path.join(TRAIN_REGISTRY_DIR, "preprocessor.pkl")
THRESHOLD_PATH = os.path.join(TRAIN_REGISTRY_DIR, "threshold.json")
SELECTED_FEATURES_PATH = os.path.join(TRAIN_REGISTRY_DIR, "selected_features.json")
TRAINING_CONFIG_PATH = os.path.join(TRAIN_REGISTRY_DIR, "training_config.json")
TRAINING_SUMMARY_PATH = os.path.join(TRAIN_REGISTRY_DIR, "training_summary.json")
MODEL_METADATA_PATH = os.path.join(TRAIN_REGISTRY_DIR, "model_metadata.json")
TRAIN_METRICS_PATH = os.path.join(TRAIN_REGISTRY_DIR, "train_metrics.json")

EVALUATION_METRICS_PATH = os.path.join(EVAL_REGISTRY_DIR, "evaluation_metrics.json")
CLASSIFICATION_REPORT_PATH = os.path.join(EVAL_REGISTRY_DIR, "classification_report.json")
THRESHOLD_ANALYSIS_PATH = os.path.join(EVAL_REGISTRY_DIR, "threshold_analysis.csv")
ERROR_ANALYSIS_PATH = os.path.join(EVAL_REGISTRY_DIR, "error_analysis.json")

CONFUSION_PLOT_PATH = os.path.join(EVAL_REGISTRY_DIR, "confusion_matrix.png")
ROC_PLOT_PATH = os.path.join(EVAL_REGISTRY_DIR, "roc_curve.png")
PR_PLOT_PATH = os.path.join(EVAL_REGISTRY_DIR, "precision_recall_curve.png")
CALIBRATION_PLOT_PATH = os.path.join(EVAL_REGISTRY_DIR, "calibration_curve.png")

os.makedirs(EVAL_REGISTRY_DIR, exist_ok=True)


# ---------------------------------------------------
# Constants
# ---------------------------------------------------

RANDOM_STATE = 42
TEST_SIZE = 0.15
VALID_SIZE = 0.15


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


def compute_specificity(y_true: pd.Series, y_pred: np.ndarray) -> float:
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape != (2, 2):
        return 0.0
    tn, fp, fn, tp = cm.ravel()
    return float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0


# ---------------------------------------------------
# Same training-time feature engineering
# ---------------------------------------------------

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    eps = 1e-6

    if {"radius_mean", "perimeter_mean"}.issubset(df.columns):
        df["radius_perimeter_ratio_mean"] = df["radius_mean"] / (df["perimeter_mean"] + eps)

    if {"area_mean", "radius_mean"}.issubset(df.columns):
        df["area_radius_ratio_mean"] = df["area_mean"] / (df["radius_mean"] + eps)

    if {"radius_worst", "radius_mean"}.issubset(df.columns):
        df["radius_worst_to_mean"] = df["radius_worst"] / (df["radius_mean"] + eps)

    if {"perimeter_worst", "perimeter_mean"}.issubset(df.columns):
        df["perimeter_worst_to_mean"] = df["perimeter_worst"] / (df["perimeter_mean"] + eps)

    if {"area_worst", "area_mean"}.issubset(df.columns):
        df["area_worst_to_mean"] = df["area_worst"] / (df["area_mean"] + eps)

    if {"compactness_mean", "concavity_mean"}.issubset(df.columns):
        df["compactness_concavity_interaction"] = df["compactness_mean"] * df["concavity_mean"]

    if {"concave_points_mean", "concavity_mean"}.issubset(df.columns):
        df["concave_to_concavity_ratio"] = df["concave_points_mean"] / (df["concavity_mean"] + eps)

    if {"texture_mean", "symmetry_mean"}.issubset(df.columns):
        df["texture_symmetry_interaction"] = df["texture_mean"] * df["symmetry_mean"]

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
    logger.info(f"Target distribution: {y.value_counts().to_dict()}")

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


def load_registry_artifacts() -> Tuple[Any, Any, float, Dict[str, Any]]:
    logger.info("Loading training registry artifacts...")

    required_paths = [MODEL_PATH, PREPROCESSOR_PATH, THRESHOLD_PATH]
    for path in required_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required artifact not found: {path}")

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    with open(PREPROCESSOR_PATH, "rb") as f:
        preprocessor_artifact = pickle.load(f)

    threshold_data = load_json_if_exists(THRESHOLD_PATH)

    metadata = {
        "selected_features": load_json_if_exists(SELECTED_FEATURES_PATH),
        "training_config": load_json_if_exists(TRAINING_CONFIG_PATH),
        "training_summary": load_json_if_exists(TRAINING_SUMMARY_PATH),
        "model_metadata": load_json_if_exists(MODEL_METADATA_PATH),
        "train_metrics": load_json_if_exists(TRAIN_METRICS_PATH),
        "threshold_data": threshold_data
    }

    threshold = extract_threshold(threshold_data)

    logger.info(f"Loaded threshold: {threshold:.4f}")

    return model, preprocessor_artifact, threshold, metadata


# ---------------------------------------------------
# Preprocessing application
# ---------------------------------------------------

def transform_features(X: pd.DataFrame, preprocessor_artifact: Any) -> Any:
    logger.info("Applying saved preprocessing contract...")

    if isinstance(preprocessor_artifact, dict):
        if "fitted_feature_pipeline" in preprocessor_artifact:
            return preprocessor_artifact["fitted_feature_pipeline"].transform(X)

        if "sklearn_preprocessor" in preprocessor_artifact and "feature_selector" in preprocessor_artifact:
            X_processed = preprocessor_artifact["sklearn_preprocessor"].transform(X)
            return preprocessor_artifact["feature_selector"].transform(X_processed)

        if "preprocessor" in preprocessor_artifact:
            return preprocessor_artifact["preprocessor"].transform(X)

    if hasattr(preprocessor_artifact, "transform"):
        return preprocessor_artifact.transform(X)

    raise ValueError("Unsupported preprocessor artifact format in preprocessor.pkl")


# ---------------------------------------------------
# Metrics
# ---------------------------------------------------

def compute_metrics(
    y_true: pd.Series,
    y_prob: np.ndarray,
    threshold: float
) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(compute_specificity(y_true, y_pred)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "threshold": float(threshold),
        "prevalence": float(np.mean(y_true)),
        "positive_prediction_rate": float(np.mean(y_pred))
    }

    return metrics


# ---------------------------------------------------
# Threshold analysis
# ---------------------------------------------------

def build_threshold_analysis(y_true: pd.Series, y_prob: np.ndarray) -> pd.DataFrame:
    logger.info("Running threshold sensitivity analysis...")

    rows = []
    roc_auc = float(roc_auc_score(y_true, y_prob))
    pr_auc = float(average_precision_score(y_true, y_prob))
    brier = float(brier_score_loss(y_true, y_prob))

    for threshold in np.arange(0.10, 0.91, 0.05):
        y_pred = (y_prob >= threshold).astype(int)

        rows.append({
            "threshold": round(float(threshold), 2),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "specificity": float(compute_specificity(y_true, y_pred)),
            "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "brier_score": brier,
            "positive_prediction_rate": float(np.mean(y_pred))
        })

    threshold_df = pd.DataFrame(rows)
    threshold_df.to_csv(THRESHOLD_ANALYSIS_PATH, index=False)
    logger.info(f"Saved threshold analysis: {THRESHOLD_ANALYSIS_PATH}")

    return threshold_df


# ---------------------------------------------------
# Plots
# ---------------------------------------------------

def save_confusion_matrix_plot(cm: np.ndarray) -> None:
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()

    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["Benign (0)", "Malignant (1)"])
    plt.yticks(tick_marks, ["Benign (0)", "Malignant (1)"])

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], "d"), ha="center", va="center")

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(CONFUSION_PLOT_PATH, bbox_inches="tight")
    plt.close()


def save_roc_curve_plot(y_true: pd.Series, y_prob: np.ndarray) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_score = roc_auc_score(y_true, y_prob)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f"ROC-AUC = {auc_score:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ROC_PLOT_PATH, bbox_inches="tight")
    plt.close()


def save_pr_curve_plot(y_true: pd.Series, y_prob: np.ndarray) -> None:
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap_score = average_precision_score(y_true, y_prob)

    plt.figure(figsize=(7, 5))
    plt.plot(recall, precision, label=f"PR-AUC = {ap_score:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PR_PLOT_PATH, bbox_inches="tight")
    plt.close()


def save_calibration_curve_plot(y_true: pd.Series, y_prob: np.ndarray) -> None:
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="uniform")

    plt.figure(figsize=(7, 5))
    plt.plot(mean_pred, frac_pos, marker="o", label="Model")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect Calibration")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title("Calibration Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(CALIBRATION_PLOT_PATH, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------
# Report builders
# ---------------------------------------------------

def build_classification_report_json(
    y_true: pd.Series,
    y_prob: np.ndarray,
    threshold: float
) -> Dict[str, Any]:
    y_pred = (y_prob >= threshold).astype(int)
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    return {
        "dataset_name": DATASET_NAME,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "threshold": float(threshold),
        "report": report
    }


def build_error_analysis(
    y_true: pd.Series,
    y_prob: np.ndarray,
    threshold: float
) -> Dict[str, Any]:
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)

    tn, fp, fn, tp = cm.ravel()

    fp_idx = np.where((y_true.values == 0) & (y_pred == 1))[0]
    fn_idx = np.where((y_true.values == 1) & (y_pred == 0))[0]

    fp_probs = y_prob[fp_idx].tolist()
    fn_probs = y_prob[fn_idx].tolist()

    hardest_fp = sorted(
        [{"test_index": int(i), "predicted_probability": float(y_prob[i])} for i in fp_idx],
        key=lambda x: x["predicted_probability"],
        reverse=True
    )[:10]

    hardest_fn = sorted(
        [{"test_index": int(i), "predicted_probability": float(y_prob[i])} for i in fn_idx],
        key=lambda x: x["predicted_probability"]
    )[:10]

    return {
        "dataset_name": DATASET_NAME,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "threshold": float(threshold),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp)
        },
        "error_counts": {
            "total_errors": int(fp + fn),
            "false_positives": int(fp),
            "false_negatives": int(fn)
        },
        "false_positive_probability_summary": {
            "count": int(len(fp_probs)),
            "mean": float(np.mean(fp_probs)) if fp_probs else None,
            "max": float(np.max(fp_probs)) if fp_probs else None
        },
        "false_negative_probability_summary": {
            "count": int(len(fn_probs)),
            "mean": float(np.mean(fn_probs)) if fn_probs else None,
            "min": float(np.min(fn_probs)) if fn_probs else None
        },
        "hardest_false_positives_top10": hardest_fp,
        "hardest_false_negatives_top10": hardest_fn,
        "clinical_risk_note": "False negatives are especially important in breast cancer screening because missed malignant cases carry higher clinical risk."
    }


def build_evaluation_metrics_artifact(
    metrics: Dict[str, float],
    metadata: Dict[str, Any]
) -> Dict[str, Any]:
    return {
        "dataset_name": DATASET_NAME,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "metrics": metrics,
        "evaluation_scope": "untouched recreated test split",
        "source_training_artifacts": {
            "model_path": MODEL_PATH,
            "preprocessor_path": PREPROCESSOR_PATH,
            "threshold_path": THRESHOLD_PATH,
            "selected_features_path": SELECTED_FEATURES_PATH
        },
        "train_summary_reference": metadata.get("training_summary", {}),
        "artifacts": {
            "classification_report_json": CLASSIFICATION_REPORT_PATH,
            "threshold_analysis_csv": THRESHOLD_ANALYSIS_PATH,
            "error_analysis_json": ERROR_ANALYSIS_PATH,
            "confusion_matrix_plot": CONFUSION_PLOT_PATH,
            "roc_curve_plot": ROC_PLOT_PATH,
            "precision_recall_curve_plot": PR_PLOT_PATH,
            "calibration_curve_plot": CALIBRATION_PLOT_PATH
        }
    }


# ---------------------------------------------------
# Main evaluation pipeline
# ---------------------------------------------------

def run_evaluation_pipeline() -> Dict[str, float]:
    logger.info("Starting breast cancer evaluation pipeline...")

    X, y = load_data()
    _, _, X_test, _, _, y_test = recreate_split(X, y)

    model, preprocessor_artifact, threshold, metadata = load_registry_artifacts()

    X_test_ready = transform_features(X_test, preprocessor_artifact)

    logger.info("Running inference on untouched test set...")
    y_prob = model.predict_proba(X_test_ready)[:, 1]

    metrics = compute_metrics(y_test, y_prob, threshold)

    logger.info("Final evaluation metrics:")
    for k, v in metrics.items():
        logger.info(f"{k}: {v:.6f}")

    threshold_df = build_threshold_analysis(y_test, y_prob)

    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred)

    save_confusion_matrix_plot(cm)
    save_roc_curve_plot(y_test, y_prob)
    save_pr_curve_plot(y_test, y_prob)
    save_calibration_curve_plot(y_test, y_prob)

    classification_report_json = build_classification_report_json(y_test, y_prob, threshold)
    error_analysis_json = build_error_analysis(y_test, y_prob, threshold)
    evaluation_metrics_json = build_evaluation_metrics_artifact(metrics, metadata)

    save_json(classification_report_json, CLASSIFICATION_REPORT_PATH)
    save_json(error_analysis_json, ERROR_ANALYSIS_PATH)
    save_json(evaluation_metrics_json, EVALUATION_METRICS_PATH)

    logger.info(f"Saved evaluation metrics: {EVALUATION_METRICS_PATH}")
    logger.info(f"Saved classification report: {CLASSIFICATION_REPORT_PATH}")
    logger.info(f"Saved threshold analysis: {THRESHOLD_ANALYSIS_PATH}")
    logger.info(f"Saved error analysis: {ERROR_ANALYSIS_PATH}")
    logger.info("Breast cancer evaluation pipeline completed successfully.")

    return metrics


if __name__ == "__main__":
    run_evaluation_pipeline()
