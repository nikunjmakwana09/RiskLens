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

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    brier_score_loss
)
from sklearn.model_selection import train_test_split
from sklearn.calibration import calibration_curve
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

DATASET_NAME = "heart"
TARGET_COLUMN = "target"
RANDOM_STATE = 42
TEST_SIZE = 0.20

DATA_PATH = os.path.join(BASE_DIR, "data", "processed", f"{DATASET_NAME}_clean.csv")

TRAIN_REGISTRY_DIR = os.path.join(BASE_DIR, "model_registry", DATASET_NAME, "train")
EVAL_REGISTRY_DIR = os.path.join(BASE_DIR, "model_registry", DATASET_NAME, "evaluate")
os.makedirs(EVAL_REGISTRY_DIR, exist_ok=True)

MODEL_PATH = os.path.join(TRAIN_REGISTRY_DIR, "model.pkl")
PREPROCESSOR_PATH = os.path.join(TRAIN_REGISTRY_DIR, "preprocessor.pkl")
SELECTED_FEATURES_PATH = os.path.join(TRAIN_REGISTRY_DIR, "selected_features.json")
THRESHOLD_PATH = os.path.join(TRAIN_REGISTRY_DIR, "threshold.json")
MODEL_METADATA_PATH = os.path.join(TRAIN_REGISTRY_DIR, "model_metadata.json")
TRAINING_CONFIG_PATH = os.path.join(TRAIN_REGISTRY_DIR, "training_config.json")
TRAINING_SUMMARY_PATH = os.path.join(TRAIN_REGISTRY_DIR, "training_summary.json")
TRAIN_METRICS_PATH = os.path.join(TRAIN_REGISTRY_DIR, "train_metrics.json")

EVALUATION_METRICS_PATH = os.path.join(EVAL_REGISTRY_DIR, "evaluation_metrics.json")
CLASSIFICATION_REPORT_JSON_PATH = os.path.join(EVAL_REGISTRY_DIR, "classification_report.json")
THRESHOLD_ANALYSIS_PATH = os.path.join(EVAL_REGISTRY_DIR, "threshold_analysis.csv")
ERROR_ANALYSIS_PATH = os.path.join(EVAL_REGISTRY_DIR, "error_analysis.json")

CONFUSION_MATRIX_PATH = os.path.join(EVAL_REGISTRY_DIR, "confusion_matrix.png")
ROC_CURVE_PATH = os.path.join(EVAL_REGISTRY_DIR, "roc_curve.png")
PR_CURVE_PATH = os.path.join(EVAL_REGISTRY_DIR, "precision_recall_curve.png")
CALIBRATION_CURVE_PATH = os.path.join(EVAL_REGISTRY_DIR, "calibration_curve.png")


# ---------------------------------------------------
# Required Columns
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
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if isinstance(obj, pd.Series):
        return obj.to_dict()
    return to_serializable(obj)


def save_json(data: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(make_json_safe(data), f, indent=4, ensure_ascii=False)
    logger.info(f"Saved JSON: {path}")


# ---------------------------------------------------
# Feature Engineering
# ---------------------------------------------------

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    eps = 1e-6

    required_pairs = [
        ("age", "thalach", "age_thalach_ratio"),
        ("chol", "age", "chol_age_ratio"),
        ("trestbps", "age", "bp_age_ratio"),
        ("oldpeak", "thalach", "oldpeak_thalach_ratio"),
    ]

    for num_col, den_col, new_col in required_pairs:
        if num_col in df.columns and den_col in df.columns:
            df[new_col] = df[num_col] / (df[den_col] + eps)

    return df


# ---------------------------------------------------
# Load Dataset
# ---------------------------------------------------

def load_dataset() -> pd.DataFrame:
    logger.info("Loading cleaned heart disease dataset...")

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    if df.empty:
        raise ValueError("Dataset is empty.")

    if df.isnull().sum().sum() != 0:
        raise ValueError("Processed dataset still contains missing values.")

    df = feature_engineering(df)

    logger.info(f"Dataset loaded successfully. Shape: {df.shape}")
    return df


# ---------------------------------------------------
# Registry Artifact Loading
# ---------------------------------------------------

def load_json_if_exists(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_threshold(payload: Dict[str, Any]) -> float | None:
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

    return None


def load_registry_artifacts() -> Tuple[Any, Any | None, List[str], float | None, Dict[str, Any]]:
    logger.info("Loading registry artifacts...")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    preprocessor = None
    if os.path.exists(PREPROCESSOR_PATH):
        with open(PREPROCESSOR_PATH, "rb") as f:
            preprocessor = pickle.load(f)

    selected_features_payload = load_json_if_exists(SELECTED_FEATURES_PATH)
    selected_features = selected_features_payload.get("selected_features", [])
    if not isinstance(selected_features, list):
        selected_features = []

    threshold_payload = load_json_if_exists(THRESHOLD_PATH)
    threshold = extract_threshold(threshold_payload)

    metadata = {
        "model_metadata": load_json_if_exists(MODEL_METADATA_PATH),
        "training_config": load_json_if_exists(TRAINING_CONFIG_PATH),
        "training_summary": load_json_if_exists(TRAINING_SUMMARY_PATH),
        "train_metrics": load_json_if_exists(TRAIN_METRICS_PATH),
        "selected_features_payload": selected_features_payload,
        "threshold_payload": threshold_payload
    }

    logger.info("Registry artifacts loaded successfully.")
    return model, preprocessor, selected_features, threshold, metadata


# ---------------------------------------------------
# Split Recreation
# ---------------------------------------------------

def create_holdout_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
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
# Preprocessing Compatibility
# ---------------------------------------------------

def transform_features(
    X: pd.DataFrame,
    model: Any,
    preprocessor_artifact: Any | None,
    selected_features: List[str]
):
    if preprocessor_artifact is not None:
        if isinstance(preprocessor_artifact, dict):
            if "fitted_feature_pipeline" in preprocessor_artifact:
                return preprocessor_artifact["fitted_feature_pipeline"].transform(X)
            if "preprocessor" in preprocessor_artifact:
                return preprocessor_artifact["preprocessor"].transform(X)
            if "sklearn_preprocessor" in preprocessor_artifact:
                Xt = preprocessor_artifact["sklearn_preprocessor"].transform(X)
                if "feature_selector" in preprocessor_artifact:
                    Xt = preprocessor_artifact["feature_selector"].transform(Xt)
                return Xt

        if hasattr(preprocessor_artifact, "transform"):
            return preprocessor_artifact.transform(X)

    if selected_features:
        missing = [col for col in selected_features if col not in X.columns]
        if missing:
            raise ValueError(f"Missing features required by trained model: {missing}")
        return X[selected_features]

    return X


# ---------------------------------------------------
# Metrics Utilities
# ---------------------------------------------------

def compute_specificity(y_true: pd.Series, y_pred: np.ndarray) -> float:
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape != (2, 2):
        return 0.0
    tn, fp, fn, tp = cm.ravel()
    return float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0


def calculate_metrics(y_true: pd.Series, y_prob: np.ndarray, threshold: float) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "specificity": float(compute_specificity(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "prevalence": float(np.mean(y_true)),
        "positive_prediction_rate": float(np.mean(y_pred))
    }

    return metrics


# ---------------------------------------------------
# Threshold Analysis
# ---------------------------------------------------

def generate_threshold_analysis(y_true: pd.Series, y_prob: np.ndarray) -> Tuple[float, pd.DataFrame]:
    thresholds = np.arange(0.10, 0.91, 0.01)

    roc_auc = float(roc_auc_score(y_true, y_prob))
    pr_auc = float(average_precision_score(y_true, y_prob))
    brier = float(brier_score_loss(y_true, y_prob))

    records = []

    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)

        records.append({
            "threshold": round(float(threshold), 2),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
            "specificity": float(compute_specificity(y_true, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "brier_score": brier,
            "positive_prediction_rate": float(np.mean(y_pred))
        })

    threshold_df = pd.DataFrame(records)

    filtered_df = threshold_df[
        (threshold_df["recall"] >= 0.80) &
        (threshold_df["specificity"] >= 0.60)
    ].copy()

    if not filtered_df.empty:
        filtered_df["selection_score"] = (
            0.35 * filtered_df["recall"] +
            0.25 * filtered_df["f1_score"] +
            0.15 * filtered_df["specificity"] +
            0.10 * filtered_df["precision"] +
            0.10 * filtered_df["balanced_accuracy"] +
            0.05 * filtered_df["pr_auc"]
        )
        best_row = filtered_df.sort_values(
            by=["selection_score", "f1_score", "precision"],
            ascending=False
        ).iloc[0]
    else:
        threshold_df["selection_score"] = (
            0.30 * threshold_df["recall"] +
            0.28 * threshold_df["f1_score"] +
            0.15 * threshold_df["specificity"] +
            0.12 * threshold_df["precision"] +
            0.10 * threshold_df["balanced_accuracy"] +
            0.05 * threshold_df["pr_auc"]
        )
        best_row = threshold_df.sort_values(
            by=["selection_score", "f1_score", "precision"],
            ascending=False
        ).iloc[0]

    threshold_df = threshold_df.sort_values(
        by=["f1_score", "recall", "precision"],
        ascending=[False, False, False]
    ).reset_index(drop=True)

    threshold_df.to_csv(THRESHOLD_ANALYSIS_PATH, index=False)
    logger.info(f"Threshold analysis saved to: {THRESHOLD_ANALYSIS_PATH}")

    return float(best_row["threshold"]), threshold_df


# ---------------------------------------------------
# Classification Report
# ---------------------------------------------------

def build_classification_report_json(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, Any]:
    report = classification_report(y_true, y_pred, digits=4, output_dict=True)
    cleaned_report = make_json_safe(report)
    return cleaned_report


# ---------------------------------------------------
# Error Analysis
# ---------------------------------------------------

def build_error_analysis(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y_prob: np.ndarray,
    threshold: float
) -> Dict[str, Any]:
    y_pred = (y_prob >= threshold).astype(int)

    errors = X_test.copy()
    errors["y_true"] = y_test.values
    errors["y_pred"] = y_pred
    errors["y_prob"] = y_prob
    errors["error_type"] = "correct"

    errors.loc[(errors["y_true"] == 0) & (errors["y_pred"] == 1), "error_type"] = "false_positive"
    errors.loc[(errors["y_true"] == 1) & (errors["y_pred"] == 0), "error_type"] = "false_negative"

    false_positives = errors[errors["error_type"] == "false_positive"].copy()
    false_negatives = errors[errors["error_type"] == "false_negative"].copy()

    difficult_cases = errors.copy()
    difficult_cases["distance_from_decision_boundary"] = np.abs(difficult_cases["y_prob"] - 0.5)
    difficult_cases = difficult_cases.sort_values(by="distance_from_decision_boundary", ascending=True).head(10)

    hardest_fp = false_positives.sort_values(by="y_prob", ascending=False).head(10)
    hardest_fn = false_negatives.sort_values(by="y_prob", ascending=True).head(10)

    result = {
        "dataset_name": DATASET_NAME,
        "evaluated_at_utc": datetime.utcnow().isoformat() + "Z",
        "threshold_used": float(threshold),
        "total_test_samples": int(len(y_test)),
        "total_errors": int((errors["y_true"] != errors["y_pred"]).sum()),
        "false_positives": {
            "count": int(len(false_positives)),
            "rate_percent": round(float((len(false_positives) / len(y_test)) * 100), 2),
            "mean_probability": round(float(false_positives["y_prob"].mean()), 6) if len(false_positives) > 0 else None
        },
        "false_negatives": {
            "count": int(len(false_negatives)),
            "rate_percent": round(float((len(false_negatives) / len(y_test)) * 100), 2),
            "mean_probability": round(float(false_negatives["y_prob"].mean()), 6) if len(false_negatives) > 0 else None
        },
        "hardest_false_positives_top10": make_json_safe(
            hardest_fp[["y_prob"]].reset_index().rename(columns={"index": "test_index"}).to_dict(orient="records")
        ),
        "hardest_false_negatives_top10": make_json_safe(
            hardest_fn[["y_prob"]].reset_index().rename(columns={"index": "test_index"}).to_dict(orient="records")
        ),
        "most_uncertain_cases": make_json_safe(
            difficult_cases[[
                "y_true", "y_pred", "y_prob", "error_type", "distance_from_decision_boundary"
            ]].to_dict(orient="records")
        ),
        "clinical_note": "False negatives are important because missed heart disease risk can delay intervention; false positives may increase follow-up testing burden."
    }

    return result


# ---------------------------------------------------
# Plots
# ---------------------------------------------------

def plot_confusion_matrix(y_true: pd.Series, y_pred: np.ndarray, save_path: str) -> None:
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()

    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["No Disease", "Disease"])
    plt.yticks(tick_marks, ["No Disease", "Disease"])

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close("all")

    logger.info(f"Confusion matrix saved to: {save_path}")


def plot_roc_curve(y_true: pd.Series, y_prob: np.ndarray, save_path: str) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_value = roc_auc_score(y_true, y_prob)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc_value:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close("all")

    logger.info(f"ROC curve saved to: {save_path}")


def plot_precision_recall_curve(y_true: pd.Series, y_prob: np.ndarray, save_path: str) -> None:
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    baseline = float(np.mean(y_true))

    plt.figure(figsize=(7, 5))
    plt.plot(recall, precision, label=f"PR-AUC = {pr_auc:.4f}")
    plt.hlines(y=baseline, xmin=0, xmax=1, linestyles="--", label=f"Baseline = {baseline:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close("all")

    logger.info(f"Precision-Recall curve saved to: {save_path}")


def plot_calibration_curve_chart(y_true: pd.Series, y_prob: np.ndarray, save_path: str) -> Dict[str, Any]:
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="uniform")

    plt.figure(figsize=(7, 5))
    plt.plot(mean_pred, frac_pos, marker="o", label="Model")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect Calibration")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title("Calibration Curve")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close("all")

    logger.info(f"Calibration curve saved to: {save_path}")

    return {
        "bins_used": int(len(frac_pos)),
        "mean_absolute_calibration_error": float(np.mean(np.abs(frac_pos - mean_pred))) if len(frac_pos) > 0 else None,
        "prob_true": [float(x) for x in frac_pos],
        "prob_pred": [float(x) for x in mean_pred]
    }


# ---------------------------------------------------
# Main Evaluation
# ---------------------------------------------------

def run_evaluation_pipeline() -> None:
    logger.info("Starting heart disease evaluation pipeline...")

    df = load_dataset()
    model, preprocessor, selected_features, registry_threshold, metadata = load_registry_artifacts()

    _, X_test, _, y_test = create_holdout_split(df)

    X_test_ready = transform_features(
        X=X_test,
        model=model,
        preprocessor_artifact=preprocessor,
        selected_features=selected_features
    )

    if registry_threshold is None:
        logger.warning("No saved threshold found. Falling back to analysis-derived threshold.")

    y_prob = model.predict_proba(X_test_ready)[:, 1]

    analysis_threshold, threshold_df = generate_threshold_analysis(y_test, y_prob)
    final_threshold = float(registry_threshold) if registry_threshold is not None else analysis_threshold
    y_pred = (y_prob >= final_threshold).astype(int)

    model_name = metadata.get("model_metadata", {}).get("model_name", "unknown")

    evaluation_metrics = {
        "dataset_name": DATASET_NAME,
        "model_name": model_name,
        "evaluated_at_utc": datetime.utcnow().isoformat() + "Z",
        "saved_training_threshold": float(registry_threshold) if registry_threshold is not None else None,
        "analysis_best_threshold": float(analysis_threshold),
        "final_threshold_used": float(final_threshold),
        "test_samples": int(len(y_test)),
        "metrics": calculate_metrics(y_test, y_prob, final_threshold),
        "source_artifacts": {
            "model_path": MODEL_PATH,
            "preprocessor_path": PREPROCESSOR_PATH,
            "selected_features_path": SELECTED_FEATURES_PATH,
            "threshold_path": THRESHOLD_PATH
        }
    }

    classification_report_json = build_classification_report_json(y_test, y_pred)
    error_analysis = build_error_analysis(X_test, y_test, y_prob, final_threshold)
    calibration_summary = plot_calibration_curve_chart(y_test, y_prob, CALIBRATION_CURVE_PATH)

    save_json(evaluation_metrics, EVALUATION_METRICS_PATH)
    save_json(classification_report_json, CLASSIFICATION_REPORT_JSON_PATH)
    save_json(error_analysis, ERROR_ANALYSIS_PATH)

    plot_confusion_matrix(y_test, y_pred, CONFUSION_MATRIX_PATH)
    plot_roc_curve(y_test, y_prob, ROC_CURVE_PATH)
    plot_precision_recall_curve(y_test, y_prob, PR_CURVE_PATH)

    logger.info(f"Top threshold analysis rows:\n{threshold_df.head(10).to_string(index=False)}")
    logger.info(f"Calibration summary: {json.dumps(make_json_safe(calibration_summary), indent=2)}")
    logger.info(f"Evaluation artifacts saved in: {EVAL_REGISTRY_DIR}")
    logger.info("Heart disease evaluation pipeline completed successfully.")


if __name__ == "__main__":
    run_evaluation_pipeline()
