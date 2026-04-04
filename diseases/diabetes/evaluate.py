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

DATASET_NAME = "diabetes"
TARGET_COLUMN = "Outcome"
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
TRAINING_CONFIG_PATH = os.path.join(TRAIN_REGISTRY_DIR, "training_config.json")
TRAINING_SUMMARY_PATH = os.path.join(TRAIN_REGISTRY_DIR, "training_summary.json")
MODEL_METADATA_PATH = os.path.join(TRAIN_REGISTRY_DIR, "model_metadata.json")
TRAIN_METRICS_PATH = os.path.join(TRAIN_REGISTRY_DIR, "train_metrics.json")

EVALUATION_METRICS_PATH = os.path.join(EVAL_REGISTRY_DIR, "evaluation_metrics.json")
CLASSIFICATION_REPORT_PATH = os.path.join(EVAL_REGISTRY_DIR, "classification_report.json")
THRESHOLD_ANALYSIS_PATH = os.path.join(EVAL_REGISTRY_DIR, "threshold_analysis.csv")
ERROR_ANALYSIS_PATH = os.path.join(EVAL_REGISTRY_DIR, "error_analysis.json")

CONFUSION_MATRIX_PLOT_PATH = os.path.join(EVAL_REGISTRY_DIR, "confusion_matrix.png")
ROC_CURVE_PATH = os.path.join(EVAL_REGISTRY_DIR, "roc_curve.png")
PR_CURVE_PATH = os.path.join(EVAL_REGISTRY_DIR, "precision_recall_curve.png")
CALIBRATION_CURVE_PATH = os.path.join(EVAL_REGISTRY_DIR, "calibration_curve.png")


# ---------------------------------------------------
# Required columns
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
# JSON helpers
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
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if isinstance(obj, pd.Series):
        return obj.to_dict()
    return to_serializable(obj)


def save_json(data: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(make_json_safe(data), f, indent=4, ensure_ascii=False)


# ---------------------------------------------------
# Same feature engineering as training
# ---------------------------------------------------

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
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
# Data loading
# ---------------------------------------------------

def load_dataset() -> pd.DataFrame:
    logger.info("Loading cleaned diabetes dataset...")

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


def recreate_holdout_split(
    df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X = df.drop(columns=[TARGET_COLUMN]).copy()
    y = df[TARGET_COLUMN].astype(int).copy()

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
# Registry loading
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

    nested_paths = [
        ("validation_metrics_at_optimal_threshold", "threshold"),
        ("selected_metrics", "threshold")
    ]
    for outer, inner in nested_paths:
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

    selected_payload = load_json_if_exists(SELECTED_FEATURES_PATH)
    selected_features = selected_payload.get("selected_features", [])
    if not isinstance(selected_features, list):
        selected_features = []

    threshold_payload = load_json_if_exists(THRESHOLD_PATH)
    threshold = extract_threshold(threshold_payload)

    metadata = {
        "selected_features_payload": selected_payload,
        "threshold_payload": threshold_payload,
        "training_config": load_json_if_exists(TRAINING_CONFIG_PATH),
        "training_summary": load_json_if_exists(TRAINING_SUMMARY_PATH),
        "model_metadata": load_json_if_exists(MODEL_METADATA_PATH),
        "train_metrics": load_json_if_exists(TRAIN_METRICS_PATH)
    }

    logger.info("Registry artifacts loaded successfully.")
    return model, preprocessor, selected_features, threshold, metadata


# ---------------------------------------------------
# Preprocessing compatibility
# ---------------------------------------------------

def transform_features(
    X: pd.DataFrame,
    model: Any,
    preprocessor_artifact: Any | None,
    selected_features: List[str]
):
    """
    Supports multiple registry designs:
    1. model.pkl is a full sklearn pipeline -> use X directly
    2. separate preprocessor.pkl exists -> transform X through it
    3. no preprocessor, but selected_features are saved -> subset X
    """

    if hasattr(model, "named_steps") or hasattr(model, "feature_names_in_"):
        # likely full pipeline or sklearn estimator already trained on raw selected frame
        if preprocessor_artifact is None:
            if selected_features:
                missing = [c for c in selected_features if c not in X.columns]
                if missing:
                    raise ValueError(f"Missing required selected features: {missing}")
                return X[selected_features]
            return X

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
        missing = [c for c in selected_features if c not in X.columns]
        if missing:
            raise ValueError(f"Missing required selected features: {missing}")
        return X[selected_features]

    return X


# ---------------------------------------------------
# Metrics
# ---------------------------------------------------

def compute_specificity(y_true: pd.Series, y_pred: np.ndarray) -> float:
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape != (2, 2):
        return 0.0
    tn, fp, fn, tp = cm.ravel()
    return float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0


def calculate_metrics(y_true: pd.Series, y_prob: np.ndarray, threshold: float) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(compute_specificity(y_true, y_pred)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "prevalence": float(np.mean(y_true)),
        "positive_prediction_rate": float(np.mean(y_pred))
    }


# ---------------------------------------------------
# Threshold analysis
# ---------------------------------------------------

def run_threshold_analysis(y_true: pd.Series, y_prob: np.ndarray) -> Tuple[float, pd.DataFrame]:
    logger.info("Running threshold analysis...")

    thresholds = np.arange(0.10, 0.91, 0.01)
    roc_auc = float(roc_auc_score(y_true, y_prob))
    pr_auc = float(average_precision_score(y_true, y_prob))
    brier = float(brier_score_loss(y_true, y_prob))

    records = []
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)

        record = {
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
        }
        records.append(record)

    threshold_df = pd.DataFrame(records)

    filtered_df = threshold_df[
        (threshold_df["recall"] >= 0.80) &
        (threshold_df["specificity"] >= 0.65)
    ].copy()

    if not filtered_df.empty:
        filtered_df["selection_score"] = (
            0.38 * filtered_df["recall"] +
            0.24 * filtered_df["f1_score"] +
            0.14 * filtered_df["specificity"] +
            0.12 * filtered_df["precision"] +
            0.07 * filtered_df["balanced_accuracy"] +
            0.05 * filtered_df["pr_auc"]
        )
        best_row = filtered_df.sort_values(
            by=["selection_score", "f1_score", "precision"],
            ascending=False
        ).iloc[0]
    else:
        threshold_df["selection_score"] = (
            0.34 * threshold_df["recall"] +
            0.28 * threshold_df["f1_score"] +
            0.14 * threshold_df["specificity"] +
            0.12 * threshold_df["precision"] +
            0.07 * threshold_df["balanced_accuracy"] +
            0.05 * threshold_df["pr_auc"]
        )
        best_row = threshold_df.sort_values(
            by=["selection_score", "f1_score", "precision"],
            ascending=False
        ).iloc[0]

    best_threshold = float(best_row["threshold"])
    threshold_df.to_csv(THRESHOLD_ANALYSIS_PATH, index=False)

    logger.info(f"Threshold analysis saved to: {THRESHOLD_ANALYSIS_PATH}")
    logger.info(f"Best analysis threshold: {best_threshold:.2f}")
    return best_threshold, threshold_df


# ---------------------------------------------------
# Reporting
# ---------------------------------------------------

def generate_classification_report_json(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, Any]:
    report = classification_report(
        y_true,
        y_pred,
        output_dict=True,
        zero_division=0
    )
    save_json(report, CLASSIFICATION_REPORT_PATH)
    logger.info(f"Classification report saved to: {CLASSIFICATION_REPORT_PATH}")
    return report


def plot_confusion_matrix(y_true: pd.Series, y_pred: np.ndarray, threshold: float) -> Dict[str, int]:
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest")
    plt.title(f"Confusion Matrix (Threshold = {threshold:.2f})")
    plt.colorbar()
    plt.xticks([0, 1], ["Non-Diabetic", "Diabetic"])
    plt.yticks([0, 1], ["Non-Diabetic", "Diabetic"])

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], "d"), ha="center", va="center")

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PLOT_PATH, dpi=300, bbox_inches="tight")
    plt.close("all")

    logger.info(f"Confusion matrix saved to: {CONFUSION_MATRIX_PLOT_PATH}")

    tn, fp, fn, tp = cm.ravel()
    return {
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp)
    }


def plot_roc_curve(y_true: pd.Series, y_prob: np.ndarray) -> Dict[str, float]:
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
    plt.savefig(ROC_CURVE_PATH, dpi=300, bbox_inches="tight")
    plt.close("all")

    logger.info(f"ROC curve saved to: {ROC_CURVE_PATH}")
    return {"roc_auc": float(auc_value), "curve_points": int(len(fpr))}


def plot_precision_recall_curve(y_true: pd.Series, y_prob: np.ndarray) -> Dict[str, float]:
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
    plt.savefig(PR_CURVE_PATH, dpi=300, bbox_inches="tight")
    plt.close("all")

    logger.info(f"Precision-recall curve saved to: {PR_CURVE_PATH}")
    return {
        "pr_auc": float(pr_auc),
        "baseline_positive_rate": baseline,
        "curve_points": int(len(precision))
    }


def plot_calibration_curve(y_true: pd.Series, y_prob: np.ndarray) -> Dict[str, Any]:
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="uniform")

    plt.figure(figsize=(7, 5))
    plt.plot(prob_pred, prob_true, marker="o", label="Model")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect Calibration")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Observed Positive Rate")
    plt.title("Calibration Curve")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(CALIBRATION_CURVE_PATH, dpi=300, bbox_inches="tight")
    plt.close("all")

    logger.info(f"Calibration curve saved to: {CALIBRATION_CURVE_PATH}")

    calibration_error = float(np.mean(np.abs(prob_true - prob_pred))) if len(prob_true) > 0 else None
    return {
        "bins_used": int(len(prob_true)),
        "mean_absolute_calibration_error": calibration_error,
        "prob_true": [float(x) for x in prob_true],
        "prob_pred": [float(x) for x in prob_pred]
    }


def build_error_analysis(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y_prob: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, Any]:
    errors_df = X_test.copy()
    errors_df["actual"] = y_test.values
    errors_df["predicted"] = y_pred
    errors_df["predicted_probability"] = y_prob
    errors_df["error_type"] = "correct"

    errors_df.loc[(errors_df["actual"] == 0) & (errors_df["predicted"] == 1), "error_type"] = "false_positive"
    errors_df.loc[(errors_df["actual"] == 1) & (errors_df["predicted"] == 0), "error_type"] = "false_negative"

    total_cases = int(len(errors_df))
    false_positives = errors_df[errors_df["error_type"] == "false_positive"].copy()
    false_negatives = errors_df[errors_df["error_type"] == "false_negative"].copy()

    hardest_fp = false_positives.sort_values(
        by="predicted_probability", ascending=False
    ).head(10)[["predicted_probability"]].reset_index().rename(columns={"index": "test_index"})

    hardest_fn = false_negatives.sort_values(
        by="predicted_probability", ascending=True
    ).head(10)[["predicted_probability"]].reset_index().rename(columns={"index": "test_index"})

    difficult_cases = errors_df.copy()
    difficult_cases["distance_from_decision_boundary"] = np.abs(errors_df["predicted_probability"] - 0.5)
    difficult_cases = difficult_cases.sort_values(by="distance_from_decision_boundary", ascending=True).head(10)

    error_summary = {
        "dataset_name": DATASET_NAME,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "total_cases": total_cases,
        "total_errors": int((errors_df["actual"] != errors_df["predicted"]).sum()),
        "false_positive_count": int(len(false_positives)),
        "false_negative_count": int(len(false_negatives)),
        "average_false_positive_probability": (
            float(false_positives["predicted_probability"].mean()) if len(false_positives) > 0 else None
        ),
        "average_false_negative_probability": (
            float(false_negatives["predicted_probability"].mean()) if len(false_negatives) > 0 else None
        ),
        "hardest_false_positives_top10": make_json_safe(hardest_fp.to_dict(orient="records")),
        "hardest_false_negatives_top10": make_json_safe(hardest_fn.to_dict(orient="records")),
        "most_uncertain_cases": make_json_safe(
            difficult_cases[[
                "actual", "predicted", "predicted_probability",
                "error_type", "distance_from_decision_boundary"
            ]].to_dict(orient="records")
        ),
        "clinical_note": "False negatives matter because missed diabetes risk can delay early intervention; false positives matter because they can trigger unnecessary anxiety or follow-up testing."
    }

    save_json(error_summary, ERROR_ANALYSIS_PATH)
    logger.info(f"Error analysis saved to: {ERROR_ANALYSIS_PATH}")
    return error_summary


# ---------------------------------------------------
# Evaluation
# ---------------------------------------------------

def evaluate_model(
    model: Any,
    X_test_ready,
    X_test_raw: pd.DataFrame,
    y_test: pd.Series,
    saved_threshold: float | None = None
) -> Dict[str, Any]:
    logger.info("Generating probability predictions...")
    y_prob = model.predict_proba(X_test_ready)[:, 1]

    analysis_threshold, threshold_df = run_threshold_analysis(y_test, y_prob)
    final_threshold = float(saved_threshold) if saved_threshold is not None else analysis_threshold

    metrics = calculate_metrics(y_test, y_prob, final_threshold)
    y_pred = (y_prob >= final_threshold).astype(int)

    classification_report_json = generate_classification_report_json(y_test, y_pred)
    confusion_matrix_counts = plot_confusion_matrix(y_test, y_pred, final_threshold)
    roc_summary = plot_roc_curve(y_test, y_prob)
    pr_summary = plot_precision_recall_curve(y_test, y_prob)
    calibration_summary = plot_calibration_curve(y_test, y_prob)
    error_analysis = build_error_analysis(X_test_raw, y_test, y_prob, y_pred)

    logger.info("Final evaluation metrics:")
    for key, value in metrics.items():
        logger.info(f"{key}: {value:.4f}")

    evaluation_summary = {
        "dataset_name": DATASET_NAME,
        "evaluation_generated_at": datetime.utcnow().isoformat() + "Z",
        "saved_training_threshold": float(saved_threshold) if saved_threshold is not None else None,
        "analysis_best_threshold": float(analysis_threshold),
        "final_used_threshold": float(final_threshold),
        "metrics": metrics,
        "classification_report_file": os.path.basename(CLASSIFICATION_REPORT_PATH),
        "threshold_analysis_file": os.path.basename(THRESHOLD_ANALYSIS_PATH),
        "error_analysis_file": os.path.basename(ERROR_ANALYSIS_PATH),
        "plots": {
            "confusion_matrix": os.path.basename(CONFUSION_MATRIX_PLOT_PATH),
            "roc_curve": os.path.basename(ROC_CURVE_PATH),
            "precision_recall_curve": os.path.basename(PR_CURVE_PATH),
            "calibration_curve": os.path.basename(CALIBRATION_CURVE_PATH)
        },
        "confusion_matrix": confusion_matrix_counts,
        "roc_summary": roc_summary,
        "pr_summary": pr_summary,
        "calibration_summary": calibration_summary,
        "top_threshold_rows": make_json_safe(threshold_df.head(10).to_dict(orient="records"))
    }

    save_json(evaluation_summary, EVALUATION_METRICS_PATH)

    return {
        "evaluation_metrics": evaluation_summary,
        "classification_report": classification_report_json,
        "error_analysis": error_analysis
    }


# ---------------------------------------------------
# Main
# ---------------------------------------------------

def run_evaluation_pipeline() -> None:
    logger.info("Starting diabetes evaluation pipeline...")

    df = load_dataset()
    model, preprocessor, selected_features, saved_threshold, metadata = load_registry_artifacts()

    _, X_test, _, y_test = recreate_holdout_split(df)

    X_test_ready = transform_features(
        X=X_test,
        model=model,
        preprocessor_artifact=preprocessor,
        selected_features=selected_features
    )

    evaluate_model(
        model=model,
        X_test_ready=X_test_ready,
        X_test_raw=X_test,
        y_test=y_test,
        saved_threshold=saved_threshold
    )

    logger.info(f"Evaluation artifacts saved in: {EVAL_REGISTRY_DIR}")
    logger.info("Diabetes evaluation pipeline completed successfully.")


if __name__ == "__main__":
    run_evaluation_pipeline()
