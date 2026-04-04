from __future__ import annotations

import os
import json
import logging
import warnings
from datetime import datetime
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif


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
TOP_N = 13
FIG_DPI = 300

DATA_PATH = os.path.join(BASE_DIR, "data", "processed", f"{DATASET_NAME}_clean.csv")
REPORT_DIR = os.path.join(BASE_DIR, "reports", DATASET_NAME, "eda")
MODEL_REGISTRY_DIR = os.path.join(BASE_DIR, "model_registry", DATASET_NAME, "eda")

BASIC_SUMMARY_PATH = os.path.join(REPORT_DIR, "basic_summary.json")
STATISTICAL_SUMMARY_PATH = os.path.join(REPORT_DIR, "statistical_summary.csv")
TARGET_DISTRIBUTION_PLOT_PATH = os.path.join(REPORT_DIR, "target_distribution.png")
CORRELATION_HEATMAP_PATH = os.path.join(REPORT_DIR, "correlation_heatmap.png")
CORRELATION_MATRIX_PATH = os.path.join(REPORT_DIR, "correlation_matrix.csv")
TARGET_CORRELATIONS_PATH = os.path.join(REPORT_DIR, "target_correlations.csv")
MUTUAL_INFORMATION_PATH = os.path.join(REPORT_DIR, "mutual_information_scores.csv")
MUTUAL_INFORMATION_PLOT_PATH = os.path.join(REPORT_DIR, "mutual_information_top13.png")
OUTLIER_SUMMARY_PATH = os.path.join(REPORT_DIR, "outlier_summary.csv")
SKEWNESS_SUMMARY_PATH = os.path.join(REPORT_DIR, "skewness_summary.csv")
CLASSWISE_MEAN_PATH = os.path.join(REPORT_DIR, "classwise_mean_comparison.csv")

EDA_SUMMARY_JSON_PATH = os.path.join(MODEL_REGISTRY_DIR, "eda_summary.json")
EDA_REPORT_TXT_PATH = os.path.join(MODEL_REGISTRY_DIR, "eda_report.txt")

os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(MODEL_REGISTRY_DIR, exist_ok=True)


# ---------------------------------------------------
# Feature Configuration
# ---------------------------------------------------

NUMERICAL_FEATURES = [
    "age",
    "trestbps",
    "chol",
    "thalach",
    "oldpeak",
]

CATEGORICAL_FEATURES = [
    "sex",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "ca",
    "thal",
]

REQUIRED_COLUMNS = NUMERICAL_FEATURES + CATEGORICAL_FEATURES + [TARGET_COLUMN]


# ---------------------------------------------------
# Utility Helpers
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
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    if pd.isna(value):
        return None
    return value


def save_json(data: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    return [col for col in df.columns if col != TARGET_COLUMN]


def get_all_numeric_feature_columns(df: pd.DataFrame) -> List[str]:
    feature_columns = get_feature_columns(df)
    return df[feature_columns].select_dtypes(include=[np.number]).columns.tolist()


def classify_balance_status(class_counts: Dict[str, int]) -> str:
    total = sum(class_counts.values())
    if total == 0:
        return "unknown"

    ratios = [count / total for count in class_counts.values()]
    minority_ratio = min(ratios)

    if minority_ratio < 0.20:
        return "high_imbalance"
    if minority_ratio < 0.35:
        return "moderate_imbalance"
    return "reasonably_balanced"


def safe_close_plot() -> None:
    plt.tight_layout()
    plt.close()


# ---------------------------------------------------
# Load Dataset
# ---------------------------------------------------

def load_data() -> pd.DataFrame:
    logger.info("Loading processed heart disease dataset...")

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    logger.info("Dataset loaded successfully.")
    logger.info(f"Dataset shape: {df.shape}")

    return df


# ---------------------------------------------------
# Dataset Validation
# ---------------------------------------------------

def validate_dataset(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    logger.info("Validating dataset for EDA...")

    if df.empty:
        raise ValueError("Dataset is empty.")

    if df.shape[0] < 10:
        raise ValueError("Dataset has too few rows for stable EDA.")

    if df.shape[1] < 2:
        raise ValueError("Dataset must contain at least one feature and one target column.")

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Missing target column: {TARGET_COLUMN}")

    if df.isnull().sum().sum() != 0:
        raise ValueError("Missing values found in processed dataset. Clean dataset expected.")

    unique_targets = sorted(df[TARGET_COLUMN].dropna().unique().tolist())
    if unique_targets != [0, 1]:
        raise ValueError(f"Unexpected target values: {unique_targets}")

    target_counts = df[TARGET_COLUMN].value_counts().to_dict()
    if int(target_counts.get(0, 0)) == 0 or int(target_counts.get(1, 0)) == 0:
        raise ValueError("Both target classes must be present for EDA.")

    feature_columns = get_feature_columns(df)
    numeric_features = [col for col in NUMERICAL_FEATURES if col in feature_columns]
    categorical_features = [col for col in CATEGORICAL_FEATURES if col in feature_columns]
    all_numeric_features = get_all_numeric_feature_columns(df)

    missing_required = sorted(list(set(REQUIRED_COLUMNS) - set(df.columns)))
    if missing_required:
        raise ValueError(f"Required heart features missing: {missing_required}")

    if not numeric_features:
        raise ValueError("No core numerical features available for EDA.")

    if not all_numeric_features:
        raise ValueError("No numeric features available for correlation/MI analysis.")

    duplicate_rows = int(df.duplicated().sum())
    if duplicate_rows > 0:
        logger.warning(f"Duplicate rows still present in cleaned dataset: {duplicate_rows}")

    logger.info("Dataset validation successful.")
    return numeric_features, categorical_features, all_numeric_features


# ---------------------------------------------------
# Basic Summary
# ---------------------------------------------------

def generate_basic_summary(
    df: pd.DataFrame,
    numeric_features: List[str],
    categorical_features: List[str],
    all_numeric_features: List[str],
) -> Dict[str, Any]:
    logger.info("Generating basic summary...")

    target_distribution = {
        str(k): int(v) for k, v in df[TARGET_COLUMN].value_counts().sort_index().to_dict().items()
    }

    target_rate_percent = {
        str(k): round((v / len(df)) * 100, 4)
        for k, v in df[TARGET_COLUMN].value_counts().sort_index().to_dict().items()
    }

    summary = {
        "dataset_name": DATASET_NAME,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "shape": {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
        },
        "target_column": TARGET_COLUMN,
        "feature_count": int(len(get_feature_columns(df))),
        "numeric_feature_count": int(len(all_numeric_features)),
        "categorical_feature_count": int(len(categorical_features)),
        "core_numerical_features": numeric_features,
        "categorical_features": categorical_features,
        "all_numeric_features": all_numeric_features,
        "target_distribution": target_distribution,
        "target_rate_percent": target_rate_percent,
        "class_balance_status": classify_balance_status(target_distribution),
        "missing_values": {
            col: int(val) for col, val in df.isnull().sum().to_dict().items()
        },
        "data_types": {
            col: str(dtype) for col, dtype in df.dtypes.items()
        },
        "duplicate_rows": int(df.duplicated().sum()),
    }

    df.describe(include="all").T.to_csv(STATISTICAL_SUMMARY_PATH)
    save_json(summary, BASIC_SUMMARY_PATH)

    logger.info("Basic summary saved successfully.")
    return summary


# ---------------------------------------------------
# Target Distribution Plot
# ---------------------------------------------------

def plot_target_distribution(df: pd.DataFrame) -> None:
    logger.info("Generating target distribution plot...")

    counts = df[TARGET_COLUMN].value_counts().sort_index()
    labels = ["No Heart Disease (0)", "Heart Disease (1)"]

    plt.figure(figsize=(7, 5), dpi=FIG_DPI)
    plt.bar(labels, counts.values)
    plt.title("Target Distribution")
    plt.ylabel("Count")
    plt.xlabel("Class")
    plt.savefig(TARGET_DISTRIBUTION_PLOT_PATH, bbox_inches="tight")
    safe_close_plot()

    logger.info(f"Target distribution plot saved: {TARGET_DISTRIBUTION_PLOT_PATH}")


# ---------------------------------------------------
# Correlation Analysis
# ---------------------------------------------------

def plot_correlation_heatmap(df: pd.DataFrame, all_numeric_features: List[str]) -> pd.DataFrame:
    logger.info("Generating correlation heatmap...")

    corr_columns = all_numeric_features + [TARGET_COLUMN]
    corr = df[corr_columns].corr(numeric_only=True)

    if corr.empty:
        raise ValueError("Correlation matrix is empty.")

    plt.figure(figsize=(14, 10), dpi=FIG_DPI)
    plt.imshow(corr, aspect="auto")
    plt.colorbar()
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90, fontsize=8)
    plt.yticks(range(len(corr.columns)), corr.columns, fontsize=8)
    plt.title("Feature Correlation Heatmap")
    plt.savefig(CORRELATION_HEATMAP_PATH, bbox_inches="tight")
    safe_close_plot()

    corr.to_csv(CORRELATION_MATRIX_PATH)

    logger.info(f"Correlation outputs saved: {CORRELATION_HEATMAP_PATH}")
    return corr


# ---------------------------------------------------
# Target Correlations
# ---------------------------------------------------

def save_target_correlations(df: pd.DataFrame, all_numeric_features: List[str]) -> pd.DataFrame:
    logger.info("Analyzing feature correlation with target...")

    corr = df[all_numeric_features + [TARGET_COLUMN]].corr(numeric_only=True)

    corr_with_target = (
        corr[TARGET_COLUMN]
        .drop(TARGET_COLUMN)
        .sort_values(key=np.abs, ascending=False)
        .reset_index()
    )
    corr_with_target.columns = ["feature", "correlation_with_target"]
    corr_with_target["absolute_correlation"] = corr_with_target["correlation_with_target"].abs()

    corr_with_target.to_csv(TARGET_CORRELATIONS_PATH, index=False)

    logger.info(f"Target correlation report saved: {TARGET_CORRELATIONS_PATH}")
    return corr_with_target


# ---------------------------------------------------
# Mutual Information
# ---------------------------------------------------

def save_mutual_information(df: pd.DataFrame, all_numeric_features: List[str]) -> pd.DataFrame:
    logger.info("Computing mutual information...")

    X = df[all_numeric_features]
    y = df[TARGET_COLUMN]

    mi_scores = mutual_info_classif(X, y, random_state=RANDOM_STATE)

    mi_df = pd.DataFrame({
        "feature": X.columns,
        "mutual_information": mi_scores,
    }).sort_values("mutual_information", ascending=False).reset_index(drop=True)

    mi_df.to_csv(MUTUAL_INFORMATION_PATH, index=False)

    top_mi = mi_df.head(TOP_N).sort_values("mutual_information", ascending=True)

    plt.figure(figsize=(10, 7), dpi=FIG_DPI)
    plt.barh(top_mi["feature"], top_mi["mutual_information"])
    plt.title(f"Top {TOP_N} Features by Mutual Information")
    plt.xlabel("Mutual Information")
    plt.ylabel("Feature")
    plt.savefig(MUTUAL_INFORMATION_PLOT_PATH, bbox_inches="tight")
    safe_close_plot()

    logger.info(f"Mutual information results saved: {MUTUAL_INFORMATION_PATH}")
    return mi_df


# ---------------------------------------------------
# Outlier Summary
# ---------------------------------------------------

def save_outlier_summary(df: pd.DataFrame, numeric_features: List[str]) -> pd.DataFrame:
    logger.info("Computing outlier summary...")

    records = []

    for col in numeric_features:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outlier_count = int(((df[col] < lower_bound) | (df[col] > upper_bound)).sum())
        outlier_percentage = (outlier_count / len(df)) * 100

        records.append({
            "feature": col,
            "q1": round(float(q1), 4),
            "q3": round(float(q3), 4),
            "iqr": round(float(iqr), 4),
            "lower_bound": round(float(lower_bound), 4),
            "upper_bound": round(float(upper_bound), 4),
            "outlier_count": outlier_count,
            "outlier_percentage": round(float(outlier_percentage), 4),
        })

    outlier_df = pd.DataFrame(records).sort_values(by="outlier_count", ascending=False).reset_index(drop=True)
    outlier_df.to_csv(OUTLIER_SUMMARY_PATH, index=False)

    logger.info(f"Outlier summary saved: {OUTLIER_SUMMARY_PATH}")
    return outlier_df


# ---------------------------------------------------
# Skewness Summary
# ---------------------------------------------------

def save_skewness_summary(df: pd.DataFrame, numeric_features: List[str]) -> pd.DataFrame:
    logger.info("Analyzing skewness...")

    skewness_df = (
        df[numeric_features]
        .skew()
        .sort_values(key=lambda x: np.abs(x), ascending=False)
        .reset_index()
    )
    skewness_df.columns = ["feature", "skewness"]
    skewness_df["absolute_skewness"] = skewness_df["skewness"].abs()

    skewness_df.to_csv(SKEWNESS_SUMMARY_PATH, index=False)

    logger.info(f"Skewness summary saved: {SKEWNESS_SUMMARY_PATH}")
    return skewness_df


# ---------------------------------------------------
# Class-wise Mean Comparison
# ---------------------------------------------------

def save_classwise_means(df: pd.DataFrame, numeric_features: List[str]) -> pd.DataFrame:
    logger.info("Comparing class-wise means...")

    grouped = df.groupby(TARGET_COLUMN)[numeric_features].mean().T

    if 0 not in grouped.columns or 1 not in grouped.columns:
        raise ValueError("Expected both target classes for classwise mean comparison.")

    grouped.columns = ["no_disease_mean", "disease_mean"]
    grouped["absolute_difference"] = (grouped["disease_mean"] - grouped["no_disease_mean"]).abs()
    grouped["relative_difference_percent"] = np.where(
        grouped["no_disease_mean"].abs() > 1e-12,
        (grouped["absolute_difference"] / grouped["no_disease_mean"].abs()) * 100,
        np.nan
    )

    grouped = grouped.sort_values(by="absolute_difference", ascending=False)
    grouped.to_csv(CLASSWISE_MEAN_PATH)

    logger.info(f"Class-wise mean comparison saved: {CLASSWISE_MEAN_PATH}")
    return grouped.reset_index().rename(columns={"index": "feature"})


# ---------------------------------------------------
# Build Registry EDA Summary
# ---------------------------------------------------

def build_eda_summary(
    summary: Dict[str, Any],
    target_corr_df: pd.DataFrame,
    mi_df: pd.DataFrame,
    outlier_df: pd.DataFrame,
    skew_df: pd.DataFrame,
    mean_df: pd.DataFrame,
) -> Dict[str, Any]:
    logger.info("Building structured EDA summary...")

    total_rows = summary["shape"]["rows"]
    target_distribution = summary["target_distribution"]

    no_disease_count = int(target_distribution.get("0", 0))
    disease_count = int(target_distribution.get("1", 0))
    prevalence = round((disease_count / total_rows) * 100, 4) if total_rows else 0.0

    high_skew_df = skew_df[skew_df["absolute_skewness"] > 1]
    high_outlier_df = outlier_df[outlier_df["outlier_count"] > 0]

    recommendations = [
        "Use stratified train-validation splitting to preserve class balance.",
        "Review high-correlation and high-MI features before feature selection decisions.",
        "Use caution with ordinal-encoded cardiac variables during preprocessing and model interpretation.",
        "Handle skewness and outliers during training-time preprocessing rather than altering the cleaned dataset.",
        "Prioritize recall, calibration, and threshold tuning because missed heart disease cases are clinically important.",
    ]

    return {
        "dataset_name": DATASET_NAME,
        "analysis_type": "eda",
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "data_source": DATA_PATH,
        "dataset_overview": {
            "rows": int(summary["shape"]["rows"]),
            "columns": int(summary["shape"]["columns"]),
            "feature_count": int(summary["feature_count"]),
            "numeric_feature_count": int(summary["numeric_feature_count"]),
            "categorical_feature_count": int(summary["categorical_feature_count"]),
            "duplicate_rows": int(summary["duplicate_rows"]),
            "missing_values_total": int(sum(summary["missing_values"].values())),
        },
        "target_column": TARGET_COLUMN,
        "heart_disease_prevalence_percent": prevalence,
        "class_balance": {
            "no_disease_count": no_disease_count,
            "disease_count": disease_count,
            "no_disease_rate_percent": round((no_disease_count / total_rows) * 100, 4) if total_rows else 0.0,
            "disease_rate_percent": round((disease_count / total_rows) * 100, 4) if total_rows else 0.0,
            "balance_status": summary["class_balance_status"],
        },
        "feature_health": {
            "features_with_outliers_count": int((high_outlier_df["outlier_count"] > 0).sum()),
            "features_with_high_skew_count": int((high_skew_df["absolute_skewness"] > 1).sum()),
        },
        "top_features_by_absolute_correlation": [
            {
                "feature": row["feature"],
                "correlation_with_target": to_serializable(row["correlation_with_target"]),
                "absolute_correlation": to_serializable(row["absolute_correlation"]),
            }
            for _, row in target_corr_df.head(10).iterrows()
        ],
        "top_features_by_mutual_information": [
            {
                "feature": row["feature"],
                "mutual_information": to_serializable(row["mutual_information"]),
            }
            for _, row in mi_df.head(10).iterrows()
        ],
        "top_features_by_class_mean_difference": [
            {
                "feature": row["feature"],
                "absolute_difference": to_serializable(row["absolute_difference"]),
                "relative_difference_percent": to_serializable(row["relative_difference_percent"]),
            }
            for _, row in mean_df.head(10).iterrows()
        ],
        "top_outlier_features": [
            {
                "feature": row["feature"],
                "outlier_count": int(row["outlier_count"]),
                "outlier_percentage": to_serializable(row["outlier_percentage"]),
            }
            for _, row in outlier_df.head(10).iterrows()
        ],
        "high_skew_features": [
            {
                "feature": row["feature"],
                "skewness": to_serializable(row["skewness"]),
                "absolute_skewness": to_serializable(row["absolute_skewness"]),
            }
            for _, row in high_skew_df.head(10).iterrows()
        ],
        "training_recommendations": recommendations,
        "artifacts": {
            "basic_summary_json": BASIC_SUMMARY_PATH,
            "statistical_summary_csv": STATISTICAL_SUMMARY_PATH,
            "target_distribution_png": TARGET_DISTRIBUTION_PLOT_PATH,
            "correlation_heatmap_png": CORRELATION_HEATMAP_PATH,
            "correlation_matrix_csv": CORRELATION_MATRIX_PATH,
            "target_correlations_csv": TARGET_CORRELATIONS_PATH,
            "mutual_information_scores_csv": MUTUAL_INFORMATION_PATH,
            "mutual_information_top15_png": MUTUAL_INFORMATION_PLOT_PATH,
            "outlier_summary_csv": OUTLIER_SUMMARY_PATH,
            "skewness_summary_csv": SKEWNESS_SUMMARY_PATH,
            "classwise_mean_comparison_csv": CLASSWISE_MEAN_PATH,
            "eda_summary_json": EDA_SUMMARY_JSON_PATH,
            "eda_report_txt": EDA_REPORT_TXT_PATH,
        },
    }


# ---------------------------------------------------
# Human-Readable EDA Report
# ---------------------------------------------------

def build_eda_report_text(
    summary: Dict[str, Any],
    target_corr_df: pd.DataFrame,
    mi_df: pd.DataFrame,
    outlier_df: pd.DataFrame,
    skew_df: pd.DataFrame,
    mean_df: pd.DataFrame,
) -> str:
    logger.info("Building human-readable EDA report...")

    total_rows = summary["shape"]["rows"]
    target_distribution = summary["target_distribution"]

    no_disease_rate = (target_distribution.get("0", 0) / total_rows) * 100 if total_rows else 0.0
    disease_rate = (target_distribution.get("1", 0) / total_rows) * 100 if total_rows else 0.0

    top_corr_features = target_corr_df.head(5)["feature"].tolist()
    top_mi_features = mi_df.head(5)["feature"].tolist()
    top_mean_gap_features = mean_df.head(5)["feature"].tolist()
    high_outlier_features = outlier_df[outlier_df["outlier_count"] > 0].head(5)["feature"].tolist()
    highly_skewed_features = skew_df[skew_df["absolute_skewness"] > 1].head(5)["feature"].tolist()

    lines = [
        "HEART DISEASE - EXPLORATORY DATA ANALYSIS INSIGHT REPORT",
        "=" * 80,
        "",
        "1. DATASET OVERVIEW",
        "-" * 80,
        f"Dataset Name              : {DATASET_NAME}",
        f"Generated At              : {datetime.utcnow().isoformat()}Z",
        f"Total Rows                : {summary['shape']['rows']}",
        f"Total Columns             : {summary['shape']['columns']}",
        f"Feature Count             : {summary['feature_count']}",
        f"Numeric Features          : {summary['numeric_feature_count']}",
        f"Categorical Features      : {summary['categorical_feature_count']}",
        f"Duplicate Rows            : {summary['duplicate_rows']}",
        f"Missing Values Total      : {sum(summary['missing_values'].values())}",
        "",
        "2. TARGET DISTRIBUTION",
        "-" * 80,
        f"No Disease (0) Count      : {target_distribution.get('0', 0)}",
        f"Disease (1) Count         : {target_distribution.get('1', 0)}",
        f"No Disease (0) Rate       : {no_disease_rate:.2f}%",
        f"Disease (1) Rate          : {disease_rate:.2f}%",
        f"Class Balance Status      : {summary['class_balance_status']}",
        "",
        "3. STRONGEST PREDICTIVE SIGNALS",
        "-" * 80,
        "Top 5 Features by Absolute Correlation with Target:",
        ", ".join(top_corr_features) if top_corr_features else "None",
        "",
        "Top 5 Features by Mutual Information:",
        ", ".join(top_mi_features) if top_mi_features else "None",
        "",
        "Top 5 Features by Class-wise Mean Difference:",
        ", ".join(top_mean_gap_features) if top_mean_gap_features else "None",
        "",
        "4. DISTRIBUTION HEALTH",
        "-" * 80,
        "Top 5 Features with Most Outliers:",
        ", ".join(high_outlier_features) if high_outlier_features else "None",
        "",
        "Highly Skewed Features (|skew| > 1):",
        ", ".join(highly_skewed_features) if highly_skewed_features else "None",
        "",
        "5. CLINICAL / MODELING INTERPRETATION",
        "-" * 80,
        "Cardiac variables such as chest pain type, major vessel count, thal, exercise-induced angina, oldpeak, and maximum heart rate can carry strong diagnostic signal.",
        "Because missed heart disease cases are clinically costly, recall, calibration, and threshold tuning should be prioritized.",
        "Encoded categorical/ordinal features should be interpreted carefully during modeling and explainability.",
        "",
        "6. TRAINING RECOMMENDATIONS",
        "-" * 80,
        "Use stratified splitting for train/validation/test partitions.",
        "Review multicollinearity and strong signal features before feature selection.",
        "Evaluate ROC-AUC, PR-AUC, recall, specificity, calibration, and threshold sensitivity.",
        "Consider both linear and tree-based models, with careful preprocessing alignment.",
        "",
        "7. GOVERNANCE NOTE",
        "-" * 80,
        "This EDA report is generated for project auditability, model registry traceability, and downstream training alignment.",
    ]

    return "\n".join(lines)


# ---------------------------------------------------
# Save Registry Artifacts
# ---------------------------------------------------

def save_registry_eda_artifacts(eda_summary: Dict[str, Any], eda_report_text: str) -> None:
    save_json(eda_summary, EDA_SUMMARY_JSON_PATH)

    with open(EDA_REPORT_TXT_PATH, "w", encoding="utf-8") as f:
        f.write(eda_report_text)

    logger.info(f"EDA summary saved in model registry: {EDA_SUMMARY_JSON_PATH}")
    logger.info(f"EDA report saved in model registry: {EDA_REPORT_TXT_PATH}")


# ---------------------------------------------------
# Main Pipeline
# ---------------------------------------------------

def run_eda_pipeline() -> None:
    logger.info("Starting Heart Disease EDA pipeline...")

    df = load_data()
    numeric_features, categorical_features, all_numeric_features = validate_dataset(df)

    summary = generate_basic_summary(
        df=df,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        all_numeric_features=all_numeric_features,
    )

    plot_target_distribution(df)
    plot_correlation_heatmap(df, all_numeric_features)

    target_corr_df = save_target_correlations(df, all_numeric_features)
    mi_df = save_mutual_information(df, all_numeric_features)
    outlier_df = save_outlier_summary(df, numeric_features)
    skew_df = save_skewness_summary(df, numeric_features)
    mean_df = save_classwise_means(df, numeric_features)

    eda_summary = build_eda_summary(
        summary=summary,
        target_corr_df=target_corr_df,
        mi_df=mi_df,
        outlier_df=outlier_df,
        skew_df=skew_df,
        mean_df=mean_df,
    )

    eda_report_text = build_eda_report_text(
        summary=summary,
        target_corr_df=target_corr_df,
        mi_df=mi_df,
        outlier_df=outlier_df,
        skew_df=skew_df,
        mean_df=mean_df,
    )

    save_registry_eda_artifacts(
        eda_summary=eda_summary,
        eda_report_text=eda_report_text,
    )

    logger.info(f"EDA completed successfully. Reports saved in: {REPORT_DIR}")
    logger.info(f"Model registry EDA files saved in: {MODEL_REGISTRY_DIR}")


# ---------------------------------------------------
# Main Runner
# ---------------------------------------------------

if __name__ == "__main__":
    run_eda_pipeline()
