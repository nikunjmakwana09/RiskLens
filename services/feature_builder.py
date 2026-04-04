from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    from services.model_registry import ModelRegistry, ValidationError, create_registry
except ModuleNotFoundError:
    from model_registry import ModelRegistry, ValidationError, create_registry


# =============================================================================
# Logging
# =============================================================================

LOGGER_NAME = "feature_builder"
logger = logging.getLogger(LOGGER_NAME)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


# =============================================================================
# Exceptions
# =============================================================================

class FeatureBuilderError(Exception):
    """Base exception for feature building errors."""


class UnsupportedDiseaseError(FeatureBuilderError):
    """Raised when disease is not supported."""


class RawInputValidationError(FeatureBuilderError):
    """Raised when raw input payload is invalid."""


class FeatureEngineeringError(FeatureBuilderError):
    """Raised when disease feature engineering fails."""


# =============================================================================
# Helper Dataclasses
# =============================================================================

@dataclass
class FeatureBuildResult:
    disease: str
    raw_payload: Dict[str, Any]
    normalized_payload: Dict[str, Any]
    engineered_payload: Dict[str, Any]
    aligned_features: List[str]
    dataframe: pd.DataFrame


# =============================================================================
# Utility Functions
# =============================================================================

def _safe_divide(a: float, b: float, default: float = 0.0) -> float:
    """Safely divide two numbers."""
    try:
        if b in (0, 0.0, None):
            return default
        return float(a) / float(b)
    except Exception:
        return default


def _safe_log1p(x: float, default: float = 0.0) -> float:
    """Safely compute log(1+x)."""
    try:
        x = float(x)
        if x < -1:
            return default
        return float(np.log1p(x))
    except Exception:
        return default


def _to_float(value: Any, default: float = 0.0) -> float:
    """Convert value to float safely."""
    try:
        if value is None:
            return default
        if isinstance(value, bool):
            return float(int(value))
        if isinstance(value, (int, float, np.integer, np.floating)):
            return float(value)

        value_str = str(value).strip()
        if value_str == "":
            return default

        value_str = value_str.replace(",", "")
        return float(value_str)
    except Exception:
        return default


def _to_int(value: Any, default: int = 0) -> int:
    """Convert value to int safely."""
    try:
        return int(round(_to_float(value, default=default)))
    except Exception:
        return default


def _normalize_key(key: str) -> str:
    """Normalize raw keys for loose matching."""
    return re.sub(r"[^a-z0-9]", "", str(key).strip().lower())


def _is_missing(value: Any) -> bool:
    """Check whether a value should be treated as missing."""
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    return False


def _get_first_present(
    payload: Dict[str, Any],
    candidate_keys: List[str],
    default: Any = None,
) -> Any:
    """
    Retrieve first matching value from payload using normalized key matching.
    """
    normalized_map = {_normalize_key(k): v for k, v in payload.items()}

    for key in candidate_keys:
        nk = _normalize_key(key)
        if nk in normalized_map and not _is_missing(normalized_map[nk]):
            return normalized_map[nk]

    return default


def _canonicalize_binary(
    value: Any,
    positive_values: List[str],
    negative_values: List[str],
    default: int = 0,
) -> int:
    """Canonicalize binary/categorical yes-no style field."""
    if isinstance(value, (int, float, np.integer, np.floating)):
        return 1 if float(value) >= 1 else 0

    value_str = str(value).strip().lower()
    if value_str in [v.lower() for v in positive_values]:
        return 1
    if value_str in [v.lower() for v in negative_values]:
        return 0
    return default


def _coerce_payload_to_numeric(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Best-effort conversion of values that look numeric.
    Leaves non-numeric strings unchanged for encoded/categorical flows.
    """
    output = {}
    for key, value in payload.items():
        if isinstance(value, (int, float, np.integer, np.floating, bool)):
            output[key] = float(value) if not isinstance(value, bool) else int(value)
            continue

        if isinstance(value, str):
            v = value.strip()
            if v == "":
                output[key] = value
                continue
            try:
                output[key] = float(v.replace(",", ""))
            except Exception:
                output[key] = value
            continue

        output[key] = value

    return output


# =============================================================================
# Main Feature Builder
# =============================================================================

class FeatureBuilder:
    """
    Enterprise-grade feature builder for all diseases.

    Usage
    -----
    registry = ModelRegistry("model_registry")
    registry.discover(strict=True)

    builder = FeatureBuilder(registry)
    result = builder.build("diabetes", payload)

    df = result.dataframe
    """

    SUPPORTED_DISEASES = {
        "breast_cancer",
        "diabetes",
        "heart"
    }

    def __init__(self, registry: ModelRegistry, strict: bool = True):
        self.registry = registry
        self.strict = strict

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def build(self, disease: str, payload: Dict[str, Any]) -> FeatureBuildResult:

        disease = str(disease).strip()

        # Step 1: basic validation
        self._validate_disease_supported(disease)
        self._validate_raw_payload(payload)

        logger.info("Building features for disease: %s", disease)

        # Step 2: normalization
        normalized_payload = self._normalize_payload(disease, payload)

        # Step 3: range validation
        self._validate_ranges(disease, normalized_payload)

        # Step 4: feature engineering
        engineered_payload = self._apply_feature_engineering(disease, normalized_payload)

        # Step 5: align with registry
        aligned_payload, aligned_features = self._align_to_registry(
            disease, engineered_payload
        )

        self._validate_required_fields(disease, normalized_payload)

        # Step 6: final validation
        self._validate_final_features(aligned_payload)

        # Step 7: debug logs
        logger.debug(f"Normalized payload: {normalized_payload}")
        logger.debug(f"Engineered payload keys: {list(engineered_payload.keys())}")
        logger.debug(f"Aligned features: {aligned_features}")

        # Step 8: dataframe
        dataframe = pd.DataFrame([aligned_payload], columns=aligned_features)

        logger.info(
            "Feature building complete for disease='%s' | aligned_features=%d",
            disease,
            len(aligned_features),
        )

        return FeatureBuildResult(
            disease=disease,
            raw_payload=payload,
            normalized_payload=normalized_payload,
            engineered_payload=aligned_payload,
            aligned_features=aligned_features,
            dataframe=dataframe,
        )


    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------

    def _validate_disease_supported(self, disease: str) -> None:
        if disease not in self.SUPPORTED_DISEASES:
            raise UnsupportedDiseaseError(
                f"Unsupported disease '{disease}'. "
                f"Supported diseases: {sorted(self.SUPPORTED_DISEASES)}"
            )

    def _validate_raw_payload(self, payload: Dict[str, Any]) -> None:
        if not isinstance(payload, dict):
            raise RawInputValidationError("Input payload must be a dictionary.")
        if not payload:
            raise RawInputValidationError("Input payload cannot be empty.")

    def _validate_final_features(self, payload: Dict[str, Any]) -> None:

        for key, value in payload.items():

            if value is None:
                raise ValidationError(f"Feature '{key}' has None value.")

            if isinstance(value, float) and np.isnan(value):
                raise ValidationError(f"Feature '{key}' has NaN value.")

    def _validate_ranges(self, disease: str, payload: Dict[str, Any]) -> None:

        if disease == "diabetes":
            if payload["Glucose"] <= 0 or payload["Glucose"] > 500:
                raise ValidationError("Invalid Glucose value")
            if payload["BMI"] < 10 or payload["BMI"] > 80:
                raise ValidationError("Invalid BMI value")
            if payload["Age"] <= 0 or payload["Age"] > 120:
                raise ValidationError("Invalid Age value")

        elif disease == "heart":
            if payload["age"] <= 0 or payload["age"] > 120:
                raise ValidationError("Invalid age")
            if payload["chol"] <= 0 or payload["chol"] > 600:
                raise ValidationError("Invalid cholesterol")
            if payload["trestbps"] <= 0 or payload["trestbps"] > 300:
                raise ValidationError("Invalid blood pressure")

        elif disease == "breast_cancer":
            if payload["radius_mean"] <= 0:
                raise ValidationError("Invalid radius_mean")


    def _validate_required_fields(self, disease: str, payload: Dict[str, Any]) -> None:

        REQUIRED_FIELDS = {
            "diabetes": ["Glucose", "BMI", "Age"],
            "heart": ["age", "chol", "trestbps", "thalach"],
            "breast_cancer": ["radius_mean", "texture_mean"]
        }

        required = REQUIRED_FIELDS.get(disease, [])

        missing = [f for f in required if _is_missing(payload.get(f))]

        if missing:
            raise ValidationError(
                f"Missing required fields for '{disease}': {missing}"
            )


    # -------------------------------------------------------------------------
    # Normalization
    # -------------------------------------------------------------------------

    def _normalize_payload(self, disease: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Disease-specific normalization into canonical training-style raw columns.
        """
        payload = _coerce_payload_to_numeric(payload)

        normalizer_map = {
            "diabetes": self._normalize_diabetes,
            "heart": self._normalize_heart,
            "breast_cancer": self._normalize_breast_cancer
        }

        return normalizer_map[disease](payload)

    # -------------------------------------------------------------------------
    # Feature Engineering Dispatcher
    # -------------------------------------------------------------------------

    def _apply_feature_engineering(self, disease: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply disease-specific feature engineering.
        """
        builder_map = {
            "diabetes": self._build_diabetes_features,
            "heart": self._build_heart_features,
            "breast_cancer": self._build_breast_cancer_features
        }

        try:
            return builder_map[disease](payload)
        except Exception as exc:
            logger.exception("Feature engineering failed for disease='%s'", disease)
            raise FeatureEngineeringError(
                f"Feature engineering failed for disease='{disease}': {exc}"
            ) from exc

    # -------------------------------------------------------------------------
    # Registry Alignment
    # -------------------------------------------------------------------------

    def _align_to_registry(
        self,
        disease: str,
        engineered_payload: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], List[str]]:

        required_features = self.registry.get_required_input_features(disease)

        if not required_features:
            raise ValidationError(
                f"No required input features found in registry for disease '{disease}'."
            )

        missing_features = [
            f for f in required_features if f not in engineered_payload
        ]

        # ✅ FIX: strict mode handling
        if missing_features:
            if self.strict:
                raise ValidationError(
                    f"Missing required features for disease '{disease}': {missing_features}"
                )
            else:
                logger.warning(
                    f"Missing features for disease '{disease}', filling with 0: {missing_features}"
                )
                for f in missing_features:
                    engineered_payload[f] = 0.0

        aligned_payload = {f: engineered_payload[f] for f in required_features}

        # ✅ safety check
        if len(aligned_payload) != len(required_features):
            raise ValidationError("Feature alignment mismatch detected")

        return aligned_payload, required_features

    # =============================================================================
    # DIABETES
    # =============================================================================

    def _normalize_diabetes(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize diabetes raw payload to canonical columns.
        Typical base columns:
        - Pregnancies
        - Glucose
        - BloodPressure
        - SkinThickness
        - Insulin
        - BMI
        - DiabetesPedigreeFunction
        - Age
        """
        normalized = {
            "Pregnancies": _to_float(_get_first_present(payload, ["Pregnancies", "pregnancy_count"]), 0.0),
            "Glucose": _to_float(_get_first_present(payload, ["Glucose", "glucose", "blood_glucose"]), 0.0),
            "BloodPressure": _to_float(_get_first_present(payload, ["BloodPressure", "blood_pressure", "bp"]), 0.0),
            "SkinThickness": _to_float(_get_first_present(payload, ["SkinThickness", "skin_thickness"]), 0.0),
            "Insulin": _to_float(_get_first_present(payload, ["Insulin", "insulin"]), 0.0),
            "BMI": _to_float(_get_first_present(payload, ["BMI", "bmi", "body_mass_index"]), 0.0),
            "DiabetesPedigreeFunction": _to_float(
                _get_first_present(payload, ["DiabetesPedigreeFunction", "dpf", "pedigree", "diabetes_pedigree_function"]),
                0.0,
            ),
            "Age": _to_float(_get_first_present(payload, ["Age", "age"]), 0.0),
        }
        return normalized

    def _build_diabetes_features(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        p = dict(payload)
        eps = 1e-6

        pregnancies = _to_float(p.get("Pregnancies", 0.0))
        glucose = _to_float(p.get("Glucose", 0.0))
        insulin = _to_float(p.get("Insulin", 0.0))
        bmi = _to_float(p.get("BMI", 0.0))
        dpf = _to_float(p.get("DiabetesPedigreeFunction", 0.0))
        age = _to_float(p.get("Age", 0.0))

        p["Glucose_BMI"] = glucose * bmi
        p["Glucose_Age"] = glucose * age
        p["BMI_Age"] = bmi * age
        p["Insulin_Glucose_Ratio"] = insulin / (glucose + eps)
        p["BMI_Age_Ratio"] = bmi / (age + eps)
        p["Pregnancies_Age_Ratio"] = pregnancies / (age + eps)
        p["DPF_Age_Interaction"] = dpf * age
        p["Metabolic_Load"] = (glucose + bmi + insulin) / 3.0

        return p

    # =============================================================================
    # HEART
    # =============================================================================

    def _normalize_heart(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize heart disease payload.
        Common canonical columns from standard heart dataset:
        - age, sex, cp, trestbps, chol, fbs, restecg
        - thalach, exang, oldpeak, slope, ca, thal
        """
        normalized = {
            "age": _to_float(_get_first_present(payload, ["age", "Age"]), 0.0),
            "sex": _to_float(_get_first_present(payload, ["sex", "Sex", "gender"]), 0.0),
            "cp": _to_float(_get_first_present(payload, ["cp", "chest_pain_type", "chestpain"]), 0.0),
            "trestbps": _to_float(_get_first_present(payload, ["trestbps", "resting_blood_pressure", "blood_pressure"]), 0.0),
            "chol": _to_float(_get_first_present(payload, ["chol", "cholesterol", "serum_cholesterol"]), 0.0),
            "fbs": _to_float(_get_first_present(payload, ["fbs", "fasting_blood_sugar"]), 0.0),
            "restecg": _to_float(_get_first_present(payload, ["restecg", "resting_ecg"]), 0.0),
            "thalach": _to_float(_get_first_present(payload, ["thalach", "max_heart_rate", "maxheartrate"]), 0.0),
            "exang": _to_float(_get_first_present(payload, ["exang", "exercise_induced_angina"]), 0.0),
            "oldpeak": _to_float(_get_first_present(payload, ["oldpeak", "st_depression"]), 0.0),
            "slope": _to_float(_get_first_present(payload, ["slope", "st_slope"]), 0.0),
            "ca": _to_float(_get_first_present(payload, ["ca", "num_major_vessels", "major_vessels"]), 0.0),
            "thal": _to_float(_get_first_present(payload, ["thal", "thalassemia"]), 0.0),
        }
        return normalized

    def _build_heart_features(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        p = dict(payload)

        age = _to_float(p.get("age", 0.0))
        thalach = _to_float(p.get("thalach", 0.0))
        chol = _to_float(p.get("chol", 0.0))
        trestbps = _to_float(p.get("trestbps", 0.0))

        p["age_thalach_ratio"] = age / (thalach + 1e-6)
        p["chol_age_ratio"] = chol / (age + 1e-6)
        p["bp_age_ratio"] = trestbps / (age + 1e-6)

        return p


    # =============================================================================
    # BREAST CANCER
    # =============================================================================

    def _normalize_breast_cancer(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize breast cancer input.
        Supports common Wisconsin-like features.
        """
        canonical_keys = {
            "radius_mean": ["radius_mean", "mean_radius"],
            "texture_mean": ["texture_mean", "mean_texture"],
            "perimeter_mean": ["perimeter_mean", "mean_perimeter"],
            "area_mean": ["area_mean", "mean_area"],
            "smoothness_mean": ["smoothness_mean", "mean_smoothness"],
            "compactness_mean": ["compactness_mean", "mean_compactness"],
            "concavity_mean": ["concavity_mean", "mean_concavity"],
            "symmetry_mean": ["symmetry_mean", "mean_symmetry"],
            "fractal_dimension_mean": ["fractal_dimension_mean", "mean_fractal_dimension"],
            "radius_se": ["radius_se"],
            "texture_se": ["texture_se"],
            "perimeter_se": ["perimeter_se"],
            "area_se": ["area_se"],
            "smoothness_se": ["smoothness_se"],
            "compactness_se": ["compactness_se"],
            "concavity_se": ["concavity_se"],
            "symmetry_se": ["symmetry_se"],
            "fractal_dimension_se": ["fractal_dimension_se"],
            "radius_worst": ["radius_worst", "worst_radius"],
            "texture_worst": ["texture_worst", "worst_texture"],
            "perimeter_worst": ["perimeter_worst", "worst_perimeter"],
            "area_worst": ["area_worst", "worst_area"],
            "smoothness_worst": ["smoothness_worst", "worst_smoothness"],
            "compactness_worst": ["compactness_worst", "worst_compactness"],
            "concavity_worst": ["concavity_worst", "worst_concavity"],
            "concave_points_mean": ["concave points_mean", "concave_points_mean", "mean_concave_points"],
            "concave_points_se": ["concave points_se", "concave_points_se"],
            "concave_points_worst": ["concave points_worst", "concave_points_worst", "worst_concave_points"],
            "symmetry_worst": ["symmetry_worst", "worst_symmetry"],
            "fractal_dimension_worst": ["fractal_dimension_worst", "worst_fractal_dimension"],
        }

        normalized = {}
        for canonical_key, aliases in canonical_keys.items():
            normalized[canonical_key] = _to_float(_get_first_present(payload, aliases), 0.0)

        return normalized

    def _build_breast_cancer_features(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        p = dict(payload)
        safe_eps = 1e-6

        radius_mean = _to_float(p.get("radius_mean", 0.0))
        perimeter_mean = _to_float(p.get("perimeter_mean", 0.0))
        area_mean = _to_float(p.get("area_mean", 0.0))
        compactness_mean = _to_float(p.get("compactness_mean", 0.0))
        concavity_mean = _to_float(p.get("concavity_mean", 0.0))
        concave_points_mean = _to_float(
            p.get("concave_points_mean", p.get("concave points_mean", 0.0))
        )
        texture_mean = _to_float(p.get("texture_mean", 0.0))
        symmetry_mean = _to_float(p.get("symmetry_mean", 0.0))
        radius_worst = _to_float(p.get("radius_worst", 0.0))
        perimeter_worst = _to_float(p.get("perimeter_worst", 0.0))
        area_worst = _to_float(p.get("area_worst", 0.0))

        p["radius_perimeter_ratio_mean"] = radius_mean / (perimeter_mean + safe_eps)
        p["area_radius_ratio_mean"] = area_mean / (radius_mean + safe_eps)
        p["radius_worst_to_mean"] = radius_worst / (radius_mean + safe_eps)
        p["perimeter_worst_to_mean"] = perimeter_worst / (perimeter_mean + safe_eps)
        p["area_worst_to_mean"] = area_worst / (area_mean + safe_eps)
        p["compactness_concavity_interaction"] = compactness_mean * concavity_mean
        p["concave_to_concavity_ratio"] = concave_points_mean / (concavity_mean + safe_eps)
        p["texture_symmetry_interaction"] = texture_mean * symmetry_mean

        return p


# =============================================================================
# Convenience Functions
# =============================================================================

def build_features(
    registry: ModelRegistry,
    disease: str,
    payload: Dict[str, Any],
) -> FeatureBuildResult:
    """
    Convenience wrapper for one-shot feature building.
    """
    builder = FeatureBuilder(registry)
    return builder.build(disease, payload)


def build_feature_dataframe(
    registry: ModelRegistry,
    disease: str,
    payload: Dict[str, Any],
) -> pd.DataFrame:
    """
    Convenience wrapper that returns only the aligned dataframe.
    """
    return build_features(registry, disease, payload).dataframe


# =============================================================================
# Example CLI / Local Testing
# =============================================================================

if __name__ == "__main__":
    registry = create_registry("model_registry", strict=True)
    builder = FeatureBuilder(registry)

    sample_payloads = {
        "diabetes": {
            "Pregnancies": 2,
            "Glucose": 150,
            "BloodPressure": 85,
            "SkinThickness": 30,
            "Insulin": 120,
            "BMI": 32.5,
            "DiabetesPedigreeFunction": 0.5,
            "Age": 45,
        },
        "heart": {
            "age": 58,
            "sex": 1,
            "cp": 2,
            "trestbps": 140,
            "chol": 260,
            "fbs": 1,
            "restecg": 1,
            "thalach": 120,
            "exang": 1,
            "oldpeak": 2.3,
            "slope": 1,
            "ca": 2,
            "thal": 3,
        },
        "breast_cancer": {
            "radius_mean": 14.2,
            "texture_mean": 20.1,
            "perimeter_mean": 92.0,
            "area_mean": 654.0,
            "smoothness_mean": 0.11,
            "compactness_mean": 0.14,
            "concavity_mean": 0.12,
            "concave points_mean": 0.08,
            "symmetry_mean": 0.20,
            "fractal_dimension_mean": 0.06,
            "radius_se": 0.4,
            "texture_se": 1.2,
            "perimeter_se": 2.8,
            "area_se": 36.0,
            "smoothness_se": 0.007,
            "compactness_se": 0.03,
            "concavity_se": 0.04,
            "concave points_se": 0.015,
            "symmetry_se": 0.02,
            "fractal_dimension_se": 0.004,
            "radius_worst": 17.8,
            "texture_worst": 26.1,
            "perimeter_worst": 117.0,
            "area_worst": 900.0,
            "smoothness_worst": 0.16,
            "compactness_worst": 0.28,
            "concavity_worst": 0.31,
            "concave points_worst": 0.14,
            "symmetry_worst": 0.29,
            "fractal_dimension_worst": 0.09,
        }
    }

    for disease_name, sample_payload in sample_payloads.items():
        try:
            result = builder.build(disease_name, sample_payload)
            print("\n" + "=" * 100)
            print(f"DISEASE: {disease_name}")
            print("=" * 100)
            print("Aligned features:")
            print(result.aligned_features)
            print("\nDataFrame:")
            print(result.dataframe)
        except Exception as exc:
            print(f"[ERROR] {disease_name}: {exc}")