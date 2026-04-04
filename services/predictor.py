# =============================================================================
# RiskLens Predictor
# =============================================================================

from __future__ import annotations
import pandas as pd
import logging
import time
import uuid
from typing import Any, Dict
import sys
import os
import json
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
try:
    from services.model_registry import ModelRegistry, ValidationError
    from services.feature_builder import FeatureBuilder
except ModuleNotFoundError:
    from model_registry import ModelRegistry, ValidationError
    from feature_builder import FeatureBuilder


# =============================================================================
# Logging
# =============================================================================

logger = logging.getLogger("risklens.predictor")

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)

FEATURE_MEANING = {

    # ===============================
    # 🎗️ BREAST CANCER
    # ===============================
    "radius_mean": "Tumor Size",
    "texture_mean": "Tumor Texture",
    "perimeter_mean": "Tumor Perimeter",
    "area_mean": "Tumor Area",
    "smoothness_mean": "Tumor Smoothness",
    "compactness_mean": "Tumor Density",
    "concavity_mean": "Tumor Irregularity",
    "concave_points_mean": "Cell Structure",
    "symmetry_mean": "Cell Symmetry",
    "fractal_dimension_mean": "Cell Complexity",

    "radius_worst": "Tumor Size",
    "texture_worst": "Tumor Texture",
    "perimeter_worst": "Tumor Perimeter",
    "area_worst": "Tumor Area",
    "smoothness_worst": "Tumor Smoothness",
    "compactness_worst": "Tumor Density",
    "concavity_worst": "Tumor Irregularity",
    "concave_points_worst": "Tumor Shape",
    "symmetry_worst": "Cell Symmetry",
    "fractal_dimension_worst": "Cell Complexity",

    # ===============================
    # 🩸 DIABETES
    # ===============================
    "Pregnancies": "Pregnancy Count",
    "Glucose": "Blood Sugar",
    "BloodPressure": "Blood Pressure",
    "SkinThickness": "Skin Thickness",
    "Insulin": "Insulin Level",
    "BMI": "Body Weight",
    "DiabetesPedigreeFunction": "Genetic Risk",
    "Age": "Age Factor",

    # ===============================
    # ❤️ HEART
    # ===============================
    "age": "Age Factor",
    "sex": "Gender",
    "cp": "Chest Pain Type",
    "trestbps": "Blood Pressure",
    "chol": "Cholesterol",
    "fbs": "Fasting Blood Sugar",
    "restecg": "ECG Result",
    "thalach": "Heart Rate",
    "exang": "Exercise Angina",
    "oldpeak": "Cardiac Stress",
    "slope": "ST Segment Slope",
    "ca": "Blocked Vessels",
    "thal": "Thalassemia"
}


# =============================================================================
# Predictor
# =============================================================================

class RiskPredictor:
    """
    Enterprise-grade disease risk prediction engine.

    Features:
    - Registry-driven inference
    - Feature pipeline integration
    - Healthcare-safe risk interpretation
    - Observability (trace_id, latency)
    - Production-ready API output
    """


    def __init__(self, registry: ModelRegistry): 
        self.registry = registry
        self.feature_builder = FeatureBuilder(registry)

        self._context_cache = {}

        for disease in self.registry.list_diseases():
            try:
                self._context_cache[disease] = self.registry.prepare_prediction_context(disease)
                logger.debug(f"Preloaded: {disease}")
            except Exception as e:
                logger.warning(f"Skipping preload for {disease}: {e}")


    def _get_context(self, disease: str):
        context = self._context_cache.get(disease)

        if context is None:
            logger.debug(f"Loading model for '{disease}' into cache...")
            context = self.registry.prepare_prediction_context(disease)

            self._context_cache[disease] = context

        return context

    # -------------------------------------------------------------------------
    # MAIN PREDICTION
    # -------------------------------------------------------------------------

    def predict(self, disease: str, payload: Dict[str, Any]) -> Dict[str, Any]:

        trace_id = str(uuid.uuid4())
        start_time = time.time()

        try:
            logger.info(f"[{trace_id}] Prediction started | disease={disease}")

            # -----------------------------------------------------------------
            # 1. Input Validation
            # -----------------------------------------------------------------
            validation_report = self.registry.validate_input_payload(
                disease, payload, strict=True
            )

            # -----------------------------------------------------------------
            # 2. Feature Engineering (NO CACHE - CLEAN)
            # -----------------------------------------------------------------
            feature_result = self.feature_builder.build(disease, payload)

            # -----------------------------------------------------------------
            # 3. Load Context (CACHED MODEL)
            # -----------------------------------------------------------------
            context = self._get_context(disease)

            model = context["model"]
            preprocessor_obj = context["preprocessor"]
            threshold = context["threshold"] or 0.5
            metadata = context["metadata"]

            # -----------------------------------------------------------------
            # 4. Prepare Input
            # -----------------------------------------------------------------
            if isinstance(preprocessor_obj, dict):
                fitted_pipeline = preprocessor_obj.get("fitted_feature_pipeline")
                required_columns = preprocessor_obj.get("engineered_feature_columns")

                if fitted_pipeline is None:
                    raise ValueError("Invalid preprocessor")

            else:
                fitted_pipeline = preprocessor_obj
                required_columns = feature_result.aligned_features

            X = pd.DataFrame([feature_result.engineered_payload])

            for col in required_columns:
                if col not in X.columns:
                    X[col] = 0.0

            X = X[required_columns]

            # -----------------------------------------------------------------
            # 5. Transform + Predict
            # -----------------------------------------------------------------
            X_processed = fitted_pipeline.transform(X)
            prob = float(model.predict_proba(X_processed)[0][1])

            # -------------------------------------------------
            # 🔥 MEDICAL RULE-BASED RISK BOOST
            # -------------------------------------------------
            if disease == "diabetes":
                glucose = payload.get("Glucose", 0)
                bmi = payload.get("BMI", 0)
                age = payload.get("Age", 0)
                insulin = payload.get("Insulin", 0)

                # 🚨 Extreme risk condition
                if glucose >= 200 and bmi >= 35:
                    prob = max(prob, 0.85)

                # 🚨 High risk condition
                elif glucose >= 160 and bmi >= 30:
                    prob = max(prob, 0.65)

                # 🚨 Moderate risk
                elif glucose >= 140:
                    prob = max(prob, prob + 0.15)

                # 🚨 Age-based adjustment
                if age >= 50:
                    prob += 0.05

                # 🚨 Insulin imbalance
                if insulin >= 200:
                    prob += 0.05

                # Clamp probability
                prob = min(prob, 1.0)


            # -------------------------------------------------
            # ❤️ HEART RISK RULE BOOST
            # -------------------------------------------------
            if disease == "heart":
                age = payload.get("age", 0)
                cp = payload.get("cp", 0)
                chol = payload.get("chol", 0)
                thalach = payload.get("thalach", 0)
                exang = payload.get("exang", 0)
                oldpeak = payload.get("oldpeak", 0)

                # 🚨 Extreme risk
                if cp == 3 and exang == 1 and oldpeak >= 3:
                    prob = max(prob, 0.85)

                # 🚨 High risk
                elif chol >= 300 and thalach < 120:
                    prob = max(prob, 0.7)

                # 🚨 Moderate risk
                elif oldpeak >= 1.5:
                    prob = max(prob, prob + 0.1)

                # 🚨 Age factor
                if age >= 60:
                    prob += 0.05

                # Clamp
                prob = min(prob, 1.0)

            # -----------------------------------------------------------------
            # 6. Decision
            # -----------------------------------------------------------------
            prediction = "high_risk" if prob >= threshold else "low_risk"
            risk_level = self._get_risk_level(prob)
            confidence_score = self._confidence_score(prob, threshold)
            confidence_label = self._confidence_label(confidence_score)
            confidence_reason = self._confidence_reason(prob)
            risk_timeline = self._predict_risk_timeline(prob)
            doctor_priority = self._doctor_priority(prob)
            # -------------------------------------------------
            # 🧠 EXPLANATION (SHAP)
            # -------------------------------------------------
            try:
                explanation = self._load_explainability(disease)
            except Exception:
                explanation = {"top_features": []}

            human_explanation = self._generate_human_explanation(explanation, disease)
            clinical_explanation = self._generate_clinical_explanation(explanation, disease)
            patient_message = self._generate_patient_message(risk_level)
            # -------------------------------------------------
            # 🧠 RECOMMENDATION
            # -------------------------------------------------
            recommendation = self._generate_recommendation(disease, risk_level)

            latency_ms = round((time.time() - start_time) * 1000, 2)

            report = self._generate_medical_report({
                "disease": disease,
                "report_id": trace_id,
                "prediction": prediction,
                "risk_level": risk_level,
                "probability": round(prob, 4),
                "human_explanation": human_explanation,
                "explanation": explanation,
                "recommendation": recommendation,
                "patient_message": patient_message
            })
            severity = self._get_severity(risk_level)
            personalized_insights = self._generate_personalized_insights(explanation, disease)

            return {
                "status": "success",
                "trace_id": trace_id,

                # ===============================
                # 🎯 CORE PREDICTION
                # ===============================
                "prediction": {
                    "risk_level": risk_level,
                    "severity": severity,
                    "probability": round(prob, 4),
                    "threshold": round(threshold, 2)
                },

                # ===============================
                # 🧠 CONFIDENCE
                # ===============================
                "confidence": {
                    "score": round(confidence_score, 3),
                    "level": confidence_label,
                    "reason": confidence_reason
                },

                # ===============================
                # 🏥 MEDICAL REPORT
                # ===============================
                "medical_report": report,

                # ===============================
                # 🧬 INSIGHTS (USER + DOCTOR)
                # ===============================
                "insights": {
                    "personalized": personalized_insights,
                    "clinical_explanation": clinical_explanation,
                    "risk_context": {
                        "relative_risk": (
                            "Significantly higher than baseline population risk"
                            if risk_level in ["high", "critical"]
                            else "Within expected population range"
                        ),
                        "clinical_interpretation": "Requires medical evaluation"
                    }
                },

                # ===============================
                # 🚀 ADVANCED ANALYSIS (UNIQUE)
                # ===============================
                "advanced_analysis": {
                    "risk_timeline": risk_timeline,
                    "risk_breakdown": self._risk_breakdown(disease, feature_result.raw_payload),
                    "preventable_risk": f"{int(self._calculate_preventable_risk(disease, feature_result.raw_payload)*100)}%",
                    "doctor_priority": doctor_priority
                },
                "decision_support": {
                    "recommended_specialist": (
                        "Oncologist" if disease == "breast_cancer"
                        else "Endocrinologist" if disease == "diabetes"
                        else "Cardiologist"
                    ),
                    "care_level": (
                        "emergency" if risk_level == "critical"
                        else "urgent" if risk_level == "high"
                        else "routine"
                    )
                },
                # ===============================
                # ⚙️ SYSTEM INFO
                # ===============================
                "system": {
                    "model": {
                        "name": metadata["model_metadata"].get("model_name"),
                        "version": metadata["model_metadata"].get("version", "1.0"),
                    },
                    "features": {
                        "input": len(feature_result.raw_payload),
                        "engineered": len(feature_result.engineered_payload),
                        "aligned": len(feature_result.aligned_features),
                    },
                    "validation": validation_report
                },

                # ===============================
                # ⏱ META
                # ===============================
                "meta": {
                    "latency_ms": latency_ms,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                },

                # ===============================
                # ⚠️ DISCLAIMER
                # ===============================
                "disclaimer": "This AI prediction is not a medical diagnosis. Consult a qualified healthcare professional."
            }

        except ValidationError as ve:
            return self._error_response(trace_id, "validation_error", str(ve))

        except Exception as exc:
            logger.exception(f"[{trace_id}] Prediction failed")
            return self._error_response(trace_id, "internal_error", str(exc))


    # -------------------------------------------------------------------------
    # CONFIDENCE SCORE (NUMERIC)
    # -------------------------------------------------------------------------

    def _confidence_score(self, prob: float, threshold: float) -> float:
        return round(prob, 3)

    def _confidence_label(self, score):
        if score > 0.8:
            return "very_high"
        elif score > 0.6:
            return "high"
        elif score > 0.4:
            return "moderate"
        else:
            return "low"


    # -------------------------------------------------------------------------
    # HEALTHCARE RISK LEVEL
    # -------------------------------------------------------------------------

    def _get_risk_level(self, prob: float) -> str:
        if prob < 0.2:
            return "very_low"
        elif prob < 0.4:
            return "low"
        elif prob < 0.6:
            return "moderate"
        elif prob < 0.8:
            return "high"
        else:
            return "critical"


    # -------------------------------------------------------------------------
    # ERROR HANDLER
    # -------------------------------------------------------------------------

    def _error_response(self, trace_id: str, error_type: str, message: str) -> Dict[str, Any]:
        return {
            "status": "error",
            "trace_id": trace_id,
            "error_type": error_type,
            "message": message,
        }


    # -------------------------------------------------------------------------
    # 🧠 MEDICAL RECOMMENDATION ENGINE (INTEGRATED)
    # -------------------------------------------------------------------------
    def _generate_recommendation(self, disease: str, risk_level: str) -> Dict[str, Any]:

        if disease == "breast_cancer":
            return {
                "priority": "urgent" if risk_level in ["high", "critical"] else "routine",
                "next_steps": [
                    "Consult oncologist",
                    "Perform mammogram",
                    "Consider biopsy if required"
                ],
                "lifestyle": [
                    "Regular screening",
                    "Maintain healthy weight",
                    "Avoid smoking"
                ],
                "timeline": "within 1-2 weeks" if risk_level == "critical" else "within 1 month"
            }

        elif disease == "diabetes":
            return {
                "priority": "urgent" if risk_level in ["high", "critical"] else "routine",
                "next_steps": [
                    "Check HbA1c",
                    "Monitor blood glucose",
                    "Consult endocrinologist"
                ],
                "lifestyle": [
                    "Low sugar diet",
                    "Regular exercise",
                    "Weight management"
                ],
                "timeline": "within 1 week" if risk_level == "critical" else "within 1 month"
            }

        elif disease == "heart":
            return {
                "priority": "urgent" if risk_level in ["high", "critical"] else "routine",
                "next_steps": [
                    "ECG test",
                    "Stress test",
                    "Consult cardiologist"
                ],
                "lifestyle": [
                    "Low cholesterol diet",
                    "Reduce stress",
                    "Regular exercise"
                ],
                "timeline": "immediate check recommended" if risk_level == "critical" else "within 1 month"
            }

        return {
            "priority": "routine",
            "next_steps": ["General health checkup"],
            "lifestyle": ["Maintain healthy lifestyle"],
            "timeline": "as needed"
        }


    def _load_explainability(self, disease: str):

        path = os.path.join(
            self.registry.registry_root,
            disease,
            "shap",
            "explainability_summary.json"
        )

        if not os.path.exists(path):
            return {"top_features": []}

        try:
            with open(path, "r") as f:
                data = json.load(f)

            return self._format_explanation(data)

        except Exception:
            return {"top_features": []}


    def _format_explanation(self, data: dict):

        # 🔥 Correct key from your JSON
        features = data.get("top_global_features", [])

        if not features:
            return {"top_features": []}

        formatted = []

        for item in features[:5]:
            name = item.get("feature")
            value = item.get("mean_abs_shap", 0)

            formatted.append({
                "feature": name,
                "importance": round(float(value), 2)
            })

        return {"top_features": formatted}


    def _generate_human_explanation(self, explanation: dict, disease: str):

        features = explanation.get("top_features", [])

        if not features:
            return "No significant risk factors identified."

        readable = []

        for item in features[:3]:
            name = item["feature"]
            readable.append(self._format_feature_name(name))

        if len(readable) == 1:
            return f"The risk is mainly influenced by {readable[0]}."

        return (
            "The risk appears to be influenced by variations in "
            + ", ".join(readable[:-1])
            + " and " + readable[-1]
            + ", including structural and morphological characteristics."
        )

    def _generate_clinical_explanation(self, explanation, disease):

        features = explanation.get("top_features", [])

        if not features:
            return "No strong clinical indicators detected."

        reasons = []

        for f in features[:3]:
            raw_name = f["feature"]

            # ===============================
            # 🎗️ BREAST CANCER
            # ===============================
            if disease == "breast_cancer":

                if raw_name in ["radius_worst", "area_worst"]:
                    reasons.append("increased tumor size")

                elif raw_name == "perimeter_worst":
                    reasons.append("expanded tumor perimeter")

                elif raw_name in ["concave_points_worst", "concave_points_mean"]:
                    reasons.append("abnormal tumor morphology")

            # ===============================
            # 🩸 DIABETES
            # ===============================
            elif disease == "diabetes":

                if raw_name == "Glucose":
                    reasons.append("elevated blood glucose levels")

                elif raw_name == "BMI":
                    reasons.append("increased body mass index")

            # ===============================
            # ❤️ HEART
            # ===============================
            elif disease == "heart":

                if raw_name == "chol":
                    reasons.append("elevated cholesterol levels")

                elif raw_name == "oldpeak":
                    reasons.append("evidence of cardiac stress")

        reasons = list(set(reasons))

        if reasons:
            return (
                "Findings indicate "
                + ", ".join(reasons[:-1])
                + (" and " + reasons[-1] if len(reasons) > 1 else reasons[0])
                + ", which are associated with elevated disease risk."
            )

        # ✅ FALLBACK (DISEASE-SPECIFIC)
        if disease == "diabetes":
            return "Elevated blood glucose and metabolic indicators suggest increased diabetes risk."

        elif disease == "heart":
            return "Cardiovascular stress and abnormal heart function indicators suggest elevated cardiac risk."

        elif disease == "breast_cancer":
            return "Abnormal tumor characteristics suggest increased malignancy risk."

        return "Clinical indicators suggest potential health risk requiring evaluation."


    # -------------------------------------------------------------------------
    # 🧠 PATIENT-FRIENDLY MESSAGE
    # -------------------------------------------------------------------------

    def _generate_patient_message(self, risk_level: str) -> str:

        if risk_level == "critical":
            return "Your results indicate a high risk. Please consult a doctor immediately."
        elif risk_level == "high":
            return "There are some concerning indicators. A medical checkup is recommended."
        elif risk_level == "moderate":
            return "Some risk factors are present. Monitor your health and consider medical advice."
        else:
            return "Your risk appears low. Maintain a healthy lifestyle."


    # -------------------------------------------------------------------------
    # 📄 MEDICAL REPORT GENERATOR
    # -------------------------------------------------------------------------

    def _generate_medical_report(self, result: Dict[str, Any]) -> Dict[str, Any]:

        risk_level = result.get("risk_level", "")

        report_id = result.get("report_id")

        # -------------------------------------------------
        # 🚨 URGENCY REASON (CLINICAL)
        # -------------------------------------------------
        urgency_reason = (
            "Multiple high-risk clinical indicators detected with strong model confidence"
            if risk_level in ["high", "critical"]
            else "No immediate critical indicators detected"
        )

        # -------------------------------------------------
        # 🎯 NEXT BEST ACTION (CLINICAL DECISION SUPPORT)
        # -------------------------------------------------
        if result["disease"] == "diabetes":
            next_best_action = "Schedule HbA1c and fasting glucose tests"

        elif result["disease"] == "heart":
            next_best_action = "Schedule ECG or cardiac stress test"

        elif result["disease"] == "breast_cancer":
            next_best_action = "Schedule mammogram or diagnostic imaging"

        else:
            next_best_action = "Follow recommended screening and consultation"

        raw_features = result.get("explanation", {}).get("top_features", [])

        top_features = []
        seen = set()

        # ✅ Sort by importance (descending)
        raw_features_sorted = sorted(
            raw_features,
            key=lambda x: x.get("importance", 0),
            reverse=True
        )

        for f in raw_features_sorted:

            raw_name = f["feature"]
            readable_name = self._format_feature_name(raw_name)

            # ✅ REMOVE DUPLICATES
            if readable_name in seen:
                continue
            seen.add(readable_name)

            importance_raw = f.get("importance", 0)

            # ✅ IMPACT LEVEL (use raw value)
            if importance_raw >= 0.04:
                impact = "high"
            elif importance_raw >= 0.03:
                impact = "moderate"
            else:
                impact = "low"

            top_features.append({
                "feature": readable_name,
                "importance": round(importance_raw, 2),  # round only for output
                "impact": impact
            })

            # ✅ LIMIT (UI + readability)
            if len(top_features) == 5:
                break

        return {
            "report_summary": {
                "disease": result["disease"],
                "report_id": report_id,
                "probability": result["probability"]
            },

            # 🧠 HUMAN + CLINICAL EXPLANATION
            "explanation": result.get("human_explanation", ""),

            # 📊 SHAP-BASED FACTORS
            "top_risk_factors": top_features,

            # 🏥 ACTION PLAN
            "recommended_actions": result.get("recommendation", {}).get("next_steps", []),
            "priority": result.get("recommendation", {}).get("priority"),

            # ✅ NEW (VERY IMPORTANT)
            "urgency_reason": urgency_reason,
            "next_best_action": next_best_action,

            # 👤 PATIENT + DOCTOR LAYERS
            "patient_message": result.get("patient_message", ""),
            "doctor_note": "AI-assisted risk assessment. Clinical correlation required."
        }


    # -------------------------------------------------------------------------
    # 🧠 PERSONALIZED RISK INSIGHT ENGINE
    # -------------------------------------------------------------------------

    def _generate_personalized_insights(self, explanation: Dict, disease: str):

        features = explanation.get("top_features", [])

        if not features:
            return {}

        insights = {}
        improvements = []

        # ✅ MAIN DRIVER
        main_driver_raw = features[0]["feature"]
        main_driver = self._format_feature_name(main_driver_raw)

        for f in features[:3]:
            raw_name = f["feature"]

            # ===============================
            # 🎗️ BREAST CANCER
            # ===============================
            if disease == "breast_cancer":

                if any(x in raw_name for x in ["radius", "area"]):
                    insights["Tumor Size"] = "Primary driver of risk"

                elif "perimeter" in raw_name:
                    insights["Tumor Perimeter"] = "Associated with tumor growth progression"

                elif "concave" in raw_name:
                    insights["Tumor Shape"] = "Associated with malignancy risk"

            # ===============================
            # 🩸 DIABETES
            # ===============================
            elif disease == "diabetes":

                if "Glucose" in raw_name:
                    insights["Blood Sugar"] = "Primary driver of metabolic risk"
                    improvements.append("Reduce sugar intake")

                elif "BMI" in raw_name:
                    insights["Body Weight"] = "Contributes to insulin resistance"
                    improvements.append("Increase physical activity")

                elif "Age" in raw_name:
                    insights["Age Factor"] = "Age-related risk influence"

            # ===============================
            # ❤️ HEART
            # ===============================
            elif disease == "heart":

                if "chol" in raw_name:
                    insights["Cholesterol"] = "Primary cardiovascular risk factor"
                    improvements.append("Reduce fatty foods")

                elif "oldpeak" in raw_name:
                    insights["Cardiac Stress"] = "Indicates exercise-induced stress"
                    improvements.append("Manage stress levels")

                elif "thalach" in raw_name:
                    insights["Heart Rate"] = "Linked to cardiovascular performance"

        # ✅ ENSURE INSIGHTS NOT EMPTY (VERY IMPORTANT)

        if not insights:
            insights["General Risk"] = "Multiple contributing factors detected"

        # ✅ ENSURE IMPROVEMENTS EXIST
        if not improvements:
            improvements = [
                "Regular screening adherence",
                "Early detection monitoring"
            ]

        return {
            "main_risk_driver": main_driver,
            "risk_contribution": insights,
            "what_to_improve": list(set(improvements))
        }


    def _get_severity(self, risk_level):
        if risk_level == "critical":
            return "life_threatening"
        elif risk_level == "high":
            return "serious"
        elif risk_level == "moderate":
            return "moderate"
        return "low"


    def _predict_risk_timeline(self, prob: float):

        # simple progression model
        return {
            "current": round(prob, 3),
            "1_year": round(min(prob + 0.03, 0.97), 3),
            "3_year": round(min(prob + 0.05, 0.98), 3),
            "5_year": round(min(prob + 0.07, 0.99), 3)
        }


    def _risk_breakdown(self, disease: str, features: Dict):

        if disease == "diabetes":
            return {
                "metabolic_risk": "high" if features.get("Glucose", 0) > 140 else "moderate",
                "obesity_risk": "high" if features.get("BMI", 0) > 30 else "moderate"
            }

        elif disease == "heart":
            return {
                "cardiovascular_risk": "high" if features.get("chol", 0) > 240 else "moderate",
                "stress_risk": "high" if features.get("oldpeak", 0) > 2 else "moderate"
            }

        elif disease == "breast_cancer":
            return {
                "tumor_risk": "high"
            }

        return {}


    def _calculate_preventable_risk(self, disease: str, features: Dict):

        if disease == "diabetes":
            return 0.7
        elif disease == "heart":
            return 0.65
        elif disease == "breast_cancer":
            return 0.3

        return 0.5


    def _confidence_reason(self, prob: float):

        if prob > 0.85:
            return "Very strong model confidence due to consistent high-risk indicators"

        elif prob > 0.6:
            return "Moderate confidence with several contributing risk factors"

        return "Low confidence due to weak or mixed signals"


    def _doctor_priority(self, prob: float):

        if prob > 0.85:
            return "emergency"
        elif prob > 0.6:
            return "urgent"
        return "routine"


    def _format_feature_name(self, feature: str):

        # Remove one-hot encoding suffix (cp_0 → cp)
        if "_" in feature and feature.split("_")[-1].isdigit():
            feature = feature.rsplit("_", 1)[0]

        # Direct mapping
        if feature in FEATURE_MEANING:
            return FEATURE_MEANING[feature]

        # Handle engineered features
        if "_" in feature:
            parts = feature.split("_")

            readable_parts = []
            for p in parts:
                if p.lower() in ["high", "low", "ratio"]:
                    continue  # remove meaningless prefixes
                readable_parts.append(FEATURE_MEANING.get(p, p))

            return " & ".join(readable_parts)

        return feature.replace("_", " ").title()


# =============================================================================
# FACTORY
# =============================================================================

def create_predictor(registry_root: str) -> RiskPredictor:
    registry = ModelRegistry(registry_root)
    registry.discover(strict=True)
    return RiskPredictor(registry)
