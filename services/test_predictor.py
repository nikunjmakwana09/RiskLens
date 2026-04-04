try:
    from services.predictor import create_predictor
except ModuleNotFoundError:
    from predictor import create_predictor

import json

# =====================================================
# 🚀 INITIALIZE SYSTEM
# =====================================================

predictor = create_predictor("model_registry")


# =====================================================
# 📊 TEST CONFIGURATION
# =====================================================

TEST_SUITE = {

    # =====================================================
    # 🩸 DIABETES
    # =====================================================
    "diabetes": {
        "LOW": {
            "Pregnancies": 0,
            "Glucose": 85,
            "BloodPressure": 70,
            "SkinThickness": 20,
            "Insulin": 80,
            "BMI": 22,
            "DiabetesPedigreeFunction": 0.2,
            "Age": 25
        },
        "EDGE": {
            "Pregnancies": 2,
            "Glucose": 120,
            "BloodPressure": 80,
            "SkinThickness": 25,
            "Insulin": 100,
            "BMI": 28,
            "DiabetesPedigreeFunction": 0.5,
            "Age": 40
        },
        "HIGH": {
            "Pregnancies": 5,
            "Glucose": 180,
            "BloodPressure": 95,
            "SkinThickness": 35,
            "Insulin": 200,
            "BMI": 35,
            "DiabetesPedigreeFunction": 1.2,
            "Age": 55
        },
        "EXTREME": {
            "Pregnancies": 8,
            "Glucose": 250,
            "BloodPressure": 110,
            "SkinThickness": 45,
            "Insulin": 300,
            "BMI": 42,
            "DiabetesPedigreeFunction": 2.0,
            "Age": 70
        }
    },

    # =====================================================
    # ❤️ HEART
    # =====================================================
    "heart": {
        "LOW": {
            "age": 30,
            "sex": 0,
            "cp": 0,
            "trestbps": 110,
            "chol": 180,
            "fbs": 0,
            "restecg": 0,
            "thalach": 170,
            "exang": 0,
            "oldpeak": 0.5,
            "slope": 2,
            "ca": 0,
            "thal": 1
        },
        "EDGE": {
            "age": 45,
            "sex": 1,
            "cp": 1,
            "trestbps": 130,
            "chol": 220,
            "fbs": 0,
            "restecg": 1,
            "thalach": 150,
            "exang": 1,
            "oldpeak": 1.5,
            "slope": 1,
            "ca": 1,
            "thal": 2
        },
        "HIGH": {
            "age": 60,
            "sex": 1,
            "cp": 3,
            "trestbps": 160,
            "chol": 280,
            "fbs": 1,
            "restecg": 2,
            "thalach": 120,
            "exang": 1,
            "oldpeak": 3.0,
            "slope": 0,
            "ca": 3,
            "thal": 3
        },
        "EXTREME": {
            "age": 75,
            "sex": 1,
            "cp": 3,
            "trestbps": 190,
            "chol": 350,
            "fbs": 1,
            "restecg": 2,
            "thalach": 90,
            "exang": 1,
            "oldpeak": 5.0,
            "slope": 0,
            "ca": 4,
            "thal": 3
        }
    },

    # =====================================================
    # 🎗️ BREAST CANCER
    # =====================================================

    "breast_cancer": {
        "LOW": {
            "radius_mean": 12.0, "texture_mean": 14.0, "perimeter_mean": 78.0, "area_mean": 450.0,
            "smoothness_mean": 0.08, "compactness_mean": 0.05, "concavity_mean": 0.02,
            "concave_points_mean": 0.01, "symmetry_mean": 0.18, "fractal_dimension_mean": 0.06,

            "radius_se": 0.3, "texture_se": 0.8, "perimeter_se": 2.0, "area_se": 20.0,
            "smoothness_se": 0.005, "compactness_se": 0.01, "concavity_se": 0.01,
            "concave_points_se": 0.005, "symmetry_se": 0.02, "fractal_dimension_se": 0.002,

            "radius_worst": 13.0, "texture_worst": 16.0, "perimeter_worst": 85.0, "area_worst": 500.0,
            "smoothness_worst": 0.09, "compactness_worst": 0.07, "concavity_worst": 0.03,
            "concave_points_worst": 0.02, "symmetry_worst": 0.2, "fractal_dimension_worst": 0.07
        },

        "EDGE": {
            "radius_mean": 15.0, "texture_mean": 18.0, "perimeter_mean": 95.0, "area_mean": 700.0,
            "smoothness_mean": 0.095, "compactness_mean": 0.12, "concavity_mean": 0.1,
            "concave_points_mean": 0.05, "symmetry_mean": 0.22, "fractal_dimension_mean": 0.065,

            "radius_se": 0.6, "texture_se": 1.2, "perimeter_se": 4.0, "area_se": 60.0,
            "smoothness_se": 0.01, "compactness_se": 0.03, "concavity_se": 0.03,
            "concave_points_se": 0.015, "symmetry_se": 0.03, "fractal_dimension_se": 0.004,

            "radius_worst": 18.0, "texture_worst": 22.0, "perimeter_worst": 110.0, "area_worst": 900.0,
            "smoothness_worst": 0.12, "compactness_worst": 0.18, "concavity_worst": 0.2,
            "concave_points_worst": 0.1, "symmetry_worst": 0.28, "fractal_dimension_worst": 0.08
        },

        "HIGH": {
            "radius_mean": 20.0, "texture_mean": 25.0, "perimeter_mean": 130.0, "area_mean": 1200.0,
            "smoothness_mean": 0.12, "compactness_mean": 0.3, "concavity_mean": 0.4,
            "concave_points_mean": 0.2, "symmetry_mean": 0.3, "fractal_dimension_mean": 0.09,

            "radius_se": 1.5, "texture_se": 2.0, "perimeter_se": 10.0, "area_se": 200.0,
            "smoothness_se": 0.02, "compactness_se": 0.1, "concavity_se": 0.1,
            "concave_points_se": 0.05, "symmetry_se": 0.05, "fractal_dimension_se": 0.01,

            "radius_worst": 25.0, "texture_worst": 30.0, "perimeter_worst": 170.0, "area_worst": 2000.0,
            "smoothness_worst": 0.15, "compactness_worst": 0.5, "concavity_worst": 0.6,
            "concave_points_worst": 0.3, "symmetry_worst": 0.4, "fractal_dimension_worst": 0.1
        },

        "EXTREME": {
            "radius_mean": 28.0, "texture_mean": 35.0, "perimeter_mean": 180.0, "area_mean": 2200.0,
            "smoothness_mean": 0.18, "compactness_mean": 0.6, "concavity_mean": 0.7,
            "concave_points_mean": 0.35, "symmetry_mean": 0.45, "fractal_dimension_mean": 0.12,

            "radius_se": 2.5, "texture_se": 3.5, "perimeter_se": 18.0, "area_se": 400.0,
            "smoothness_se": 0.03, "compactness_se": 0.2, "concavity_se": 0.2,
            "concave_points_se": 0.1, "symmetry_se": 0.08, "fractal_dimension_se": 0.02,

            "radius_worst": 32.0, "texture_worst": 40.0, "perimeter_worst": 210.0, "area_worst": 3000.0,
            "smoothness_worst": 0.22, "compactness_worst": 0.8, "concavity_worst": 0.9,
            "concave_points_worst": 0.45, "symmetry_worst": 0.6, "fractal_dimension_worst": 0.15
        }
    }
}

# =====================================================
# 🧪 EXECUTION ENGINE
# =====================================================

def run_tests():
    print("\n" + "=" * 60)
    print("🚀 AI HEALTH RISK TEST SUITE")
    print("=" * 60)

    for disease, scenarios in TEST_SUITE.items():
        print(f"\n\n===== TESTING: {disease.upper()} =====")

        for scenario, payload in scenarios.items():

            try:
                result = predictor.predict(disease, payload)
            except Exception as e:
                result = {"status": "error", "message": str(e)}

            print(f"\n--- {scenario} ---")
            print(json.dumps(result, indent=2))

    print("\n" + "=" * 60)
    print("✅ TESTING COMPLETED")
    print("=" * 60)


# =====================================================
# ▶️ RUN
# =====================================================

if __name__ == "__main__":
    run_tests()
