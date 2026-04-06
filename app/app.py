import streamlit as st
import sys
import os

# -----------------------------
# LOAD PREDICTOR
# -----------------------------
sys.path.append(os.path.abspath("."))

from services.predictor import create_predictor


@st.cache_resource
def load_model():
    return create_predictor("model_registry")


predictor = load_model()


st.markdown("""
<style>

/* Base Glass */
.glass {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 18px;
    backdrop-filter: blur(10px);
    transition: all 0.25s ease;
    animation: fadeln 0.5s ease-in-out;
}

/* Hover */
.glass:hover {
    transform: translateY(-3px) scale(1.01);
    box-shadow: 0 10px 30px rgba(59,130,246,0.15);
}

/* HERO CARD (IMPORTANT) */
.glass-hero {
    background: rgba(59,130,246,0.08);
    border: 1px solid rgba(59,130,246,0.4);
    border-radius: 16px;
    padding: 22px;
    backdrop-filter: blur(12px);
}

/* SECTION TITLE */
.section-title {
    font-size: 22px;
    font-weight: 600;
    margin-bottom: 16px;
}

/* SUB TITLE */
.sub-title {
    font-size: 15px;
    color: #9ca3af;
    margin-bottom: 8px;
}

/* SPACING SYSTEM */
.gap-lg { margin-top: 28px; }
.gap-md { margin-top: 18px; }

/* GRID GAP FIX */
div[data-testid="column"] {
    padding-right: 8px !important;
}

.gradient-text {
    background: linear-gradient(90deg, #3b82f6, #06b6d4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 600;
}

button[data-baseweb="tab"] {
    font-weight: 500;
}

button[data-baseweb="tab"][aria-selected="true"] {
    border-bottom: 2px solid #3b82f6;
}

.fade-in {
    animation: fadeIn 0.6s ease-in-out;
}

@keyframes fadeIn {
    from {opacity: 0; transform: translateY(8px);}
    to {opacity: 1; transform: translateY(0);}
}
</style>
""", unsafe_allow_html=True)


def input_card(title, subtitle, icon=""):
    return f"""
    <div class="glass" style="
        padding:16px;
        border-radius:14px;
        margin-bottom:10px;
        background: rgba(255,255,255,0.03);
        border:1px solid rgba(255,255,255,0.08);
    ">
        <div style="font-size:13px;color:#9ca3af;">
            {icon} {title}
        </div>
        <div style="font-size:12px;color:#6b7280;">
            {subtitle}
        </div>
    </div>
    """


# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="RiskLens AI", layout="wide")

st.title("🧠 RiskLens AI")
st.caption("Clinical-Grade Disease Risk Prediction System")

# -----------------------------
# TOP CONTROLS
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    disease = st.selectbox(
        "Select Disease",
        ["diabetes", "heart", "breast_cancer"]
    )

with col2:
    mode = st.selectbox(
        "Mode",
        ["Smart", "Expert"]
    )


# =========================================================
# 🩸 DIABETES
# =========================================================
def diabetes_form(mode):

    if mode == "Smart":
        st.subheader("🩸 Diabetes - Smart Mode")

        col1, col2 = st.columns(2)

        glucose = col1.number_input("Glucose", 50, 300, 120)
        bmi = col2.number_input("BMI", 10.0, 50.0, 25.0)
        age = col1.number_input("Age", 1, 100, 30)
        insulin = col2.number_input("Insulin", 0, 400, 80)
        bp = col1.number_input("Blood Pressure", 50, 200, 80)

        return {
            "Pregnancies": 1,
            "Glucose": glucose,
            "BloodPressure": bp,
            "SkinThickness": 25,
            "Insulin": insulin,
            "BMI": bmi,
            "DiabetesPedigreeFunction": 0.5,
            "Age": age
        }

    else:

        payload = {}

        with st.expander("🧠 Core Clinical Features", expanded=True):
            col1, col2 = st.columns(2)
            payload["Glucose"] = col1.number_input("Glucose", value=120)
            payload["BMI"] = col2.number_input("BMI", value=25.0)
            payload["Age"] = col1.number_input("Age", value=30)
            payload["Insulin"] = col2.number_input("Insulin", value=80)

        with st.expander("📊 Secondary Factors"):
            col1, col2 = st.columns(2)
            payload["BloodPressure"] = col1.number_input("Blood Pressure", value=80)
            payload["DiabetesPedigreeFunction"] = col2.number_input("Genetic Risk", value=0.5)

        with st.expander("⚙️ Additional Inputs"):
            col1, col2 = st.columns(2)
            payload["Pregnancies"] = col1.number_input("Pregnancies", value=1)
            payload["SkinThickness"] = col2.number_input("Skin Thickness", value=25)

        return payload


# =========================================================
# ❤️ HEART
# =========================================================
def heart_form(mode):

    if mode == "Smart":
        st.subheader("❤️ Heart - Smart Mode")

        col1, col2 = st.columns(2)

        age = col1.number_input("Age", 20, 100, 45)
        cp = col2.selectbox("Chest Pain Type", [0,1,2,3])
        chol = col1.number_input("Cholesterol", 100, 400, 200)
        thalach = col2.number_input("Heart Rate", 60, 200, 150)
        oldpeak = col1.number_input("Cardiac Stress", 0.0, 6.0, 1.0)
        exang = col2.selectbox("Exercise Angina", [0,1])

        return {
            "age": age,
            "sex": 1,
            "cp": cp,
            "trestbps": 120,
            "chol": chol,
            "fbs": 0,
            "restecg": 1,
            "thalach": thalach,
            "exang": exang,
            "oldpeak": oldpeak,
            "slope": 1,
            "ca": 0,
            "thal": 2
        }

    else:

        payload = {}

        with st.expander("🧠 Core Cardiac Indicators", expanded=True):
            col1, col2 = st.columns(2)
            payload["age"] = col1.number_input("Age", value=45)
            payload["cp"] = col2.selectbox("Chest Pain Type", [0,1,2,3])
            payload["chol"] = col1.number_input("Cholesterol", value=200)
            payload["thalach"] = col2.number_input("Heart Rate", value=150)
            payload["oldpeak"] = col1.number_input("Cardiac Stress", value=1.0)

        with st.expander("📊 Secondary Indicators"):
            col1, col2 = st.columns(2)
            payload["exang"] = col1.selectbox("Exercise Angina", [0,1])
            payload["trestbps"] = col2.number_input("Blood Pressure", value=120)
            payload["ca"] = col1.selectbox("Blocked Vessels", [0,1,2,3])
            payload["thal"] = col2.selectbox("Thalassemia", [1,2,3])

        with st.expander("⚙️ Additional Inputs"):
            col1, col2 = st.columns(2)
            payload["sex"] = col1.selectbox("Sex", [0,1])
            payload["fbs"] = col2.selectbox("FBS", [0,1])
            payload["restecg"] = col1.selectbox("ECG", [0,1,2])
            payload["slope"] = col2.selectbox("Slope", [0,1,2])

        return payload


# =========================================================
# 🎗️ BREAST CANCER
# =========================================================
def breast_form(mode):

    if mode == "Smart":
        st.subheader("🎗️ Breast Cancer - Smart Mode")

        col1, col2 = st.columns(2)

        radius = col1.slider("Tumor Radius", 10.0, 30.0, 15.0)
        area = col2.slider("Tumor Area", 300.0, 2000.0, 700.0)
        perimeter = col1.slider("Perimeter", 70.0, 200.0, 100.0)
        concavity = col2.slider("Concavity", 0.0, 1.0, 0.2)

        return {
            "radius_mean": radius,
            "texture_mean": 20,
            "perimeter_mean": perimeter,
            "area_mean": area,
            "smoothness_mean": 0.1,
            "compactness_mean": 0.2,
            "concavity_mean": concavity,
            "concave_points_mean": 0.1,
            "symmetry_mean": 0.2,
            "fractal_dimension_mean": 0.07,

            "radius_se": 1, "texture_se": 1, "perimeter_se": 1,
            "area_se": 1, "smoothness_se": 0.01,
            "compactness_se": 0.01, "concavity_se": 0.01,
            "concave_points_se": 0.01, "symmetry_se": 0.02,
            "fractal_dimension_se": 0.01,

            "radius_worst": radius + 5,
            "texture_worst": 25,
            "perimeter_worst": perimeter + 20,
            "area_worst": area + 200,
            "smoothness_worst": 0.12,
            "compactness_worst": 0.3,
            "concavity_worst": 0.3,
            "concave_points_worst": 0.2,
            "symmetry_worst": 0.3,
            "fractal_dimension_worst": 0.08
        }

    else:

        payload = {}

        with st.expander("📊 Mean Features", expanded=True):
            cols = st.columns(3)
            fields = [
                "radius_mean","texture_mean","perimeter_mean","area_mean",
                "smoothness_mean","compactness_mean","concavity_mean",
                "concave_points_mean","symmetry_mean","fractal_dimension_mean"
            ]
            for i, f in enumerate(fields):
                payload[f] = cols[i % 3].number_input(f, value=1.0)

        with st.expander("📉 Standard Error Features"):
            cols = st.columns(3)
            fields = [
                "radius_se","texture_se","perimeter_se","area_se",
                "smoothness_se","compactness_se","concavity_se",
                "concave_points_se","symmetry_se","fractal_dimension_se"
            ]
            for i, f in enumerate(fields):
                payload[f] = cols[i % 3].number_input(f, value=0.5)

        with st.expander("⚠️ Worst Features"):
            cols = st.columns(3)
            fields = [
                "radius_worst","texture_worst","perimeter_worst","area_worst",
                "smoothness_worst","compactness_worst","concavity_worst",
                "concave_points_worst","symmetry_worst","fractal_dimension_worst"
            ]
            for i, f in enumerate(fields):
                payload[f] = cols[i % 3].number_input(f, value=2.0)

        return payload


# -----------------------------
# LOAD FORM
# -----------------------------
if disease == "diabetes":
    payload = diabetes_form(mode)
elif disease == "heart":
    payload = heart_form(mode)
else:
    payload = breast_form(mode)

# -----------------------------
# PREDICT
# -----------------------------
if st.button("🚀 Analyze Risk"):
    with st.spinner("🧠 Analyzing clinical data..."):
        result = predictor.predict(disease, payload)

    if result["status"] == "success":

        pred = result["prediction"]
        conf = result["confidence"]
        insights = result["insights"]
        report = result["medical_report"]
        adv = result["advanced_analysis"]
        sys = result["system"]
        meta = result["meta"]
        risk = pred["risk_level"]
        prob = pred["probability"]
        severity = pred["severity"]
        confidence = conf["level"]
        context = insights["risk_context"]
        # =========================================================
        # 🎯 HERO SECTION (PREMIUM)
        # =========================================================
        st.markdown("## 🧠 AI Clinical Risk Assessment")

        # Dynamic color based on risk
        color = {
            "very_low": "#22c55e",
            "low": "#22c55e",
            "moderate": "#f59e0b",
            "high": "#ef4444",
            "critical": "#dc2626"
        }.get(risk.lower(), "#3b82f6")

        st.markdown(f"""
        <div class="glass-hero" style="
        padding:28px;
        border-radius:18px;
        background:linear-gradient(135deg, rgba(59,130,246,0.25), rgba(6,182,212,0.08));
        border:1px solid rgba(59,130,246,0.5);
        box-shadow: 0 20px 60px rgba(59,130,246,0.2);
        position:relative;
        overflow:hidden;
        ">

        <div style="font-size:13px;color:#9ca3af;margin-bottom:8px;">
        AI Clinical Decision Engine
        </div>

        <h1 style="color:{color};margin-bottom:14px;font-size:32px;">
        {risk.upper()} RISK
        </h1>

        <div style="display:flex;gap:30px;font-size:14px;">
        <div>📊 Probability <b>{round(prob*100,1)}%</b></div>
        <div>⚠️ Severity <b>{severity}</b></div>
        <div>🎯 Confidence <b>{confidence}</b></div>
        </div>

        </div>
        """, unsafe_allow_html=True)

        st.markdown("")

        # =========================================================
        # 📊 PREMIUM METRIC CARDS
        # =========================================================
        def premium_card(title, value, color="#3b82f6", icon="", big=False):
            return f"""
            <div class="glass" style="
                padding:{'24px' if big else '18px'};
                border-radius:16px;
                border-left:5px solid {color};
                height:100%;
                background: rgba(255,255,255,0.04);
            ">
                <div style="font-size:13px;color:#9ca3af;margin-bottom:8px;">
                    {icon} {title}
                </div>
                <div style="font-size:{'22px' if big else '18px'};font-weight:600;">
                    {value}
                </div>
            </div>
            """

        st.markdown(
            premium_card(
                "Clinical Summary",
                conf["reason"],
                "#6366f1",
                "🧠"
            ),
            unsafe_allow_html=True
        )

        st.markdown('<div style="margin-top:16px"></div>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)

        col1.markdown(
            premium_card("Doctor Priority", adv["doctor_priority"], "#3b82f6", "🩺"),
            unsafe_allow_html=True
        )

        col2.markdown(
            premium_card("Care Level", result["decision_support"]["care_level"], "#10b981", "💚"),
            unsafe_allow_html=True
        )

        col3.markdown(
            premium_card("Preventable Risk", adv["preventable_risk"], "#f59e0b", "⚡"),
            unsafe_allow_html=True
        )

        st.divider()

        # =========================================================
        # 📑 TABS
        # =========================================================

        tabs = st.tabs([
            "📊 Overview",
            "🧠 Insights",
            "📄 Report",
            "🚀 Analytics",
            "⚙️ System"
        ])

        # =========================================================
        # 📊 OVERVIEW
        # =========================================================
        with tabs[0]:

            st.markdown('<h2 class="gradient-text">AI Risk Overview</h2>', unsafe_allow_html=True)
            # HERO CARD
            st.markdown(f"""
            <div class="glass-hero" >
                <b>Risk Level:</b> {pred["risk_level"].upper()}<br><br>
                <b>Probability:</b> {round(pred["probability"]*100,1)}%<br><br>
                <b>Clinical Insight:</b><br>{conf["reason"]}
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<div class="gap-lg"></div>', unsafe_allow_html=True)

            col1, col2, col3, col4 = st.columns(4)

            col1.markdown(premium_card("Severity", pred["severity"], "#ef4444", "⚠️"), unsafe_allow_html=True)
            col2.markdown(premium_card("Confidence", conf["level"], "#3b82f6", "🎯"), unsafe_allow_html=True)
            col3.markdown(premium_card("Care Level", result["decision_support"]["care_level"], "#10b981", "💚"), unsafe_allow_html=True)
            col4.markdown(premium_card("Doctor Priority", adv["doctor_priority"], "#f59e0b", "🩺"), unsafe_allow_html=True)

            st.markdown('<div class="gap-lg"></div>', unsafe_allow_html=True)

            st.markdown('<div class="sub-title">Clinical Context</div>', unsafe_allow_html=True)

            col1, col2 = st.columns(2)

            col1.markdown(f"""
            <div class="glass" style="border-left:4px solid #22c55e">
            {context["relative_risk"]}
            </div>
            """, unsafe_allow_html=True)

            col2.markdown(f"""
            <div class="glass" style="border-left:4px solid #3b82f6">
            {context["clinical_interpretation"]}
            </div>
            """, unsafe_allow_html=True)


        # =========================================================
        # 🧠 INSIGHTS
        # =========================================================
        with tabs[1]:

            st.markdown('<h2 class="gradient-text">AI Insights</h2>', unsafe_allow_html=True)
            # HERO DRIVER
            st.markdown(f"""
            <div class="glass-hero" >
                🎯 <b>Main Risk Driver</b><br><br>
                {insights["personalized"]["main_risk_driver"]}
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<div class="gap-lg"></div>', unsafe_allow_html=True)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### 📊 Contributions")
                for k, v in insights["personalized"]["risk_contribution"].items():
                    st.markdown(f'<div class="glass">{k}<br>{v}</div>', unsafe_allow_html=True)

            with col2:
                st.markdown("### 💡 Improvements")
                for item in insights["personalized"]["what_to_improve"]:
                    st.markdown(f'<div class="glass">✔ {item}</div>', unsafe_allow_html=True)

            st.markdown('<div class="gap-lg"></div>', unsafe_allow_html=True)

            st.markdown(f'<div class="glass">🏥 {result["insights"]["clinical_explanation"]}</div>', unsafe_allow_html=True)


        # =========================================================
        # 📄 REPORT
        # =========================================================
        with tabs[2]:

            st.markdown('<h2 class="gradient-text">Medical Report</h2>', unsafe_allow_html=True)
            st.markdown('<div class="gap-lg"></div>', unsafe_allow_html=True)

            report = result["medical_report"]

            st.markdown(f'<div class="glass">{report["explanation"]}</div>', unsafe_allow_html=True)

            st.markdown('<div class="gap-lg"></div>', unsafe_allow_html=True)

            st.markdown("### ⚠️ Risk Factors")
            for f in report["top_risk_factors"]:
                st.markdown(f"""
                <div class="glass" style="display:flex;justify-content:space-between">
                    <span>{f["feature"]}</span>
                    <span style="color:#f59e0b">{f["impact"]}</span>
                </div>
                """, unsafe_allow_html=True)

            st.markdown('<div class="gap-lg"></div>', unsafe_allow_html=True)

            st.markdown("### 🏥 Recommended Actions")
            for a in report["recommended_actions"]:
                st.markdown(f'<div class="glass">✔ {a}</div>', unsafe_allow_html=True)

            st.markdown('<div class="gap-lg"></div>', unsafe_allow_html=True)

            st.markdown(f'<div class="glass">🚨 {report["urgency_reason"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="glass">🎯 {report["next_best_action"]}</div>', unsafe_allow_html=True)

            # ADD THIS (missing earlier)
            st.markdown(f'<div class="glass">👤 {report["patient_message"]}</div>', unsafe_allow_html=True)


        # =========================================================
        # 🚀 ANALYTICS
        # =========================================================
        with tabs[3]:

            st.markdown('<h2 class="gradient-text">Predictive Analytics</h2>', unsafe_allow_html=True)
            st.markdown('<div class="gap-lg"></div>', unsafe_allow_html=True)

            adv = result["advanced_analysis"]
            timeline = adv["risk_timeline"]

            # Timeline (ONLY ONCE)
            col1, col2, col3, col4 = st.columns(4)

            labels = ["Now", "1Y", "3Y", "5Y"]
            values = [
                timeline["current"],
                timeline["1_year"],
                timeline["3_year"],
                timeline["5_year"]
            ]

            for col, label, value in zip([col1, col2, col3, col4], labels, values):
                col.markdown(
                    premium_card(label, value, "#6366f1", "📊"),
                    unsafe_allow_html=True
                )

            st.markdown('<div class="gap-lg"></div>', unsafe_allow_html=True)

            # ADD breakdown (missing)
            st.markdown("### 📊 Risk Breakdown")
            for k, v in adv["risk_breakdown"].items():
                st.markdown(f"• {k}: {v}")

            st.markdown('<div class="gap-lg"></div>', unsafe_allow_html=True)

            st.markdown(f'<div class="glass">Preventable Risk: {adv["preventable_risk"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="glass">Doctor Priority: {adv["doctor_priority"]}</div>', unsafe_allow_html=True)


        # =========================================================
        # ⚙️ SYSTEM
        # =========================================================
        with tabs[4]:

            st.markdown('<h2 class="gradient-text">System Intelligence</h2>', unsafe_allow_html=True)
            st.markdown('<div class="gap-lg"></div>', unsafe_allow_html=True)

            sys = result["system"]
            meta = result["meta"]

            col1, col2 = st.columns(2)

            col1.markdown(
                premium_card("Model", sys["model"]["name"], "#3b82f6", "🤖"),
                unsafe_allow_html=True
            )

            col2.markdown(
                premium_card("Version", sys["model"]["version"], "#10b981", "📦"),
                unsafe_allow_html=True
            )

            st.markdown('<div class="gap-lg"></div>', unsafe_allow_html=True)

            st.markdown(f'<div class="glass">Latency: {meta["latency_ms"]} ms</div>', unsafe_allow_html=True)

            st.markdown('<div class="gap-lg"></div>', unsafe_allow_html=True)

            st.markdown(f'<div class="glass">Features: {sys["features"]}</div>', unsafe_allow_html=True)

            st.markdown('<div class="gap-lg"></div>', unsafe_allow_html=True)

            st.markdown(f'<div class="glass">⚠️ {result["disclaimer"]}</div>', unsafe_allow_html=True)

            # ADD RAW OUTPUT (VERY IMPORTANT)
            st.markdown('<div class="gap-lg"></div>', unsafe_allow_html=True)

            with st.expander("🔍 Full Model Output"):
                st.json(result)
