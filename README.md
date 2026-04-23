# 🧠 RiskLens AI

### Clinical-Grade Multi-Disease Risk Prediction & Decision Intelligence Platform

RiskLens AI is a **production-grade, explainable, multi-disease AI system** designed for **early disease risk prediction and clinical decision support**.

The platform transforms machine learning models into **real-world healthcare intelligence**, delivering **real-time, interpretable, and actionable insights** for proactive and preventive care.

---

# 🌐 Live System

### 🔗 Web Application (Streamlit Dashboard)

```
https://risklensai.streamlit.app
```

### 🔗 Prediction API (FastAPI)

```
https://risklens-8axc.onrender.com
```

### 📄 API Documentation

```
https://risklens-8axc.onrender.com/docs
```

---

# 🏥 Project Overview

Modern healthcare systems face challenges in:

* Late disease detection
* Lack of explainable AI
* Absence of real-time monitoring

RiskLens AI addresses these gaps by building a **clinical-grade AI platform** that enables:

✔ Early risk detection
✔ Explainable predictions
✔ Personalized medical insights
✔ Scalable healthcare deployment

---

# ✨ Key Features

## 🧠 Multi-Disease AI Prediction

Supports multiple diseases:

* Diabetes
* Heart Disease
* Breast Cancer

Provides:

* Risk probability
* Severity classification
* Confidence scoring

---

## 🔍 Explainable AI (SHAP-Based)

* Identifies **top risk-driving factors**
* Provides **human + clinical explanations**
* Enhances **trust and transparency**

---

## 📊 Advanced Clinical Intelligence

* Risk timeline (Now, 1Y, 3Y, 5Y)
* Preventable risk estimation
* Risk breakdown (metabolic, cardiovascular, etc.)
* Doctor priority & care level

---

## 📄 Automated Medical Report Generation

Generates structured reports including:

* Risk explanation
* Top contributing factors
* Recommended actions
* Clinical urgency
* Patient-friendly message

---

## 🎯 Personalized Insights Engine

* Main risk driver detection
* Feature contribution analysis
* Lifestyle improvement suggestions

---

## ⚙️ Dual Prediction System (High Reliability)

* Cloud API inference
* Local fallback model

Ensures:

* High availability
* Zero downtime prediction

---

## 🎨 Interactive AI Dashboard

Modern UI built with Streamlit:

* Smart & Expert modes
* Clinical visualization
* Real-time analytics
* Multi-tab insights system

---

# 🏗️ System Architecture

```
User
 │
 ▼
Streamlit AI Dashboard
 │
 ▼
Prediction Request
 │
 ├── FastAPI Cloud API
 │       │
 │       ▼
 │   RiskLens Predictor Engine
 │       │
 │       ├── Model Registry
 │       ├── Feature Builder
 │       ├── ML Model (Calibrated)
 │       ├── Explainability (SHAP)
 │       └── Clinical Intelligence Layer
 │
 └── Local Fallback Predictor
```

---

# ⚙️ Core Components

## 🔹 Model Registry

* Dynamic model loading
* Multi-disease management
* Metadata & version control

## 🔹 Feature Builder

* Input validation
* Feature engineering
* Schema alignment

## 🔹 Prediction Engine

* Probabilistic ML inference
* Threshold-based classification
* Rule-based clinical adjustments

## 🔹 Explainability Layer

* SHAP-based interpretation
* Feature importance extraction

## 🔹 Clinical Intelligence Layer

* Risk scoring
* Severity classification
* Medical recommendations

---

# 🤖 Machine Learning Pipeline

1. Data ingestion
2. Data preprocessing
3. Feature engineering
4. Model training & tuning
5. Model evaluation
6. Explainability integration
7. Deployment via API
8. Monitoring & validation

---

# 📊 Model Performance

| Disease       | F1 Score | ROC-AUC |
| ------------- | -------- | ------- |
| Breast Cancer | 0.97     | 0.99    |
| Heart Disease | 0.84     | 0.88    |
| Diabetes      | 0.68     | 0.83    |

✔ Threshold-tuned predictions
✔ Calibrated probability outputs
✔ Stable real-time inference (~25–30 ms)

---

# 📡 API Endpoints

## 🔹 Predict Risk

```
POST /predict
```

### Example Request

```json
{
  "disease": "diabetes",
  "payload": {
    "Glucose": 120,
    "BMI": 25,
    "Age": 30,
    "Insulin": 80
  }
}
```

---

## 🔹 Health Check

```
GET /health
```

Returns:

* API status
* model info
* system health

---

# 🧪 Example Output

```json
{
  "risk_level": "very_low",
  "probability": 0.034,
  "severity": "low",
  "confidence": "low",
  "doctor_priority": "routine"
}
```

---

# 💻 Technology Stack

## 🧠 Machine Learning

* Scikit-learn
* Logistic Regression (Calibrated)
* SHAP

## ⚙️ Backend

* FastAPI
* Python

## 🎨 Frontend

* Streamlit
* Plotly
* Custom UI Styling

## ☁️ Deployment

* Render (API)
* Streamlit Cloud

---

# 📂 Project Structure

```
RiskLens/
│
├── app/
│   ├── app.py
│
├── data/
│   ├── processed/
│       ├── breast_cancer_clean.csv
│       ├── diabetes_clean.csv
│       ├── heart_clean.csv
│   ├── raw/
│       ├── breast_cancer.csv
│       ├── diabetes.csv
│       ├── heart.csv
│
├── diseases/
│   ├── breast_cancer/
│       ├── data_cleaning.py
│       ├── eda.py
│       ├── evaluate.py
│       ├── feature_importance.py
│       ├── shap_explainer.py
│       ├── train.py
│   ├── diabetes/
│       ├── data_cleaning.py
│       ├── eda.py
│       ├── evaluate.py
│       ├── feature_importance.py
│       ├── shap_explainer.py
│       ├── train.py
│   ├── heart/
│       ├── data_cleaning.py
│       ├── eda.py
│       ├── evaluate.py
│       ├── feature_importance.py
│       ├── shap_explainer.py
│       ├── train.py
│
├── model_registry/
│   ├── breast_cancer/
│   ├── diabetes/
│   ├── heart/
│
├── reports/
│   ├── breast_cancer/
│   ├── diabetes/
│   ├── heart/
│
├── services/
│   ├── api_service.py
│   ├── feature_builder.py
│   ├── model_registry.py
│   ├── predictor.py
│
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

---

# 🚀 Real-World Impact

* Enables **early disease detection**
* Reduces **clinical risk and treatment cost**
* Supports **doctor decision-making**
* Provides **personalized healthcare insights**
* Deployable across **clinics, hospitals, digital health platforms**

---

# 🔮 Future Scope

* Integration with **EHR systems**
* Wearable device data ingestion
* Advanced deep learning models
* Real-time patient monitoring
* Mobile healthcare applications
* Multi-language support

---

# ⚠️ Disclaimer

This system is intended for **decision support only** and is **not a medical diagnosis tool**.
Always consult a qualified healthcare professional.

---

# 🙌 Acknowledgement

This project demonstrates the application of **AI, machine learning, and explainable systems in healthcare**, bridging the gap between **predictive models and real-world clinical decision-making**.

---
