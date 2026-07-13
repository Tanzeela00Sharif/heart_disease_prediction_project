import pandas as pd
import numpy as np
import joblib
import streamlit as st
import sklearn

# -----------------------------
# Load trained model and columns
# -----------------------------
model = joblib.load('diabeties_model.pkl')
model_columns = joblib.load('diabeties_columns.pkl')

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Diabetes Prediction App",
    page_icon="🩺",
    layout="wide"
)

# -----------------------------
# Global CSS — Design Language
# -----------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@700;800&family=Open+Sans:wght@400;600&display=swap');

/* Kill default Streamlit padding so hero can bleed edge-to-edge */
.block-container {
    padding-top: 0rem;
    padding-bottom: 0rem;
    padding-left: 0rem;
    padding-right: 0rem;
    max-width: 100%;
}
header[data-testid="stHeader"] {
    background: #0D0D0D;
}
[data-testid="stAppViewContainer"] {
    background: #FFFFFF;
}

/* ---------- NAVBAR ---------- */
.navbar {
    background-color: #0D0D0D;
    padding: 18px 48px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid #262626;
}
.navbar-title {
    color: #FFFFFF;
    font-family: 'Montserrat', sans-serif;
    font-weight: 800;
    font-size: 20px;
    letter-spacing: 1px;
}
.navbar-links a {
    color: #E5E5E5;
    font-family: 'Open Sans', sans-serif;
    font-size: 13px;
    letter-spacing: 1px;
    margin-left: 28px;
    text-decoration: none;
}

/* ---------- HERO ---------- */
.hero {
    background-color: #0D0D0D;
    padding: 80px 48px 90px 48px;
    position: relative;
    overflow: hidden;
}
.hero-eyebrow {
    color: #E8385C;
    font-family: 'Open Sans', sans-serif;
    font-weight: 600;
    letter-spacing: 3px;
    font-size: 12px;
    margin-bottom: 14px;
}
.hero-title {
    color: #FFFFFF;
    font-family: 'Montserrat', sans-serif;
    font-weight: 800;
    font-size: 46px;
    text-transform: uppercase;
    letter-spacing: 1px;
    line-height: 1.15;
    max-width: 720px;
}
.hero-title span {
    color: #E8385C;
}
.hero-desc {
    color: #C9C9C9;
    font-family: 'Open Sans', sans-serif;
    font-size: 15px;
    line-height: 1.6;
    max-width: 560px;
    margin-top: 22px;
}
.hero-badges {
    margin-top: 34px;
    display: flex;
    gap: 14px;
}
.badge-primary {
    background-color: #E8385C;
    color: #FFFFFF;
    font-family: 'Open Sans', sans-serif;
    font-weight: 700;
    font-size: 12px;
    letter-spacing: 1px;
    padding: 12px 26px;
    border-radius: 4px;
    display: inline-block;
}
.badge-secondary {
    background-color: #E8A33D;
    color: #0D0D0D;
    font-family: 'Open Sans', sans-serif;
    font-weight: 700;
    font-size: 12px;
    letter-spacing: 1px;
    padding: 12px 26px;
    border-radius: 4px;
    display: inline-block;
}
.pulse-line {
    width: 100%;
    height: 60px;
    margin-top: 50px;
    opacity: 0.85;
}

/* ---------- FORM SECTION ---------- */
.form-section {
    background: linear-gradient(180deg, #FFFFFF 0%, #EAF2F5 45%, #C3D8DE 100%);
    padding: 60px 48px 70px 48px;
}
.form-heading {
    text-align: center;
    color: #4A4A4A;
    font-family: 'Montserrat', sans-serif;
    font-weight: 800;
    font-size: 30px;
    letter-spacing: 1px;
    text-transform: uppercase;
}
.form-subheading {
    text-align: center;
    color: #6E8A94;
    font-family: 'Open Sans', sans-serif;
    font-size: 15px;
    margin-top: 6px;
    margin-bottom: 42px;
}

/* ---------- INPUT FIELDS ---------- */
div[data-testid="stNumberInput"] label,
div[data-testid="stNumberInputInputComponent"] label {
    font-family: 'Open Sans', sans-serif;
    font-weight: 600;
    color: #4A4A4A;
    font-size: 13px;
}
div[data-testid="stNumberInput"] input {
    background-color: #FBFBC9 !important;
    border: 1px solid #E5E5C9 !important;
    border-radius: 4px !important;
    height: 42px !important;
    font-family: 'Open Sans', sans-serif;
    color: #333333 !important;
}
div[data-testid="stNumberInput"] {
    background-color: #E5E5E5;
    border-radius: 4px;
    padding: 6px 6px 2px 6px;
}

/* ---------- BUTTON ---------- */
div.stButton > button {
    background-color: #E8385C;
    color: #FFFFFF;
    font-family: 'Open Sans', sans-serif;
    font-weight: 700;
    letter-spacing: 1px;
    font-size: 13px;
    text-transform: uppercase;
    border: none;
    border-radius: 4px;
    padding: 12px 40px;
    width: 100%;
    box-shadow: none;
}
div.stButton > button:hover {
    background-color: #D12E4F;
    color: #FFFFFF;
}

/* ---------- RESULT / FOOTER ---------- */
.result-section {
    background-color: #FFFFFF;
    padding: 10px 48px 40px 48px;
}
.footer {
    background-color: #0D0D0D;
    padding: 30px 48px;
    text-align: center;
    color: #7A7A7A;
    font-family: 'Open Sans', sans-serif;
    font-size: 12px;
    letter-spacing: 1px;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# NAVBAR
# -----------------------------
st.markdown("""
<div class="navbar">
    <div class="navbar-title">🩺 DIABETES PREDICTOR</div>
    <div class="navbar-links">
        <a href="#">ABOUT</a>
        <a href="#">HOW IT WORKS</a>
    </div>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# HERO SECTION
# -----------------------------
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">MACHINE LEARNING · HEALTH SCREENING</div>
    <div class="hero-title">PREDICTING YOUR <span>DIABETES</span> RISK !</div>
    <div class="hero-desc">
        This tool uses a trained machine learning model to estimate the likelihood of diabetes
        based on patient health metrics. Enter the details below to generate a prediction.
        For educational purposes only — not a substitute for professional medical advice.
    </div>
    <div class="hero-badges">
        <div class="badge-primary">CHECK YOUR RISK</div>
        <div class="badge-secondary">HOW IT WORKS</div>
    </div>
    <svg class="pulse-line" viewBox="0 0 1200 60" preserveAspectRatio="none">
        <polyline points="0,30 100,30 130,5 160,55 190,30 350,30 380,10 410,50 440,30 600,30 630,5 660,55 690,30 850,30 880,10 910,50 940,30 1100,30 1130,5 1160,55 1190,30 1200,30"
        fill="none" stroke="#E8385C" stroke-width="2.5"/>
    </svg>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# FORM SECTION
# -----------------------------
st.markdown('<div class="form-section">', unsafe_allow_html=True)
st.markdown('<div class="form-heading">PATIENT DATA</div>', unsafe_allow_html=True)
st.markdown('<div class="form-subheading">Input the patient\'s health metrics below</div>', unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
with c1:
    pregnancies = st.number_input("🩸 Pregnancies", min_value=0, max_value=20, value=1, step=1)
    insulin = st.number_input("🩸 Insulin Level", min_value=0, max_value=900, value=80)
with c2:
    glucose = st.number_input("🩸 Glucose Level", min_value=0, max_value=300, value=120)
    bmi = st.number_input("🩸 BMI", min_value=0.0, max_value=70.0, value=25.0, step=0.1)
with c3:
    blood_pressure = st.number_input("🩸 Blood Pressure (mm Hg)", min_value=0, max_value=200, value=70)
    dpf = st.number_input("🩸 Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, step=0.01)
with c4:
    skin_thickness = st.number_input("🩸 Skin Thickness (mm)", min_value=0, max_value=100, value=20)
    age = st.number_input("🩸 Age", min_value=1, max_value=120, value=30, step=1)

st.write("")
b1, b2, b3 = st.columns([1, 1, 1])
with b2:
    predict_clicked = st.button("ANALYZE", type="primary")

st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# Prediction
# -----------------------------
st.markdown('<div class="result-section">', unsafe_allow_html=True)

if predict_clicked:
    input_dict = {
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": dpf,
        "Age": age,
    }

    input_df = pd.DataFrame([input_dict])
    input_df = input_df.reindex(columns=model_columns, fill_value=0)
    prediction = model.predict(input_df)[0]

    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(input_df)[0][1]

    st.subheader("Result")
    if prediction == 1:
        st.error("⚠️ The model predicts this patient is **likely diabetic**.")
    else:
        st.success("✅ The model predicts this patient is **not likely diabetic**.")

    if proba is not None:
        st.write(f"Predicted probability of diabetes: **{proba:.2%}**")
        st.progress(min(max(proba, 0.0), 1.0))

    st.caption(
        "Disclaimer: This prediction is generated by a machine learning model and should "
        "not be used as a medical diagnosis. Please consult a healthcare professional."
    )

st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("""
<div class="footer">
    © 2026 DIABETES PREDICTOR · FOR EDUCATIONAL USE ONLY
</div>
""", unsafe_allow_html=True)
