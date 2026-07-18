import pandas as pd
import numpy as np
import joblib
import streamlit as st
import sklearn

# -----------------------------
# Load trained model and columns
# -----------------------------
model = joblib.load('heart_Diasease_model.pkl')
model_columns = joblib.load('heart_Disease_columns.pkl')  # NOTE: verify this matches the exact filename you saved with joblib.dump

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Heart Disease Prediction App",
    page_icon="❤️",
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
    padding: 40px 48px 36px 48px;
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
    height: 50px;
    margin-top: 30px;
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
.result-text {
    font-family: 'Montserrat', sans-serif;
    font-weight: 800;
    font-size: 34px;
    letter-spacing: 0.5px;
    margin: 14px 0 6px 0;
    line-height: 1.3;
}
.result-positive {
    color: #C62828;
}
.result-negative {
    color: #1B5E20;
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
    <div class="navbar-title" style="display:flex; align-items:center; gap:10px;">
        <svg width="22" height="22" viewBox="0 0 24 24" fill="#E8385C">
            <path d="M12 21.35l-1.45-1.32C5.4 15.36 2 12.28 2 8.5
                     2 5.42 4.42 3 7.5 3c1.74 0 3.41 0.81 4.5 2.09
                     C13.09 3.81 14.76 3 16.5 3 19.58 3 22 5.42 22 8.5
                     c0 3.78-3.4 6.86-8.55 11.54L12 21.35z"/>
        </svg>
        HEART DISEASE PREDICTOR
    </div>
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

    <div class="hero-title">
        PREDICTING YOUR <span>HEART DISEASE</span> RISK !
    </div>

    <div class="hero-desc">
        This tool uses a trained machine learning model to estimate the likelihood of heart
        disease based on patient health metrics. Enter the details below to generate a prediction.
        For educational purposes only — not a substitute for professional medical advice.
    </div>

    <div class="hero-badges">
        <div class="badge-primary">CHECK YOUR RISK</div>
    </div>

    <svg class="pulse-line" viewBox="0 0 1200 60" preserveAspectRatio="none">
        <polyline 
        points="0,30 100,30 130,5 160,55 190,30 350,30 380,10 410,50 440,30 600,30 630,5 660,55 690,30 850,30 880,10 910,50 940,30 1100,30 1130,5 1160,55 1190,30 1200,30"
        fill="none" 
        stroke="#E8385C" 
        stroke-width="2.5"/>
    </svg>

</div>
""", unsafe_allow_html=True)

# -----------------------------
# FORM SECTION
# -----------------------------
st.markdown('<div class="form-section">', unsafe_allow_html=True)
st.markdown('<div class="form-heading">PATIENT DATA</div>', unsafe_allow_html=True)
st.markdown('<div class="form-subheading">Input the patient\'s health metrics below to check heart disease risk</div>', unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
with c1:
    Age = st.number_input("🩸 Age", min_value=1, max_value=120, value=45, step=1)
    Fasting_Blood_Sugar = st.selectbox("🩸 Fasting Blood Sugar > 120 mg/dl", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    ST_Slope = st.selectbox("🩸 Slope of Peak Exercise ST Segment", options=[0, 1, 2])
with c2:
    Gender = st.selectbox("🩸 Sex", options=[0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
    Resting_ECG_Result = st.selectbox("🩸 Resting ECG Results", options=[0, 1, 2])
    Major_Vessels = st.selectbox("🩸 Number of Major Vessels (0–3)", options=[0, 1, 2, 3])
with c3:
    Chest_Pain_Type = st.selectbox("🩸 Chest Pain Type", options=[0, 1, 2, 3])
    Max_Heart_Rate = st.number_input("🩸 Max Heart Rate Achieved", min_value=60, max_value=220, value=150)
    Thalassemia = st.selectbox("🩸 Thalassemia", options=[0, 1, 2, 3])
with c4:
    Resting_Blood_Pressure = st.number_input("🩸 Resting Blood Pressure", min_value=80, max_value=220, value=120)
    Exercise_Induced_Angina = st.selectbox("🩸 Exercise Induced Angina", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    ST_Depression = st.number_input("🩸 ST Depression (Oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)

Cholesterol_Level = st.number_input("🩸 Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)

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
        "Age": Age,
        "Gender": Gender,
        "Chest_Pain_Type": Chest_Pain_Type,
        "Resting_Blood_Pressure": Resting_Blood_Pressure,
        "Cholesterol_Level": Cholesterol_Level,
        "Fasting_Blood_Sugar": Fasting_Blood_Sugar,
        "Resting_ECG_Result": Resting_ECG_Result,
        "Max_Heart_Rate": Max_Heart_Rate,
        "Exercise_Induced_Angina": Exercise_Induced_Angina,
        "ST_Depression": ST_Depression,
        "ST_Slope": ST_Slope,
        "Major_Vessels": Major_Vessels,
        "Thalassemia": Thalassemia,
    }

    input_df = pd.DataFrame([input_dict])
    input_df = input_df.reindex(columns=model_columns, fill_value=0)
    prediction = model.predict(input_df)[0]

    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(input_df)[0][1]

    st.subheader("Result")
    if prediction == 1:
        st.markdown(
            '<div class="result-text result-positive">⚠️ LIKELY HEART DISEASE</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="result-text result-negative">✅ NOT LIKELY HEART DISEASE</div>',
            unsafe_allow_html=True
        )

    if proba is not None:
        st.write(f"Predicted probability of heart disease: **{proba:.2%}**")
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
    © 2026 HEART DISEASE PREDICTOR · FOR EDUCATIONAL USE ONLY
</div>
""", unsafe_allow_html=True)
