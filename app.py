# ======================================================
# Diabetes Prediction App - Optimized & Visual Version
# Logistic Regression + StandardScaler
# ======================================================

import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Diabetes Risk Prediction",
    page_icon="🩺",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    with open("logistic_model.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_scaler():
    with open("scaler.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()
scaler = load_scaler()

# ---------------- SIDEBAR ----------------
st.sidebar.title("🧠 Model Info")
st.sidebar.write("""
This app predicts Diabetes risk using:

- Logistic Regression
- StandardScaler
- Trained on Pima Diabetes Dataset
""")

st.sidebar.markdown("---")
st.sidebar.success("Developed by Alex Anto 🚀")

# ---------------- MAIN TITLE ----------------
st.title("🩺 Diabetes Risk Prediction System")
st.markdown("### Machine Learning Deployment with Scaling & Visual Analytics")
st.markdown("---")

# ---------------- INPUT SECTION ----------------
st.subheader("📋 Enter Patient Details")

col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Pregnancies", min_value=0, value=0)
    glucose = st.number_input("Glucose Level", min_value=0.0, value=100.0)
    blood_pressure = st.number_input("Blood Pressure", min_value=0.0, value=70.0)
    skin_thickness = st.number_input("Skin Thickness", min_value=0.0, value=20.0)

with col2:
    insulin = st.number_input("Insulin", min_value=0.0, value=80.0)
    bmi = st.number_input("BMI", min_value=0.0, value=25.0)
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, value=0.5)
    age = st.number_input("Age", min_value=0, value=30)

st.markdown("---")

# ---------------- PREDICTION ----------------
if st.button("🔮 Predict Diabetes Risk"):

    # Correct feature order (must match training order)
    input_data = np.array([[pregnancies,
                            glucose,
                            blood_pressure,
                            skin_thickness,
                            insulin,
                            bmi,
                            dpf,
                            age]])

    # 🔥 APPLY SCALING (CRITICAL)
    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)

    diabetic_prob = float(probability[0][1])
    non_diabetic_prob = float(probability[0][0])

    # ---------------- RISK STATUS ----------------
    st.subheader("📊 Prediction Result")

    if diabetic_prob < 0.30:
        st.success("🟢 Low Risk of Diabetes")
    elif diabetic_prob < 0.60:
        st.warning("🟡 Moderate Risk of Diabetes")
    else:
        st.error("🔴 High Risk of Diabetes")

    # ---------------- GAUGE CHART ----------------
    st.markdown("### 📈 Risk Gauge")

    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=diabetic_prob * 100,
        title={'text': "Diabetes Probability (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "red"},
            'steps': [
                {'range': [0, 30], 'color': "green"},
                {'range': [30, 60], 'color': "yellow"},
                {'range': [60, 100], 'color': "red"}
            ],
        }
    ))

    st.plotly_chart(gauge, use_container_width=True)

    # ---------------- PROBABILITY BAR CHART ----------------
    st.markdown("### 📊 Probability Distribution")

    prob_df = pd.DataFrame({
        "Class": ["No Diabetes", "Diabetes"],
        "Probability": [non_diabetic_prob, diabetic_prob]
    })

    bar_chart = px.bar(
        prob_df,
        x="Class",
        y="Probability",
        text="Probability",
        color="Class",
        color_discrete_sequence=["green", "red"]
    )

    bar_chart.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    bar_chart.update_layout(yaxis_range=[0,1])

    st.plotly_chart(bar_chart, use_container_width=True)

    # ---------------- FEATURE CONTRIBUTION ----------------
    st.markdown("### 📈 Feature Influence (Model Coefficients)")

    coef_df = pd.DataFrame({
        "Feature": ["Pregnancies","Glucose","BloodPressure",
                    "SkinThickness","Insulin","BMI",
                    "DPF","Age"],
        "Coefficient": model.coef_[0]
    }).sort_values(by="Coefficient")

    coef_chart = px.bar(
        coef_df,
        x="Coefficient",
        y="Feature",
        orientation="h",
        color="Coefficient",
        color_continuous_scale="RdBu"
    )

    st.plotly_chart(coef_chart, use_container_width=True)

st.markdown("---")
st.markdown("""
<center>
<small>
Logistic Regression | StandardScaler Applied | Interactive Visual Analytics | 2026
</small>
</center>
""", unsafe_allow_html=True)
