import streamlit as st
import pickle
import numpy as np

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Diabetes Prediction App",
    page_icon="🩺",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.main {
    background-color: #f4f6f9;
}
.stButton>button {
    background-color: #2e86de;
    color: white;
    font-size: 16px;
    border-radius: 8px;
    height: 3em;
    width: 100%;
}
.stButton>button:hover {
    background-color: #1b4f72;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    with open("logistic_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

# ---------------- SIDEBAR ----------------
st.sidebar.title("🧠 About This Model")
st.sidebar.write("""
This app predicts whether a patient is likely to have **Diabetes**
using a trained **Logistic Regression model**.

Dataset Features:
- Pregnancies
- Glucose
- Blood Pressure
- Skin Thickness
- Insulin
- BMI
- Diabetes Pedigree Function
- Age
""")

st.sidebar.markdown("---")
st.sidebar.info("Developed by Alex Anto 🚀")

# ---------------- MAIN TITLE ----------------
st.title("🩺 Diabetes Prediction System")
st.markdown("### Machine Learning Deployment using Logistic Regression")

st.markdown("---")

# ---------------- INPUT SECTION ----------------
st.subheader("📋 Enter Patient Details")

col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Pregnancies", min_value=0, value=0)
    glucose = st.number_input("Glucose Level", min_value=0.0, value=100.0, format="%.2f")
    blood_pressure = st.number_input("Blood Pressure", min_value=0.0, value=70.0, format="%.2f")
    skin_thickness = st.number_input("Skin Thickness", min_value=0.0, value=20.0, format="%.2f")

with col2:
    insulin = st.number_input("Insulin", min_value=0.0, value=80.0, format="%.2f")
    bmi = st.number_input("BMI", min_value=0.0, value=25.0, format="%.2f")
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, value=0.5, format="%.2f")
    age = st.number_input("Age", min_value=0, value=30)

st.markdown("---")

# ---------------- PREDICTION ----------------
if st.button("🔮 Predict Diabetes Risk"):

    input_data = np.array([[pregnancies,
                            glucose,
                            blood_pressure,
                            skin_thickness,
                            insulin,
                            bmi,
                            dpf,
                            age]])

    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)

    diabetic_prob = float(probability[0][1])
    non_diabetic_prob = float(probability[0][0])

    st.subheader("📊 Prediction Result")

    if prediction[0] == 1:
        st.error("⚠ High Risk: Patient is Likely Diabetic")
    else:
        st.success("✅ Low Risk: Patient is Likely Not Diabetic")

    st.markdown("### 📈 Risk Probability")

    st.progress(diabetic_prob)

    col3, col4 = st.columns(2)
    col3.metric("Diabetes Probability", f"{diabetic_prob*100:.2f}%")
    col4.metric("No Diabetes Probability", f"{non_diabetic_prob*100:.2f}%")

st.markdown("---")

# ---------------- FOOTER ----------------
st.markdown("""
<center>
<small>
Diabetes Prediction App | Logistic Regression | Streamlit Deployment | 2026
</small>
</center>
""", unsafe_allow_html=True)
