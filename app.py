import streamlit as st
import pickle
import numpy as np

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Logistic Regression ML App",
    page_icon="📊",
    layout="wide"
)

# ---------------- CUSTOM STYLING ----------------
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 16px;
        border-radius: 8px;
        height: 3em;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #45a049;
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
st.sidebar.title("📌 About This App")
st.sidebar.write("""
This application deploys a trained **Logistic Regression model**
using Streamlit Community Cloud.

Developed for Machine Learning deployment.
""")

st.sidebar.markdown("---")
st.sidebar.info("Built by Alex Anto 🚀")

# ---------------- MAIN TITLE ----------------
st.title("🚀 Logistic Regression Deployment Dashboard")
st.markdown("### Interactive Machine Learning Prediction System")

st.markdown("---")

# ---------------- INPUT SECTION ----------------
st.subheader("🔢 Enter Feature Values")

col1, col2 = st.columns(2)

with col1:
    feature1 = st.number_input("Feature 1", value=0.0)
    feature2 = st.number_input("Feature 2", value=0.0)

with col2:
    feature3 = st.number_input("Feature 3", value=0.0)
    feature4 = st.number_input("Feature 4", value=0.0)

st.markdown("---")

# ---------------- PREDICTION ----------------
if st.button("🔮 Predict Now"):

    input_data = np.array([[feature1, feature2, feature3, feature4]])

    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)

    st.subheader("📊 Prediction Result")

    if prediction[0] == 1:
        st.success("Predicted Class: 1")
    else:
        st.error("Predicted Class: 0")

    st.write("### 📈 Prediction Probability")

    st.progress(float(probability[0][1]))

    st.write("Probability Distribution:")
    st.write({
        "Class 0": float(probability[0][0]),
        "Class 1": float(probability[0][1])
    })

st.markdown("---")

# ---------------- FOOTER ----------------
st.markdown("""
<center>
    <small>
    Logistic Regression Model Deployment | Streamlit Cloud | 2026
    </small>
</center>
""", unsafe_allow_html=True)
