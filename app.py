import streamlit as st
import pickle
import numpy as np

st.set_page_config(page_title="Logistic Regression App")

st.title("🚀 Logistic Regression Deployment")
st.write("This app predicts output using a trained Logistic Regression model.")

# Load model
@st.cache_resource
def load_model():
    with open("logistic_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

st.subheader("Enter Input Features")

# 🔥 Change these according to your dataset
feature1 = st.number_input("Feature 1")
feature2 = st.number_input("Feature 2")
feature3 = st.number_input("Feature 3")
feature4 = st.number_input("Feature 4")

if st.button("Predict"):
    
    input_data = np.array([[feature1, feature2, feature3, feature4]])
    
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)

    st.success(f"Predicted Class: {prediction[0]}")
    st.write("Prediction Probability:", probability)