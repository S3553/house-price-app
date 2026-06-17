import streamlit as st
import numpy as np
import joblib

# ---------------------------
# PAGE CONFIG (MUST BE FIRST)
# ---------------------------
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="centered"
)

# ---------------------------
# LOAD MODEL
# ---------------------------
model = joblib.load("model.pkl")

# ---------------------------
# TITLE
# ---------------------------
st.title("🏠 House Price Prediction App")
st.markdown("### Predict house price using Machine Learning 🤖")

st.divider()

# ---------------------------
# INPUT SECTION
# ---------------------------
st.subheader("Enter House Details")

col1, col2 = st.columns(2)

with col1:
    area = st.number_input("Area (sq ft)", min_value=0)
    bedrooms = st.number_input("Bedrooms", min_value=0)
    bathrooms = st.number_input("Bathrooms", min_value=0)

with col2:
    stories = st.number_input("Stories", min_value=0)
    parking = st.number_input("Parking spaces", min_value=0)

st.divider()

# ---------------------------
# PREDICTION
# ---------------------------
if st.button("💰 Predict Price"):
    input_data = np.array([[area, bedrooms, bathrooms, stories, parking]])
    prediction = model.predict(input_data)

    st.success(f"🏡 Estimated House Price: ₹ {prediction[0]:,.2f}")
    st.balloons()

# ---------------------------
# FOOTER
# ---------------------------
st.markdown("---")
st.markdown("Built with ❤️ using Machine Learning + Streamlit")
