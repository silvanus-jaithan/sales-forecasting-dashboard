import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load("model/model.pkl")

st.title("📈 Sales Forecasting Dashboard")

advertising = st.number_input("Advertising Budget")
holiday = st.selectbox("Holiday?", [0, 1])
temperature = st.number_input("Temperature")
customers = st.number_input("Customers")

if st.button("Predict Sales"):
    data = np.array([[advertising, holiday, temperature, customers]])
    prediction = model.predict(data)

    st.success(f"Predicted Sales: ${prediction[0]:.2f}")
