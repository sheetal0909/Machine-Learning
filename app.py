import streamlit as st
import pickle

# Load the trained model
model = pickle.load(open(
    r"C:\Users\sheet\AVSCode\Machine Learning\Regression\linear_regression_model.pkl",
    "rb"
))

# App title
st.title("Salary Prediction App")

# Description
st.write(
    "This app predicts the salary based on years of experience "
    "using a simple linear regression model."
)

# Get years of experience
years = st.number_input(
    "Enter years of experience:",
    min_value=0.0,
    max_value=50.0,
    value=1.0
)

# Prediction button
if st.button("Predict Salary"):
    prediction = model.predict([[years]])
    st.success(f"Predicted Salary: ${prediction[0]:,.2f}")