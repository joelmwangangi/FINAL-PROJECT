# app.py — Loan Default Prediction Interface using Streamlit

import streamlit as st
import pandas as pd
import joblib
import numpy as np

# -----------------------------
# Load trained model and objects
# -----------------------------
rf_model = joblib.load("models/random_forest_model.joblib")   # or mlp_model.joblib
scaler = joblib.load("models/scaler.joblib") if os.path.exists("models/scaler.joblib") else None

st.set_page_config(page_title="Loan Default Predictor", page_icon="💰")

# -----------------------------
# UI Title
# -----------------------------
st.title("💰 Loan Default Prediction System")
st.write("Enter borrower details below to predict whether they are likely to default.")

# -----------------------------
# Define input fields
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 18, 100, 30)
    income = st.number_input("Annual Income (USD)", 0, 1000000, 50000)
    loan_amount = st.number_input("Loan Amount", 0, 500000, 10000)
    credit_score = st.number_input("Credit Score", 300, 850, 650)

with col2:
    dependents = st.number_input("Number of Dependents", 0, 10, 0)
    employment_years = st.number_input("Years at Current Job", 0, 40, 5)
    education = st.selectbox("Education Level", ["High School", "Bachelor", "Master", "PhD"])
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Widowed"])

# -----------------------------
# Convert inputs into model-ready dataframe
# -----------------------------
input_data = pd.DataFrame({
    "age": [age],
    "income": [income],
    "loan_amount": [loan_amount],
    "credit_score": [credit_score],
    "dependents": [dependents],
    "employment_years": [employment_years],
    "education": [education],
    "marital_status": [marital_status]
})

# Convert categorical features to dummy variables (same as training)
input_data = pd.get_dummies(input_data, drop_first=True)

# Align columns with model (handle missing dummies)
model_features = rf_model.feature_names_in_
for col in model_features:
    if col not in input_data.columns:
        input_data[col] = 0
input_data = input_data[model_features]

# Scale if scaler available
if scaler:
    input_data = pd.DataFrame(scaler.transform(input_data), columns=model_features)

# -----------------------------
# Predict
# -----------------------------
if st.button("🔮 Predict Default Likelihood"):
    pred_prob = rf_model.predict_proba(input_data)[0][1]
    pred_label = "❌ Likely to Default" if pred_prob >= 0.5 else "✅ Unlikely to Default"
    st.subheader(pred_label)
    st.metric("Predicted Probability of Default", f"{pred_prob*100:.2f}%")

    if pred_prob >= 0.7:
        st.warning("⚠️ High risk! Consider stricter loan approval conditions.")
    elif pred_prob >= 0.5:
        st.info("🟠 Moderate risk. Review credit history carefully.")
    else:
        st.success("🟢 Low risk. Borrower looks financially stable.")
