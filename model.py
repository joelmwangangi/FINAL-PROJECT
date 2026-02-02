# =====================================
# app.py — Smart Loan Default Predictor
# =====================================

import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

# -----------------------------
# 1. Load trained model
# -----------------------------
MODEL_PATH = "models/random_forest_model.joblib"  # You can switch to mlp_model.joblib
SCALER_PATH = "models/scaler.joblib"

st.set_page_config(page_title="Loan Default Predictor", page_icon="💰", layout="centered")

st.title("💰 Loan Default Prediction System")
st.write("Predict the likelihood of a loan applicant defaulting based on financial and demographic data.")

# Load model
if not os.path.exists(MODEL_PATH):
    st.error("❌ Model not found. Please ensure your trained model is in the 'models/' folder.")
    st.stop()

model = joblib.load(MODEL_PATH)
st.success("✅ Model loaded successfully!")

# Load scaler if available
scaler = joblib.load(SCALER_PATH) if os.path.exists(SCALER_PATH) else None

# -----------------------------
# 2. Prepare dynamic input fields
# -----------------------------
features = model.feature_names_in_ if hasattr(model, "feature_names_in_") else []

# If no feature names available, ask user for manual CSV
if len(features) == 0:
    st.warning("⚠️ Model does not contain feature names. Please ensure it was trained with scikit-learn ≥1.0.")
    st.stop()

# Create a form dynamically
st.subheader("📋 Enter Borrower Information")

user_input = {}
for feat in features:
    # Guess numeric vs categorical based on name
    if any(k in feat.lower() for k in ["age","income","amount","score","year","balance","debt","payment","rate"]):
        user_input[feat] = st.number_input(f"{feat.replace('_',' ').title()}", value=0.0)
    elif any(k in feat.lower() for k in ["gender","education","marital","job","region","state","city"]):
        user_input[feat] = st.selectbox(f"{feat.replace('_',' ').title()}", ["Select", "Option1", "Option2", "Option3"])
    else:
        # Default fallback
        user_input[feat] = st.text_input(f"{feat.replace('_',' ').title()}", "")

# Convert to dataframe
input_df = pd.DataFrame([user_input])

# Handle non-numeric dummies
for col in features:
    if col not in input_df.columns:
        input_df[col] = 0
input_df = input_df[features]

# Scale if scaler available
if scaler:
    input_df = pd.DataFrame(scaler.transform(input_df), columns=features)

# -----------------------------
# 3. Prediction
# -----------------------------
if st.button("🔮 Predict Loan Default"):
    prob = model.predict_proba(input_df)[0][1]
    label = "❌ Likely to Default" if prob >= 0.5 else "✅ Unlikely to Default"
    st.subheader(label)
    st.metric("Predicted Probability of Default", f"{prob*100:.2f}%")

    if prob >= 0.7:
        st.warning("⚠️ High risk — consider strict lending conditions.")
    elif prob >= 0.5:
        st.info("🟠 Moderate risk — manual review recommended.")
    else:
        st.success("🟢 Low risk — borrower seems reliable.")

st.markdown("---")
st.caption("Powered by Scikit-learn • Streamlit • Loan Risk Intelligence System")

