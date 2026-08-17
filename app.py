# ─────────────────────────────────────────────
# Streamlit App - Hospital Uncompensated Care 
# Cost Predictor
# User inputs hospital details and gets a 
# predicted uncompensated care cost
# ─────────────────────────────────────────────

import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load the saved model
model = joblib.load('model/random_forest_model.pkl')

# App title and description
st.title("🏥 Hospital Uncompensated Care Cost Predictor")
st.markdown("""
This tool predicts how much **uncompensated care cost** a hospital 
will absorb based on its size, location, and financial profile.
Built using CMS HCRIS data (2019–2023) and a Random Forest model 
with R² = 0.9977.
""")

st.divider()

# ── Input Section ──
st.header("Enter Hospital Details")

col1, col2 = st.columns(2)

with col1:
    beds = st.number_input("Number of Beds", min_value=1, max_value=2000, value=150)
    fte = st.number_input("FTE Employees on Payroll", min_value=1, max_value=20000, value=500)
    discharges = st.number_input("Total Discharges", min_value=0, max_value=100000, value=5000)
    total_costs = st.number_input("Total Costs ($)", min_value=0, max_value=5000000000, value=50000000, step=1000000)
    net_income = st.number_input("Net Income ($)", min_value=-500000000, max_value=500000000, value=1000000, step=100000)

with col2:
    is_urban = st.selectbox("Location Type", options=["Rural", "Urban"])
    medicaid_days = st.number_input("Total Medicaid Days (Title XIX)", min_value=0, max_value=200000, value=10000)
    medicare_days = st.number_input("Total Medicare Days (Title XVIII)", min_value=0, max_value=200000, value=20000)
    total_patient_revenue = st.number_input("Total Patient Revenue ($)", min_value=0, max_value=5000000000, value=80000000, step=1000000)

st.divider()

# ── Predict Button ──
if st.button("Predict Uncompensated Care Cost", type="primary"):

    # Build feature array in same order as training
    uncompensated_per_bed = total_costs / max(beds, 1)
    is_urban_encoded = 1 if is_urban == "Urban" else 0

    features = pd.DataFrame([{
        'Number of Beds': beds,
        'FTE - Employees on Payroll': fte,
        'Total Days Title XVIII': medicare_days,
        'Total Days Title XIX': medicaid_days,
        'Total Discharges (V + XVIII + XIX + Unknown)': discharges,
        'Total Patient Revenue': total_patient_revenue,
        'Net Patient Revenue': total_patient_revenue * 0.85,
        'Net Income': net_income,
        'Net Revenue from Medicaid': medicaid_days * 500,
        'Total Assets': total_costs * 1.5,
        'Total Liabilities': total_costs * 0.8,
        'Total Costs': total_costs,
        'Disproportionate Share Adjustment': 0,
        'Type of Control': 2,
        'Provider Type': 1,
        'Is_Urban': is_urban_encoded,
        'Uncompensated_Care_Per_Bed': uncompensated_per_bed,
        'year' : 2023,
    }])

    # Predict in log scale then reverse to dollars
    log_prediction = model.predict(features)[0]
    dollar_prediction = np.expm1(log_prediction)

    # Display result
    st.success(f"### Predicted Uncompensated Care Cost:")
    st.metric(label="Estimated Cost", value=f"${dollar_prediction:,.0f}")

    # Context
    st.markdown(f"""
    **What this means:**
    - This hospital is predicted to absorb **${dollar_prediction:,.0f}** 
      in uncompensated care
    - National median is **$2,894,337** per hospital per year
    - This hospital is {'above' if dollar_prediction > 2894337 else 'below'} the national median
    """)