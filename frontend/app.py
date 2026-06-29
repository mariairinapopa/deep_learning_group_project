"""
Step 5 — Frontend: a Streamlit form a non-technical HR user can fill in to get a
live attrition probability for one employee.

The integration story (what the rubric calls "seamless end-to-end"):
- Load the trained .keras model, the fitted scaler, and the saved feature-column
  order ONCE (cached).
- Collect a handful of human-readable inputs from the form.
- Reconstruct a single feature row in the EXACT column order the model was
  trained on, one-hot encode the categoricals the same way, scale the same
  numeric columns with the same scaler, and call model.predict().
- Show the probability as a verdict + gauge, in real time, on submit.

Run from the project root with:  streamlit run frontend/app.py
"""

from tensorflow import keras
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "attrition_ann.keras"
SCALER_PATH = ROOT / "models" / "scaler.joblib"
COLS_PATH = ROOT / "models" / "feature_columns.json"


@st.cache_resource
def load_artifacts():
    model = keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    with open(COLS_PATH) as f:
        meta = json.load(f)
    return model, scaler, meta["feature_columns"], meta["numeric_cols"]


def build_row(form: dict, feature_columns, numeric_cols, scaler) -> np.ndarray:
    """Turn raw form inputs into a scaled, correctly-ordered feature vector."""
    # One-row frame, one-hot encode, then reindex to the trained column space.
    df = pd.DataFrame([form])
    df = pd.get_dummies(df)
    # Any trained column missing from this single row gets filled with 0.
    df = df.reindex(columns=feature_columns, fill_value=0)

    numeric_idx = [feature_columns.index(c) for c in numeric_cols]
    arr = df.values.astype("float32")
    arr[:, numeric_idx] = scaler.transform(arr[:, numeric_idx])
    return arr


st.set_page_config(page_title="Attrition Predictor", page_icon="📉")
st.title("Employee Attrition Predictor")
st.caption("Estimate the probability that an employee is about to leave.")

model, scaler, feature_columns, numeric_cols = load_artifacts()

# A pragmatic subset of the most decision-relevant inputs. Anything not shown is
# reindexed to 0 (for dummies) — fine for a demo; expand the form for production.
col1, col2 = st.columns(2)
with col1:
    age = st.slider("Age", 18, 60, 35)
    monthly_income = st.number_input("Monthly income", 1000, 20000, 5000, step=250)
    years_at_company = st.slider("Years at company", 0, 40, 5)
    job_satisfaction = st.select_slider("Job satisfaction (1-4)", [1, 2, 3, 4], 3)
    work_life_balance = st.select_slider("Work-life balance (1-4)", [1, 2, 3, 4], 3)
with col2:
    overtime = st.selectbox("Works overtime?", ["No", "Yes"])
    business_travel = st.selectbox(
        "Business travel", ["Non-Travel", "Travel_Rarely", "Travel_Frequently"]
    )
    job_role = st.selectbox(
        "Job role",
        [
            "Sales Executive", "Research Scientist", "Laboratory Technician",
            "Manufacturing Director", "Healthcare Representative", "Manager",
            "Sales Representative", "Research Director", "Human Resources",
        ],
    )
    marital_status = st.selectbox("Marital status", ["Single", "Married", "Divorced"])
    distance = st.slider("Distance from home (km)", 1, 30, 8)

if st.button("Predict", type="primary"):
    form = {
        "Age": age,
        "MonthlyIncome": monthly_income,
        "YearsAtCompany": years_at_company,
        "JobSatisfaction": job_satisfaction,
        "WorkLifeBalance": work_life_balance,
        "DistanceFromHome": distance,
        "OverTime": overtime,
        "BusinessTravel": business_travel,
        "JobRole": job_role,
        "MaritalStatus": marital_status,
    }
    row = build_row(form, feature_columns, numeric_cols, scaler)
    prob = float(model.predict(row, verbose=0).ravel()[0])

    st.metric("Attrition probability", f"{prob:.0%}")
    st.progress(prob)
    if prob >= 0.5:
        st.error("High flight risk — worth a retention conversation.")
    elif prob >= 0.25:
        st.warning("Moderate risk — keep an eye on engagement.")
    else:
        st.success("Low risk.")
    st.caption(
        "Probabilities are model estimates, not certainties. Use as a "
        "prioritisation signal, not an HR decision on their own."
    )
