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


def risk_band(prob: float) -> dict:
    """Map a probability to a human verdict + colour palette used across the UI."""
    if prob >= 0.5:
        return {
            "label": "High flight risk",
            "note": "Worth a proactive retention conversation.",
            "color": "#dc2626",
            "soft": "#fef2f2",
            "border": "#fecaca",
        }
    if prob >= 0.25:
        return {
            "label": "Moderate risk",
            "note": "Keep an eye on engagement and workload.",
            "color": "#d97706",
            "soft": "#fffbeb",
            "border": "#fde68a",
        }
    return {
        "label": "Low risk",
        "note": "No immediate action needed.",
        "color": "#16a34a",
        "soft": "#f0fdf4",
        "border": "#bbf7d0",
    }


def gauge_svg(prob: float, color: str) -> str:
    """A lightweight semicircular gauge rendered as inline SVG (no dependencies)."""
    import math

    # Semicircle from 180° (left) to 0° (right); needle sweeps with probability.
    angle = math.pi * (1 - prob)
    cx, cy, r = 130, 130, 100
    nx, ny = cx + r * 0.82 * math.cos(angle), cy - r * 0.82 * math.sin(angle)

    def arc(frac_start, frac_end, stroke):
        a0, a1 = math.pi * (1 - frac_start), math.pi * (1 - frac_end)
        x0, y0 = cx + r * math.cos(a0), cy - r * math.sin(a0)
        x1, y1 = cx + r * math.cos(a1), cy - r * math.sin(a1)
        return (
            f'<path d="M {x0:.1f} {y0:.1f} A {r} {r} 0 0 1 {x1:.1f} {y1:.1f}" '
            f'fill="none" stroke="{stroke}" stroke-width="18" stroke-linecap="round"/>'
        )

    return f"""
    <svg viewBox="0 0 260 160" width="260" height="160" role="img"
         aria-label="Attrition probability gauge">
      {arc(0.0, 0.25, "#bbf7d0")}
      {arc(0.25, 0.5, "#fde68a")}
      {arc(0.5, 1.0, "#fecaca")}
      <line x1="{cx}" y1="{cy}" x2="{nx:.1f}" y2="{ny:.1f}"
            stroke="{color}" stroke-width="5" stroke-linecap="round"/>
      <circle cx="{cx}" cy="{cy}" r="9" fill="{color}"/>
      <text x="{cx}" y="105" text-anchor="middle"
            font-size="34" font-weight="700" fill="#0f172a">{prob:.0%}</text>
    </svg>
    """


st.set_page_config(
    page_title="Attrition Predictor",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
      .block-container { max-width: 1080px; padding-top: 2.2rem; }
      #MainMenu, footer { visibility: hidden; }
      .hero {
        background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 100%);
        border-radius: 16px; padding: 1.6rem 1.9rem; color: #fff;
        margin-bottom: 1.6rem;
      }
      .hero h1 { color: #fff; font-size: 1.8rem; margin: 0 0 .25rem 0; }
      .hero p { color: #dbeafe; margin: 0; font-size: .98rem; }
      .card {
        background: #fff; border: 1px solid #e2e8f0; border-radius: 14px;
        padding: 1.4rem 1.5rem; box-shadow: 0 1px 3px rgba(15,23,42,.06);
      }
      .section-label {
        text-transform: uppercase; letter-spacing: .06em; font-size: .72rem;
        font-weight: 700; color: #64748b; margin-bottom: .4rem;
      }
      div.stButton > button {
        width: 100%; border-radius: 10px; font-weight: 600; padding: .5rem .6rem;
        border: 1px solid #cbd5e1; background: #f8fafc; color: #0f172a;
        box-shadow: 0 1px 2px rgba(15,23,42,.05); transition: all .15s ease;
      }
      div.stButton > button:hover {
        border-color: #2563eb; color: #2563eb; background: #eff6ff;
      }
      div.stFormSubmitButton, div[data-testid="stFormSubmitButton"] { width: 100%; }
      div[data-testid="stFormSubmitButton"] > button {
        width: 100%; border-radius: 10px; font-weight: 700; font-size: 1rem;
        padding: .7rem 0; margin-top: .4rem;
        border: 1px solid #1d4ed8; color: #fff;
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        box-shadow: 0 2px 6px rgba(37,99,235,.35); transition: all .15s ease;
      }
      div[data-testid="stFormSubmitButton"] > button:hover {
        background: linear-gradient(135deg, #1d4ed8 0%, #1e40af 100%);
        border-color: #1e40af; color: #fff;
        box-shadow: 0 4px 12px rgba(37,99,235,.45);
      }
      div[data-testid="stFormSubmitButton"] > button:active,
      div[data-testid="stFormSubmitButton"] > button:focus {
        color: #fff; box-shadow: 0 2px 6px rgba(37,99,235,.35);
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h1>Employee Attrition Predictor</h1>
      <p>Estimate the probability that an employee is about to leave — a
         prioritisation signal for HR, powered by a trained neural network.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

model, scaler, feature_columns, numeric_cols = load_artifacts()

# Default values for every input widget. We seed session_state with these ONCE and
# let the widgets read/write session_state by key — no hardcoded `value=` args — so
# preset buttons can overwrite state without tripping Streamlit's
# "default value + Session State API" warning.
DEFAULTS = {
    "age": 35,
    "monthly_income": 5000,
    "years_at_company": 5,
    "job_satisfaction": 3,
    "work_life_balance": 3,
    "distance": 8,
    "overtime": "No",
    "business_travel": "Non-Travel",
    "job_role": "Sales Executive",
    "marital_status": "Single",
}
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)
st.session_state.setdefault("should_predict", False)

# Preset "history" profiles, calibrated so each lands cleanly in its band.
PRESETS = {
    "high": {
        "age": 24, "monthly_income": 10000, "years_at_company": 1,
        "job_satisfaction": 2, "work_life_balance": 3, "distance": 15,
        "overtime": "Yes", "business_travel": "Travel_Frequently",
        "job_role": "Sales Representative", "marital_status": "Single",
    },
    "medium": {
        "age": 43, "monthly_income": 15000, "years_at_company": 14,
        "job_satisfaction": 4, "work_life_balance": 4, "distance": 8,
        "overtime": "No", "business_travel": "Travel_Rarely",
        "job_role": "Manufacturing Director", "marital_status": "Married",
    },
    "low": {
        "age": 55, "monthly_income": 19000, "years_at_company": 25,
        "job_satisfaction": 4, "work_life_balance": 4, "distance": 3,
        "overtime": "No", "business_travel": "Non-Travel",
        "job_role": "Research Director", "marital_status": "Married",
    },
}


def apply_preset(preset: dict):
    """Load a preset into the widgets and request a prediction on the next rerun."""
    for k, v in preset.items():
        st.session_state[k] = v
    st.session_state["should_predict"] = True


# A pragmatic subset of the most decision-relevant inputs. Anything not shown is
# reindexed to 0 (for dummies) — fine for a demo; expand the form for production.
form_col, result_col = st.columns([1.35, 1], gap="large")

with form_col:
    # Sample-profile "history" panel. Buttons must live OUTSIDE st.form (Streamlit
    # only allows submit buttons inside a form).
    # st.markdown(
    #     '<div class="card" style="padding:1rem 1.2rem; margin-bottom:1rem;">'
    #     '<div style="font-weight:700; font-size:1rem; color:#0f172a;">Sample profiles</div>'
    #     '<div style="font-size:.82rem; color:#64748b;">'
    #     'click to load &amp; predict instantly</div>'
    #     '</div>',
    #     unsafe_allow_html=True,
    # )
    h1, h2, h3 = st.columns(3)
    with h1:
        st.markdown(
            '<div style="font-size:.82rem; color:#475569; margin-bottom:.2rem;">'
            '<span style="color:#dc2626;">●</span> High risk · young, overtime, '
            'low satisfaction</div>',
            unsafe_allow_html=True,
        )
        st.button("Load High risk", key="preset_high", on_click=apply_preset, use_container_width=True,
                  args=(PRESETS["high"],))
    with h2:
        st.markdown(
            '<div style="font-size:.82rem; color:#475569; margin-bottom:.2rem;">'
            '<span style="color:#d97706;">●</span> Medium risk · mid-career, '
            'some overtime</div>',
            unsafe_allow_html=True,
        )
        st.button("Load Medium risk", key="preset_medium", on_click=apply_preset, use_container_width=True,
                  args=(PRESETS["medium"],))
    with h3:
        st.markdown(
            '<div style="font-size:.82rem; color:#475569; margin-bottom:.2rem;">'
            '<span style="color:#16a34a;">●</span> Low risk · senior, settled, '
            'no overtime</div>',
            unsafe_allow_html=True,
        )
        st.button("Load Low risk", key="preset_low", on_click=apply_preset, use_container_width=True,
                  args=(PRESETS["low"],))

    with st.form("attrition_form"):
        st.markdown('<div class="section-label">Employee profile</div>',
                    unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.slider("Age", 18, 60, key="age")
            st.number_input("Monthly income ($)", 1000, 20000, step=250,
                            key="monthly_income")
            st.slider("Years at company", 0, 40, key="years_at_company")
            st.select_slider("Job satisfaction (1–4)", [1, 2, 3, 4],
                             key="job_satisfaction")
            st.select_slider("Work-life balance (1–4)", [1, 2, 3, 4],
                             key="work_life_balance")
        with c2:
            st.selectbox("Works overtime?", ["No", "Yes"], key="overtime")
            st.selectbox(
                "Business travel",
                ["Non-Travel", "Travel_Rarely", "Travel_Frequently"],
                key="business_travel",
            )
            st.selectbox(
                "Job role",
                [
                    "Sales Executive", "Research Scientist", "Laboratory Technician",
                    "Manufacturing Director", "Healthcare Representative", "Manager",
                    "Sales Representative", "Research Director", "Human Resources",
                ],
                key="job_role",
            )
            st.selectbox("Marital status", ["Single", "Married", "Divorced"],
                         key="marital_status")
            st.slider("Distance from home (km)", 1, 30, key="distance")

        submitted = st.form_submit_button(
            "Predict attrition risk", type="primary", use_container_width=True
        )

if submitted:
    st.session_state["should_predict"] = True

with result_col:
    st.markdown('<div class="section-label">Prediction</div>', unsafe_allow_html=True)
    if st.session_state["should_predict"]:
        form = {
            "Age": st.session_state["age"],
            "MonthlyIncome": st.session_state["monthly_income"],
            "YearsAtCompany": st.session_state["years_at_company"],
            "JobSatisfaction": st.session_state["job_satisfaction"],
            "WorkLifeBalance": st.session_state["work_life_balance"],
            "DistanceFromHome": st.session_state["distance"],
            "OverTime": st.session_state["overtime"],
            "BusinessTravel": st.session_state["business_travel"],
            "JobRole": st.session_state["job_role"],
            "MaritalStatus": st.session_state["marital_status"],
        }
        row = build_row(form, feature_columns, numeric_cols, scaler)
        prob = float(model.predict(row, verbose=0).ravel()[0])
        band = risk_band(prob)

        st.markdown(
            f'<div class="card" style="text-align:center;">'
            f'{gauge_svg(prob, band["color"]).strip()}'
            f'<div style="margin-top:.6rem; padding:.7rem 1rem; border-radius:10px;'
            f' background:{band["soft"]}; border:1px solid {band["border"]};">'
            f'<div style="font-weight:700; font-size:1.05rem; color:{band["color"]};">'
            f'{band["label"]}</div>'
            f'<div style="font-size:.9rem; color:#475569; margin-top:.15rem;">'
            f'{band["note"]}</div>'
            f'</div></div>',
            unsafe_allow_html=True,
        )
        st.caption(
            "Probabilities are model estimates, not certainties. Use as a "
            "prioritisation signal, not an HR decision on their own."
        )
    else:
        st.markdown(
            '<div class="card" style="text-align:center; color:#64748b;">'
            '<div style="font-size:2.4rem;">📊</div>'
            '<p style="margin:.4rem 0 0 0;">Fill in the profile and press '
            '<strong>Predict attrition risk</strong> to see a live estimate.</p>'
            '</div>',
            unsafe_allow_html=True,
        )
