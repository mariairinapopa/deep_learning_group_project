# Employee Attrition Predictor

*An early-warning system for employee retention — built with a feed-forward neural network and served through a Streamlit app a non-technical HR user can run.*

![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow%2FKeras-ANN-FF6F00?logo=tensorflow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-frontend-FF4B4B?logo=streamlit&logoColor=white)
![Task](https://img.shields.io/badge/task-binary%20classification-1E2761)

This project predicts the probability that an individual employee is about to leave, early
enough for HR to act. It is a **predictive (non-generative) deep-learning** MVP: structured HR
records go in, a single risk score comes out, and the whole thing runs end to end from two
commands.

---

## The problem it solves

By the time an employee hands in their notice, the decision is already made. HR teams spend
their energy *reacting* to resignations — running exit interviews, scrambling to backfill —
instead of *preventing* them, because they have no early signal to work from.

That gap is expensive. Replacing one employee typically costs **50–200% of their annual salary**
once recruiting, onboarding, lost institutional knowledge, and the team disruption that triggers
the *next* departure are all counted. A tool that surfaces flight risk months ahead of the
resignation lets a manager have a retention conversation while it still changes the outcome —
turning an expensive, reactive process into a cheap, proactive one.

The value proposition is concrete: point a limited retention budget at the people who are
actually at risk, catch a meaningful share of leavers before they go, and pay only the cost of a
short check-in for the occasional false alarm — against a saved exit worth up to twice a salary.

---

## What it does

An HR manager opens the app, enters an employee profile (age, income, overtime, role, tenure,
satisfaction, and so on), and gets back a probability and a colour-coded risk band in real time.
No notebook, no analyst, no waiting — the model is the backend, the form is the interface, and
the prediction is live.

---

## How it works

One reproducible pipeline, five stages, no manual steps and no hidden state:

```
raw HR data  →  preprocess  →  train ANN  →  evaluate  →  Streamlit app
```

Anyone can clone the repo, drop in the dataset, and reproduce the exact results below with
`python -m src.train` followed by `python -m src.evaluate`.

### Data & preprocessing

The dataset is **IBM HR Analytics** (1,470 employees, 35 features, no missing values). The
preprocessing step is built to be **leakage-safe**, because leakage is the classic way to earn a
great score that collapses in production:

- **Feature handling** — four columns that are constant (`EmployeeCount`, `StandardHours`,
  `Over18`) or pure identifiers (`EmployeeNumber`) are dropped; they carry no signal.
- **Encoding** — categoricals are one-hot encoded *before* the split. This only fixes which
  columns exist; it never touches the target, so it is leakage-free and avoids mismatched dummy
  columns between train and test.
- **Scaling** — the scaler is fit on the **training split only**, then applied blind to test.
  Fitting it on test rows would leak their distribution into the model.
- **Stratified split** — the ~16% leaver rate is held identical in train and test, so the
  evaluation is honest rather than a lucky draw.

### Model architecture

A deliberately **small** feed-forward network: `Input → Dense(64) → Dropout → Dense(32) →
Dropout → Dense(1, sigmoid)`.

| Choice | Value | Why |
|---|---|---|
| Network size | 64 → 32 → 1 | With only ~1,470 rows, a large net memorises instantly; this keeps parameters well below the row count. |
| Dropout | 0.3 after each hidden layer | Primary overfitting defence — forces redundancy instead of memorisation. |
| Output | 1 sigmoid neuron | Output is literally *P(employee leaves)* — exactly what the business needs. |
| Optimizer / loss | Adam (1e-3) / binary cross-entropy | Standard, correct pairing for single-probability binary classification. |
| Imbalance | class weights (not SMOTE) | Re-weights *real* employees rather than inventing synthetic ones — simpler to defend, nothing fabricated. |
| Stopping | early stopping on val loss, restore best weights | The saved model is the best-generalising epoch, not the over-trained final one — concrete evidence of overfitting prevention. |

---

## Results

Evaluated on a held-out, stratified test set (294 employees). Because only ~16% of employees
leave, **accuracy is the wrong yardstick** — a model that predicts "nobody quits" would score
~84% and be useless — so the model is judged on metrics that survive imbalance.

| Metric | Value | How to read it |
|---|---|---|
| **ROC-AUC** | **0.708** | Above the 0.5 coin-flip line — the model has learned real signal. |
| Recall (leavers) | 0.60 | Catches 60% of employees who actually leave. |
| Precision (leavers) | 0.31 | About 1 in 3 flagged employees is a true leaver. |
| F1 (leavers) | 0.41 | At the chosen threshold. |
| Decision threshold | 0.28 | Selected **from the data** to maximise F1 — not defaulted to 0.5. |

**Confusion matrix (test set):**

|  | Predicted: Stay | Predicted: Leave |
|---|---|---|
| **Actual: Stay** | 186 | 61 |
| **Actual: Leave** | 19 | 28 |

**Reading it as a business decision.** The model is intentionally tuned for **recall over
precision**, and the threshold is set from validation data rather than the default 0.5. The
reasoning is an asymmetry: a false alarm costs a manager a 15-minute check-in, while a missed
leaver can cost up to twice that person's salary. At that ratio, catching 60% of real leavers is
worth accepting more false positives. The model is a **triage signal that prioritises who to talk
to first**, not an automated verdict — which is also why the threshold is exposed as a deliberate
lever rather than buried.

> This dataset is known to be hard for neural networks specifically — tree-based models often
> edge them out here — so an ANN landing around 0.70 ROC-AUC with sound methodology is a fair,
> defensible result for a structured-data MVP.

---

## The app

The Streamlit frontend is wired directly to the trained backend:

1. The form collects a handful of human-readable inputs.
2. On submit, the **exact saved scaler and feature-column order** from training rebuild that
   single employee into the model's input space — so the live prediction is faithful, not an
   approximation.
3. `model.predict()` runs and returns a probability, rendered as a percentage plus a risk band.

That faithful single-row transform is what makes the end-to-end integration robust: the demo
uses the same preprocessing as training, with no hardcoding.

```bash
streamlit run frontend/app.py
```

---

## Run it yourself

**1. Get the data.** Download `WA_Fn-UseC_-HR-Employee-Attrition.csv` from the Kaggle *IBM HR
Analytics Employee Attrition* dataset, rename it to `attrition.csv`, and place it in
`data/raw/`.

**2. Set up the environment.**

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**3. Train, evaluate, demo.**

```bash
python -m src.train        # preprocess + train; saves model, scaler, feature columns
python -m src.evaluate     # ROC-AUC, F1, confusion matrix, chosen threshold
streamlit run frontend/app.py   # launch the live MVP (run from the project root)
```

---

## Project layout

```
deeplearning-hr-attrition/
├── data/
│   ├── raw/                 # attrition.csv (from Kaggle — not committed)
│   └── processed/
├── models/                  # trained model.keras, scaler.joblib, feature_columns.json
├── src/
│   ├── __init__.py
│   ├── preprocess.py        # leakage-safe load → encode → split → scale
│   ├── model.py             # the ANN definition
│   ├── train.py             # class weights + early stopping
│   └── evaluate.py          # ROC-AUC, F1, threshold selection
├── frontend/
│   └── app.py               # Streamlit interface
├── requirements.txt
└── README.md
```

---

## Limitations & roadmap

Stated plainly, because a model is only trustworthy if its limits are known:

- It is trained on **one company's snapshot**; the patterns may not transfer to other firms or a
  different year without retraining.
- It captures **correlation, not causation** — it flags who is at risk, not why.
- A score is a **prompt for a human conversation, never an automatic HR decision.**

Planned next steps:

- [ ] **SHAP explanations** — show *why* each employee is flagged, turning a score into an action.
- [ ] **Retrain on a client's own history** for sharper, fairer signals.
- [ ] **Fairness audit** to confirm the model doesn't proxy for protected attributes before any real use.
- [ ] Expand the input form to the full feature set.

---

*Final presentation deck (PDF) is included with the submission.*