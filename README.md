# Employee Attrition Predictor

A predictive (non-generative) Deep Learning MVP for the IE University Deep Learning
final project. A feed-forward **ANN** estimates the probability that an employee is
about to leave, exposed through a **Streamlit** frontend an HR user can operate
directly.

**Business case.** Replacing an employee costs roughly 50–200% of their annual
salary. A model that flags flight risk early lets HR intervene before the resignation
letter, turning a reactive, expensive process into a proactive, cheap one.

**Architecture choice.** This is a tabular binary-classification problem
(structured HR records → leave / stay), which is the canonical use case for an
**Artificial Neural Network** — the architecture the brief assigns to tabular data.

---

## Pipeline (run top to bottom)

| Step | File | What it does | Key decision |
|------|------|--------------|--------------|
| 1 | `src/preprocess.py` | Load, drop dead columns, one-hot encode, stratified split, scale | Scaler fit on **train only** to avoid leakage; one-hot before split is leakage-free |
| 2 | `src/model.py` | Define the ANN (64 → 32 → 1, dropout) | Small net + dropout because the dataset is tiny (~1,470 rows) |
| 3 | `src/train.py` | Train with class weights + early stopping | `class_weight` over SMOTE for honesty; early stopping = overfitting evidence |
| 4 | `src/evaluate.py` | ROC-AUC, F1, confusion matrix, chosen threshold | ROC-AUC survives imbalance; threshold picked from data, not defaulted to 0.5 |
| 5 | `frontend/app.py` | Streamlit form → live probability | Reuses the exact saved scaler + column order for a faithful single-row prediction |

---

## Dataset

IBM HR Analytics Employee Attrition (Kaggle). Download
`WA_Fn-UseC_-HR-Employee-Attrition.csv`, rename it to `attrition.csv`, and place it
in `data/raw/`. ~1,470 rows, 35 columns, no missing values.

> Note: attrition is ~16% positive. This imbalance is handled explicitly with class
> weighting and by reporting ROC-AUC/F1 rather than raw accuracy (which would hit
> ~84% by predicting "nobody leaves" and learning nothing).

---

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

```bash
# 1-4: build features, train, and evaluate
python -m src.train        # preprocesses + trains, saves model + scaler + columns
python -m src.evaluate     # prints ROC-AUC, F1, confusion matrix

# 5: launch the demo (run from the project root)
streamlit run frontend/app.py
```

## Roadmap

- [x] Step 1 — Preprocessing pipeline (leakage-safe, stratified, scaled)
- [x] Step 2 — ANN architecture with dropout
- [x] Step 3 — Training with class weights + early stopping
- [x] Step 4 — Evaluation (ROC-AUC, F1, threshold selection)
- [x] Step 5 — Streamlit frontend with live prediction
- [ ] Stretch — SHAP feature importance for "why this employee is flagged"
- [ ] Stretch — expand the form to all input features
# deep_learning_group_project
