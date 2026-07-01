"""
Step 1 — Preprocess: turn the raw IBM HR CSV into model-ready arrays.

Design choices (and why):
- Drop zero-variance / ID columns. EmployeeCount, StandardHours and Over18 are
  constant across every row in this dataset, and EmployeeNumber is an identifier.
  None of them carry signal; keeping them just adds parameters and noise.
- One-hot encode the string categoricals BEFORE the train/test split. One-hot
  encoding only fixes the *column space* (which categories exist) — it never
  looks at the target — so doing it on the full frame is leakage-free and avoids
  the classic "train and test ended up with different dummy columns" bug.
- Fit the StandardScaler on the TRAIN split ONLY, then transform test. Scaling
  uses the mean/std of the data, which IS information — fitting it on test rows
  would leak. This mirrors the "validate the threshold against real data" care
  from the RAG project: the boundary is set from train, then applied blind.
- Stratified split. Attrition is ~16% positive, so a random split could hand you
  a test set with a very different positive rate. Stratify keeps the ratio fixed.
- Persist the scaler, the full feature-column order, and the list of numeric
  columns. The Streamlit app must reconstruct an identical feature vector for a
  single new employee, so it needs the exact same column order and the exact
  same scaler. Save them now, load them at inference time.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Resolve paths relative to the project root so the pipeline runs from anywhere.
ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "attrition.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"

TARGET = "Attrition"
DROP_COLS = ["EmployeeCount", "StandardHours", "Over18", "EmployeeNumber"]

# The seven free-text categoricals in the IBM HR dataset. Everything else is
# already numeric (continuous, or integer-coded ordinals like JobSatisfaction).
CATEGORICAL_COLS = [
    "BusinessTravel",
    "Department",
    "EducationField",
    "Gender",
    "JobRole",
    "MaritalStatus",
    "OverTime",
]

RANDOM_STATE = 42


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path}.\n"
            "Download 'WA_Fn-UseC_-HR-Employee-Attrition.csv' from the Kaggle "
            "'IBM HR Analytics Employee Attrition' dataset, rename it to "
            "attrition.csv, and place it in data/raw/."
        )
    return pd.read_csv(path)


def build_features(df: pd.DataFrame):
    """Return (X, y) where X is fully numeric and y is 0/1."""
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

    # Target: "Yes"/"No" -> 1/0
    y = (df[TARGET].astype(str).str.strip() == "Yes").astype(int).values
    X = df.drop(columns=[TARGET])

    # Remember which columns are numeric BEFORE we add dummy columns, so we only
    # scale the genuinely continuous/ordinal features and leave one-hot flags 0/1.
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    # One-hot encode the string categoricals. drop_first=True removes one
    # redundant column per category (k categories -> k-1 columns) to avoid the
    # dummy-variable trap; it doesn't lose information.
    X = pd.get_dummies(X, columns=CATEGORICAL_COLS, drop_first=True)

    # get_dummies can yield bool columns on newer pandas; cast to int for Keras.
    X = X.astype({c: "int8" for c in X.columns if X[c].dtype == "bool"})

    return X, y, numeric_cols


def prepare(test_size: float = 0.2, save: bool = True):
    """Full Step 1: load -> features -> stratified split -> scale -> persist."""
    df = load_raw()
    X, y, numeric_cols = build_features(df)

    feature_columns = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X.values,
        y,
        test_size=test_size,
        stratify=y,                 # keep the ~16% positive rate in both splits
        random_state=RANDOM_STATE,
    )

    # Scale ONLY the numeric columns; fit on train, apply to both.
    numeric_idx = [feature_columns.index(c) for c in numeric_cols]
    scaler = StandardScaler()
    X_train[:, numeric_idx] = scaler.fit_transform(X_train[:, numeric_idx])
    X_test[:, numeric_idx] = scaler.transform(X_test[:, numeric_idx])

    if save:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, MODELS_DIR / "scaler.joblib")
        with open(MODELS_DIR / "feature_columns.json", "w") as f:
            json.dump(
                {"feature_columns": feature_columns, "numeric_cols": numeric_cols},
                f,
                indent=2,
            )
        np.savez(
            PROCESSED_DIR / "splits.npz",
            X_train=X_train.astype("float32"),
            X_test=X_test.astype("float32"),
            y_train=y_train,
            y_test=y_test,
        )

    return (
        X_train.astype("float32"),
        X_test.astype("float32"),
        y_train,
        y_test,
        feature_columns,
        numeric_cols,
    )


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, cols, _ = prepare()
    pos_rate = y_train.mean()
    print(f"Features: {len(cols)} | train: {X_train.shape} | test: {X_test.shape}")
    print(f"Positive (attrition) rate in train: {pos_rate:.1%}")
