"""HeartWise: interactive, explainable heart-disease ML exploration app.

This is an educational decision-support demo, not a medical device.
"""
from __future__ import annotations

import io
import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore", category=UserWarning)

TARGET = "HeartDisease"
EXPECTED_COLUMNS = {
    "Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol", "FastingBS",
    "RestingECG", "MaxHR", "ExerciseAngina", "Oldpeak", "ST_Slope", TARGET
}
NUMERIC_HINTS = ["Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR", "Oldpeak"]


def normalize_target(series: pd.Series) -> pd.Series:
    """Normalize common binary target representations to 0/1."""
    if pd.api.types.is_numeric_dtype(series):
        vals = set(pd.Series(series.dropna().unique()).astype(float).tolist())
        if vals <= {0.0, 1.0}:
            return series.astype(int)
    mapped = series.astype(str).str.strip().str.lower().map({
        "0": 0, "1": 1, "no": 0, "yes": 1, "false": 0, "true": 1,
        "absence": 0, "presence": 1, "negative": 0, "positive": 1,
        "normal": 0, "heart disease": 1, "no heart disease": 0
    })
    if mapped.isna().any():
        raise ValueError("Target column must contain recognizable binary values (0/1 or yes/no).")
    return mapped.astype(int)


def prepare_dataset(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Validate and lightly clean data; imputation is fitted inside each model pipeline."""
    df = raw.copy()
    df.columns = [str(c).strip() for c in df.columns]
    if TARGET not in df.columns:
        raise ValueError(f"Missing required target column '{TARGET}'.")
    if df.empty:
        raise ValueError("The uploaded CSV has no rows.")
    df = df.drop_duplicates().reset_index(drop=True)
    y = normalize_target(df.pop(TARGET))
    if y.nunique() != 2:
        raise ValueError("The target must contain both classes (0 and 1).")
    # Common sentinel values in this dataset are known to be invalid; turn them into
    # missing values instead of imputing globally before the train/test split.
    for col in ("Cholesterol", "RestingBP"):
        if col in df.columns:
            numeric = pd.to_numeric(df[col], errors="coerce")
            df[col] = numeric.mask(numeric <= 0, np.nan)
    for col in NUMERIC_HINTS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # Remove columns with no information; avoid leaking target or row identifiers.
    df = df.drop(columns=[c for c in df.columns if df[c].nunique(dropna=True) <= 1], errors="ignore")
    if df.shape[1] == 0:
        raise ValueError("No usable feature columns remain after cleaning.")
    return df, y, list(df.columns)


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric = X.select_dtypes(include=["number"]).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler())
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore"))
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipe, numeric),
        ("categorical", categorical_pipe, categorical)
    ], remainder="drop", sparse_threshold=0.0)


def train_models(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, seed: int = 42):
    if len(X) < 20:
        raise ValueError("Upload at least 20 usable rows for a meaningful train/test demonstration.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=1500, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=250, min_samples_leaf=2,
                                                class_weight="balanced", random_state=seed, n_jobs=1),
        "Gradient Boosting": HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08,
                                                             l2_regularization=1.0, random_state=seed)
    }
    fitted, rows = {}, []
    for name, estimator in candidates.items():
        pipe = Pipeline([("preprocess", make_preprocessor(X_train)), ("model", estimator)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        prob = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe, "predict_proba") else None
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred, zero_division=0),
            "Recall": recall_score(y_test, pred, zero_division=0),
            "F1": f1_score(y_test, pred, zero_division=0),
            "ROC-AUC": roc_auc_score(y_test, prob) if prob is not None and y_test.nunique() == 2 else np.nan,
        })
        fitted[name] = pipe
    metrics = pd.DataFrame(rows).sort_values(["F1", "ROC-AUC"], ascending=False, na_position="last").reset_index(drop=True)
    return fitted, metrics, X_test, y_test


def make_patient_row(X: pd.DataFrame, values: dict[str, Any]) -> pd.DataFrame:
    """Align a user-entered patient record to the exact training feature schema."""
    row = {}
    for col in X.columns:
        value = values.get(col, None)
        if value is None:
            # A neutral fallback is explicitly disclosed in the UI.
            value = X[col].median() if pd.api.types.is_numeric_dtype(X[col]) else X[col].mode(dropna=True).iloc[0]
        row[col] = value
    return pd.DataFrame([row], columns=X.columns)


