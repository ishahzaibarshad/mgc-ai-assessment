from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "leads.csv"
MODEL_PATH = ROOT / "part3_ml" / "lead_scoring_model.joblib"

FEATURES = [
    "source",
    "city",
    "area",
    "property_type",
    "budget_pkr_lac",
    "bedrooms",
    "is_overseas",
    "referred_by_existing_client",
    "has_financing_approved",
]
TARGET = "converted"
CITY_MAP = {
    "ISLAMABAD": "Islamabad",
    "ISB": "Islamabad",
    "RAWALPINDI": "Rawalpindi",
    "RWP": "Rawalpindi",
    "LAHORE": "Lahore",
    "KARACHI": "Karachi",
    "KHI": "Karachi",
    "PESHAWAR": "Peshawar",
    "FAISALABAD": "Faisalabad",
    "MULTAN": "Multan",
    "GUJRANWALA": "Gujranwala",
}
CATEGORICAL_FEATURES = ["source", "city", "area", "property_type"]
NUMERIC_FEATURES = [
    "budget_pkr_lac",
    "bedrooms",
    "is_overseas",
    "referred_by_existing_client",
    "has_financing_approved",
]


def build_pipeline():
    preprocess = ColumnTransformer(
        [
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                NUMERIC_FEATURES,
            ),
        ]
    )
    return Pipeline(
        [
            ("preprocess", preprocess),
            ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )


def train_and_evaluate(data=None, *, data_path=DATA_PATH, model_path=None):
    """Fit the leakage-aware baseline and return measured holdout results."""
    df = pd.read_csv(data_path) if data is None else data.copy()
    required = set(FEATURES + [TARGET])
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Training data is missing required columns: {', '.join(sorted(missing))}")

    df["city"] = df["city"].astype("string").str.strip().str.upper().replace(CITY_MAP)
    X = df[FEATURES].copy()
    y = pd.to_numeric(df[TARGET], errors="raise").astype(int)
    if not set(y.unique()).issubset({0, 1}):
        raise ValueError("The target column must contain only 0 and 1 values.")
    if y.nunique() != 2 or y.value_counts().min() < 2:
        raise ValueError("Training requires at least two examples from each target class.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    model = build_pipeline()
    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)[:, 1]
    average_precision = float(average_precision_score(y_test, probabilities))

    if model_path is not None:
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path)

    return {
        "model": model,
        "rows": len(df),
        "positive_rate": float(y.mean()),
        "test_rows": len(y_test),
        "average_precision": average_precision,
        "model_path": Path(model_path) if model_path is not None else None,
    }


def main():
    result = train_and_evaluate(model_path=MODEL_PATH)
    print(f"Rows: {result['rows']}")
    print(f"Positive rate: {result['positive_rate']:.4f} ({result['positive_rate'] * 100:.2f}%)")
    print(f"Test rows: {result['test_rows']}")
    print(f"Average Precision (PR-AUC): {result['average_precision']:.4f}")
    print("Why this metric: conversion is imbalanced, so a precision-recall metric is more informative for ranking likely converters than accuracy.")
    print(f"Saved model to {result['model_path']}")


if __name__ == "__main__":
    main()

