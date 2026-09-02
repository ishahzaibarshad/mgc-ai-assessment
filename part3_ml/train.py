from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
import joblib

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "data" / "leads.csv")

# Normalize obvious spelling/case noise in categorical values.
city_map = {
    "ISLAMABAD":"Islamabad", "ISB":"Islamabad",
    "RAWALPINDI":"Rawalpindi", "RWP":"Rawalpindi",
    "LAHORE":"Lahore", "KARACHI":"Karachi", "KHI":"Karachi",
    "PESHAWAR":"Peshawar", "FAISALABAD":"Faisalabad",
    "MULTAN":"Multan", "GUJRANWALA":"Gujranwala"
}
df["city"] = df["city"].str.strip().str.upper().replace(city_map)

# Score leads using information that is plausibly available for prioritization
# before the sales team starts calling them.
features = [
    "source","city","area","property_type","budget_pkr_lac","bedrooms",
    "is_overseas","referred_by_existing_client","has_financing_approved"
]
X = df[features].copy()
y = df["converted"].astype(int)

cat = ["source","city","area","property_type"]
num = ["budget_pkr_lac","bedrooms","is_overseas","referred_by_existing_client","has_financing_approved"]

preprocess = ColumnTransformer([
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]), cat),
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler())
    ]), num)
])

model = Pipeline([
    ("preprocess", preprocess),
    ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced"))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
model.fit(X_train, y_train)
proba = model.predict_proba(X_test)[:,1]
ap = average_precision_score(y_test, proba)

print(f"Rows: {len(df)}")
print(f"Positive rate: {y.mean():.4f} ({y.mean()*100:.2f}%)")
print(f"Test rows: {len(y_test)}")
print(f"Average Precision (PR-AUC): {ap:.4f}")
print("Why this metric: conversion is only about 6.9%, so a precision-recall metric is more informative for ranking likely converters than accuracy.")

out = ROOT / "part3_ml" / "lead_scoring_model.joblib"
joblib.dump(model, out)
print(f"Saved model to {out}")
