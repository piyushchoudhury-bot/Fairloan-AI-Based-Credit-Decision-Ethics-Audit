"""
train_model.py
---------------
Trains the main FairLoan RandomForest classifier used by the Loan
Prediction page and most experiment pages, and saves it (with metadata)
using joblib.

IMPORTANT: The model is trained WITHOUT the sensitive attribute (gender) as
an input feature. Gender is retained separately, purely for the post-hoc
fairness auditing pages.

Run:
    python train_model.py
"""

import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

from utils.data_utils import load_dataset, MODEL_FEATURES, TARGET_COLUMN

SEED = 42
MODEL_DIR = "model"


def main():
    print("Loading dataset...")
    df = load_dataset(dropna=True)

    import pandas.api.types as ptypes
    categorical_features = [c for c in MODEL_FEATURES if not ptypes.is_numeric_dtype(df[c])]
    numeric_features = [c for c in MODEL_FEATURES if c not in categorical_features]

    X = df[MODEL_FEATURES]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=SEED, stratify=y
    )

    preprocessor = ColumnTransformer(transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ], remainder="passthrough")

    pipeline = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=300, max_depth=12, random_state=SEED, n_jobs=-1
        )),
    ])

    print("Training RandomForestClassifier...")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    print("Evaluation metrics:")
    print(json.dumps(metrics, indent=2))

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(pipeline, os.path.join(MODEL_DIR, "loan_model.pkl"))

    metadata = {
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "model_features": MODEL_FEATURES,
        "target": TARGET_COLUMN,
        "metrics": metrics,
        "seed": SEED,
        "test_indices": X_test.index.tolist(),  # so the app can reuse the same held-out test set
    }
    joblib.dump(metadata, os.path.join(MODEL_DIR, "metadata.pkl"))

    print(f"\nModel saved to {MODEL_DIR}/loan_model.pkl")
    print(f"Metadata saved to {MODEL_DIR}/metadata.pkl")


if __name__ == "__main__":
    main()
