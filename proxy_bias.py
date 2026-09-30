"""
proxy_bias.py
-------------
Implements the "Cost-as-a-Proxy Resource Bias" experiment.

Trains two models:
  MODEL A - uses all normal features, INCLUDING resource/cost-related
            features (monthly_spending, account_balance, transaction_count).
  MODEL B - the SAME pipeline, but with resource-related features removed.

Then compares accuracy/fairness metrics between the two, to let students see
the real trade-off between predictive performance and measured fairness when
proxy-prone features are removed.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from utils.data_utils import MODEL_FEATURES, RESOURCE_PROXY_FEATURES, TARGET_COLUMN
from utils.fairness_utils import compute_group_metrics, compute_fairness_disparities

SEED = 42


def _build_pipeline(numeric_features, categorical_features):
    preprocessor = ColumnTransformer(transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ], remainder="passthrough")

    pipeline = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=200, max_depth=10, random_state=SEED, n_jobs=-1
        )),
    ])
    return pipeline


def _train_and_eval(df, features, sensitive_col="gender"):
    import pandas.api.types as ptypes
    categorical = [c for c in features if not ptypes.is_numeric_dtype(df[c])]
    numeric = [c for c in features if c not in categorical]

    X = df[features]
    y = df[TARGET_COLUMN]
    sensitive = df[sensitive_col]

    X_train, X_test, y_train, y_test, sens_train, sens_test = train_test_split(
        X, y, sensitive, test_size=0.25, random_state=SEED, stratify=y
    )

    pipeline = _build_pipeline(numeric, categorical)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    perf = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
    }

    group_metrics = compute_group_metrics(y_test, y_pred, sens_test)
    disparities = compute_fairness_disparities(y_test, y_pred, sens_test)

    return {
        "performance": perf,
        "group_metrics": group_metrics,
        "disparities": disparities,
    }


def run_proxy_bias_experiment(df):
    """Runs the full Model A vs Model B comparison. Returns a dict ready to
    be rendered in the template. All numbers are computed live -- nothing
    is hardcoded."""
    df = df.dropna(subset=MODEL_FEATURES + [TARGET_COLUMN, "gender"]).copy()

    features_with_resource = list(MODEL_FEATURES)
    features_without_resource = [f for f in MODEL_FEATURES if f not in RESOURCE_PROXY_FEATURES]

    result_a = _train_and_eval(df, features_with_resource)
    result_b = _train_and_eval(df, features_without_resource)

    comparison_table = []
    for metric in ["accuracy", "precision", "recall", "f1"]:
        comparison_table.append({
            "metric": metric.capitalize(),
            "with_resource": result_a["performance"][metric],
            "without_resource": result_b["performance"][metric],
        })

    comparison_table.append({
        "metric": "Demographic Parity Difference",
        "with_resource": result_a["disparities"]["demographic_parity_difference"],
        "without_resource": result_b["disparities"]["demographic_parity_difference"],
    })
    comparison_table.append({
        "metric": "Equal Opportunity Difference",
        "with_resource": result_a["disparities"]["equal_opportunity_difference"],
        "without_resource": result_b["disparities"]["equal_opportunity_difference"],
    })

    # group approval (selection) rates per model
    group_approval = {
        "with_resource": {
            g: round(v, 4) for g, v in result_a["group_metrics"]["by_group"]["selection_rate"].items()
        },
        "without_resource": {
            g: round(v, 4) for g, v in result_b["group_metrics"]["by_group"]["selection_rate"].items()
        },
    }

    fpr_fnr = {
        "with_resource": {
            "fpr": result_a["group_metrics"]["by_group"]["false_positive_rate"],
            "fnr": result_a["group_metrics"]["by_group"]["false_negative_rate"],
        },
        "without_resource": {
            "fpr": result_b["group_metrics"]["by_group"]["false_positive_rate"],
            "fnr": result_b["group_metrics"]["by_group"]["false_negative_rate"],
        },
    }

    # simple interpretation logic (transparent, rule based)
    acc_drop = result_a["performance"]["accuracy"] - result_b["performance"]["accuracy"]
    fairness_improvement = (
        abs(result_a["disparities"]["demographic_parity_difference"])
        - abs(result_b["disparities"]["demographic_parity_difference"])
    )

    if fairness_improvement > 0.01 and acc_drop > 0:
        interpretation = (
            f"Removing resource-related features (monthly spending, account balance, "
            f"transaction count) reduced accuracy by {acc_drop*100:.2f} percentage points, "
            f"but reduced the demographic parity gap by {fairness_improvement*100:.2f} "
            f"percentage points. This suggests the resource-related features were partly "
            f"acting as a proxy for group membership, contributing to disparate outcomes."
        )
    elif fairness_improvement > 0.01:
        interpretation = (
            "Removing resource-related features improved measured fairness with little or "
            "no loss in predictive performance, suggesting these features carried proxy "
            "information without adding much unique predictive value."
        )
    else:
        interpretation = (
            "In this run, removing resource-related features did not meaningfully change "
            "the fairness disparity, suggesting these particular features are not acting as "
            "a strong proxy for the sensitive attribute in this dataset."
        )

    return {
        "comparison_table": comparison_table,
        "group_approval": group_approval,
        "fpr_fnr": fpr_fnr,
        "interpretation": interpretation,
        "resource_features": RESOURCE_PROXY_FEATURES,
    }
