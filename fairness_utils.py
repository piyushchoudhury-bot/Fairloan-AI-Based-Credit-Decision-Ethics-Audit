"""
fairness_utils.py
------------------
Wraps Fairlearn metrics to compute group-wise performance and fairness
disparity measures for the Fairness Dashboard and Cost-as-Proxy pages.
"""

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from fairlearn.metrics import (
    MetricFrame,
    selection_rate,
    true_positive_rate,
    false_positive_rate,
    false_negative_rate,
    demographic_parity_difference,
    demographic_parity_ratio,
    equalized_odds_difference,
)


def compute_group_metrics(y_true, y_pred, sensitive_features):
    """Return a dict of per-group metrics using Fairlearn's MetricFrame."""
    metrics = {
        "accuracy": accuracy_score,
        "precision": lambda yt, yp: precision_score(yt, yp, zero_division=0),
        "recall": lambda yt, yp: recall_score(yt, yp, zero_division=0),
        "f1": lambda yt, yp: f1_score(yt, yp, zero_division=0),
        "selection_rate": selection_rate,
        "true_positive_rate": true_positive_rate,
        "false_positive_rate": false_positive_rate,
        "false_negative_rate": false_negative_rate,
    }

    mf = MetricFrame(
        metrics=metrics,
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_features,
    )

    by_group = mf.by_group.round(4).to_dict()
    overall = {k: round(float(v), 4) for k, v in mf.overall.to_dict().items()}

    return {"overall": overall, "by_group": by_group}


def compute_fairness_disparities(y_true, y_pred, sensitive_features):
    """Compute headline fairness disparity metrics."""
    dpd = demographic_parity_difference(y_true, y_pred, sensitive_features=sensitive_features)
    try:
        dpr = demographic_parity_ratio(y_true, y_pred, sensitive_features=sensitive_features)
    except Exception:
        dpr = None
    eod = equalized_odds_difference(y_true, y_pred, sensitive_features=sensitive_features)

    # Equal opportunity difference = difference in TPR between groups
    mf = MetricFrame(
        metrics={"tpr": true_positive_rate},
        y_true=y_true, y_pred=y_pred, sensitive_features=sensitive_features
    )
    tpr_by_group = mf.by_group["tpr"]
    eod_simple = float(tpr_by_group.max() - tpr_by_group.min())

    return {
        "demographic_parity_difference": round(float(dpd), 4),
        "demographic_parity_ratio": round(float(dpr), 4) if dpr is not None else None,
        "equalized_odds_difference": round(float(eod), 4),
        "equal_opportunity_difference": round(eod_simple, 4),
    }


def risk_from_disparity(value, low_cut=0.05, high_cut=0.15):
    """Translate a disparity metric (0-1 scale, larger = worse) into a
    LOW / MODERATE / HIGH label using documented, fixed cutoffs."""
    value = abs(value)
    if value <= low_cut:
        return "LOW"
    elif value <= high_cut:
        return "MODERATE"
    else:
        return "HIGH"
