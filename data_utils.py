"""
data_utils.py
-------------
Helper functions for loading the dataset and computing basic dataset-level
statistics used by the Dataset Bias Detection page (and others).
"""

import os
import pandas as pd

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "loan_data.csv")

SENSITIVE_COLUMNS = ["gender"]
MODEL_FEATURES = [
    "income", "employment", "credit_score", "education", "age",
    "location", "previous_loans", "account_balance",
    "monthly_spending", "transaction_count"
]
RESOURCE_PROXY_FEATURES = ["monthly_spending", "account_balance", "transaction_count"]
TARGET_COLUMN = "loan_approved"


def dataset_exists():
    return os.path.exists(DATA_PATH)


def load_dataset(dropna=False):
    """Load the loan dataset. Raises FileNotFoundError with a clear message
    if the dataset has not been generated yet."""
    if not dataset_exists():
        raise FileNotFoundError(
            "Dataset not found. Please run 'python generate_dataset.py' first."
        )
    df = pd.read_csv(DATA_PATH)
    if dropna:
        df = df.dropna()
    return df


def basic_dataset_stats(df):
    """Return a dictionary of basic descriptive statistics about the raw
    dataset, used on the Dataset Bias Detection page."""
    stats = {
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "missing_values": int(df.isna().sum().sum()),
        "missing_by_column": df.isna().sum()[df.isna().sum() > 0].to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "class_distribution": df[TARGET_COLUMN].value_counts(normalize=True).round(4).to_dict(),
    }
    return stats


def group_distribution(df, column):
    """Percentage distribution of categories within a given column."""
    return (df[column].value_counts(normalize=True) * 100).round(2).to_dict()


def group_approval_rates(df, group_column):
    """Loan approval rate (%) broken down by a categorical group column."""
    rates = df.groupby(group_column)[TARGET_COLUMN].mean() * 100
    counts = df.groupby(group_column)[TARGET_COLUMN].count()
    return {
        g: {"approval_rate": round(float(rates[g]), 2), "count": int(counts[g])}
        for g in rates.index
    }


def disparity_flag(rate_dict, threshold=10.0):
    """Simple heuristic: flag a 'meaningful' disparity if the max-min gap in
    approval rate across groups exceeds `threshold` percentage points.
    This is a signal for further investigation, NOT proof of discrimination.
    """
    rates = [v["approval_rate"] for v in rate_dict.values()]
    if len(rates) < 2:
        return False, 0.0
    gap = max(rates) - min(rates)
    return gap >= threshold, round(gap, 2)


def compute_dataset_bias_risk(df):
    """Aggregate a few disparity signals into a LOW / MODERATE / HIGH rating.
    This is an educational heuristic, not an official bias certification.
    """
    signals = 0
    total_checks = 0
    details = {}

    for col in ["gender", "location", "education"]:
        total_checks += 1
        rates = group_approval_rates(df, col)
        flagged, gap = disparity_flag(rates)
        details[col] = {"gap": gap, "flagged": flagged}
        if flagged:
            signals += 1

    # representation imbalance check (any group < 15% of data)
    total_checks += 1
    gender_dist = group_distribution(df, "gender")
    min_share = min(gender_dist.values()) if gender_dist else 100
    representation_flag = min_share < 15
    details["gender_representation"] = {"min_share_pct": min_share, "flagged": representation_flag}
    if representation_flag:
        signals += 1

    ratio = signals / total_checks
    if ratio <= 0.25:
        risk = "LOW"
    elif ratio <= 0.6:
        risk = "MODERATE"
    else:
        risk = "HIGH"

    return risk, details
