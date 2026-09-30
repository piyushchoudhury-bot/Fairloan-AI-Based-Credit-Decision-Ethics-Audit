"""
generate_dataset.py
--------------------
Generates a synthetic, reproducible educational loan-approval dataset for
the FairLoan academic mini-project.

IMPORTANT: This dataset is entirely SYNTHETIC. It does not contain any real
person's financial information. It is built for teaching purposes only,
to demonstrate how bias, proxy features, and fairness issues can emerge in
AI-based credit decision systems.

Run:
    python generate_dataset.py
"""

import numpy as np
import pandas as pd
import os

# Fixed seed -> reproducible dataset every time this script is run.
SEED = 42
rng = np.random.default_rng(SEED)

N_RECORDS = 8000

OUTPUT_PATH = os.path.join("data", "loan_data.csv")


def generate_dataset(n=N_RECORDS):
    genders = rng.choice(["Male", "Female"], size=n, p=[0.52, 0.48])
    locations = rng.choice(
        ["Urban", "Suburban", "Rural"], size=n, p=[0.45, 0.35, 0.20]
    )
    education = rng.choice(
        ["High School", "Bachelor", "Master", "PhD"],
        size=n, p=[0.30, 0.42, 0.23, 0.05]
    )
    employment = rng.choice(
        ["Employed", "Self-Employed", "Unemployed", "Student"],
        size=n, p=[0.62, 0.18, 0.12, 0.08]
    )

    age = rng.integers(21, 66, size=n)

    # Income depends loosely on education/employment plus noise, and
    # (deliberately, for teaching purposes) has a mild historical skew by
    # gender and location so that downstream fairness/proxy modules have
    # something realistic, but not extreme, to detect.
    base_income = rng.normal(45000, 15000, size=n)
    edu_bonus = pd.Series(education).map(
        {"High School": 0, "Bachelor": 8000, "Master": 16000, "PhD": 24000}
    ).values
    employment_bonus = pd.Series(employment).map(
        {"Employed": 5000, "Self-Employed": 3000, "Unemployed": -15000, "Student": -10000}
    ).values
    # mild historical gender gap (real-world documented wage gap), used to
    # demonstrate historical bias propagation -- NOT to force a result.
    gender_adj = np.where(genders == "Female", -4000, 0)
    location_adj = pd.Series(locations).map(
        {"Urban": 4000, "Suburban": 1000, "Rural": -3000}
    ).values

    income = base_income + edu_bonus + employment_bonus + gender_adj + location_adj
    income = np.clip(income, 8000, None).round(2)

    # Credit score: correlated with income + a fair bit of independent noise
    credit_score = (
        550
        + (income - income.mean()) / 700
        + rng.normal(0, 45, size=n)
    )
    credit_score = np.clip(credit_score, 300, 850).round(0)

    previous_loans = rng.integers(0, 6, size=n)

    account_balance = np.clip(
        income * rng.uniform(0.05, 0.6, size=n) + rng.normal(0, 3000, size=n),
        0, None
    ).round(2)

    monthly_spending = np.clip(
        income / 12 * rng.uniform(0.2, 0.7, size=n) + rng.normal(0, 300, size=n),
        100, None
    ).round(2)

    transaction_count = np.clip(
        rng.normal(35, 15, size=n) + (monthly_spending / 200), 1, None
    ).round(0)

    # ---- Loan approval target ----
    # Built from a transparent, documented scoring rule + noise, so it has
    # realistic structure while remaining fully reproducible.
    score = (
        0.0025 * (credit_score - 300)
        + 0.000010 * income
        + 0.06 * previous_loans
        + 0.000015 * account_balance
        - 0.00008 * monthly_spending
        + np.where(employment == "Employed", 0.4, 0)
        + np.where(employment == "Self-Employed", 0.15, 0)
        + np.where(employment == "Unemployed", -0.9, 0)
        + np.where(employment == "Student", -0.3, 0)
        + np.where(education == "PhD", 0.15, 0)
        + np.where(education == "Master", 0.10, 0)
        + rng.normal(0, 0.35, size=n)
    )

    threshold = np.quantile(score, 0.42)  # calibrate ~58% approval baseline
    loan_approved = (score > threshold).astype(int)

    df = pd.DataFrame({
        "income": income,
        "employment": employment,
        "credit_score": credit_score,
        "education": education,
        "age": age,
        "location": locations,
        "previous_loans": previous_loans,
        "account_balance": account_balance,
        "monthly_spending": monthly_spending,
        "transaction_count": transaction_count,
        "gender": genders,
        "loan_approved": loan_approved,
    })

    # A few missing values / a few duplicate rows, injected on purpose so
    # the "Dataset Bias Detection" page has something real to report.
    missing_idx = rng.choice(df.index, size=int(0.01 * n), replace=False)
    df.loc[missing_idx, "account_balance"] = np.nan

    dup_rows = df.sample(n=int(0.005 * n), random_state=SEED)
    df = pd.concat([df, dup_rows], ignore_index=True)

    return df


def main():
    os.makedirs("data", exist_ok=True)
    df = generate_dataset()
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Synthetic dataset generated: {OUTPUT_PATH}")
    print(f"Rows: {len(df)}, Columns: {len(df.columns)}")
    print(df['loan_approved'].value_counts(normalize=True))


if __name__ == "__main__":
    main()
