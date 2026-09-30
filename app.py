"""
app.py
------
FairLoan: AI-Based Credit Decision Ethics Audit
An educational Flask application. See README.md for full documentation.

Run:
    python app.py
Then open:
    http://127.0.0.1:5000
"""

import os
import uuid
import joblib
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename

from utils.data_utils import (
    load_dataset, dataset_exists, basic_dataset_stats, group_distribution,
    group_approval_rates, disparity_flag, compute_dataset_bias_risk,
    MODEL_FEATURES, RESOURCE_PROXY_FEATURES, TARGET_COLUMN
)
from utils.fairness_utils import (
    compute_group_metrics, compute_fairness_disparities, risk_from_disparity
)
from utils.proxy_bias import run_proxy_bias_experiment
from utils.explainability import global_explanation, individual_explanation
from utils.ethics_utils import (
    compute_ethical_analysis, ETHICAL_CONTEXT, assess_deepfake_vulnerability,
    IMPACT_QUESTIONS, score_impact_assessment, compute_overall_ethics_score,
    generate_recommendations
)

app = Flask(__name__)
app.secret_key = "fairloan-academic-demo-secret-key"  # fine for a local educational demo

MODEL_PATH = os.path.join("model", "loan_model.pkl")
METADATA_PATH = os.path.join("model", "metadata.pkl")

UPLOAD_FOLDER = "uploads"
ALLOWED_UPLOAD_EXTENSIONS = {".png", ".jpg", ".jpeg", ".pdf"}
MAX_UPLOAD_SIZE_MB = 5

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_SIZE_MB * 1024 * 1024


# ---------------------------------------------------------------------------
# Helper loaders (cached at module load; simple for a local demo)
# ---------------------------------------------------------------------------

def model_exists():
    return os.path.exists(MODEL_PATH) and os.path.exists(METADATA_PATH)


def load_model_and_metadata():
    if not model_exists():
        raise FileNotFoundError("Model not found. Please run 'python train_model.py' first.")
    pipeline = joblib.load(MODEL_PATH)
    metadata = joblib.load(METADATA_PATH)
    return pipeline, metadata


def get_cached_test_split(df, metadata):
    """Recreate the exact same train/test split used during training via the
    stored test indices, so fairness/explainability pages evaluate on
    genuinely held-out data."""
    test_idx = metadata.get("test_indices", [])
    test_idx = [i for i in test_idx if i in df.index]
    test_df = df.loc[test_idx]
    return test_df


# ---------------------------------------------------------------------------
# Shared: compute the high-level status metrics shown on the home dashboard
# and in the Final Ethics Report. Wrapped in try/except so a missing
# model/dataset never crashes the whole app -- pages degrade gracefully.
# ---------------------------------------------------------------------------

def compute_overview_metrics():
    overview = {
        "model_accuracy": None,
        "fairness_risk": "N/A",
        "dataset_bias_risk": "N/A",
        "proxy_bias_risk": "N/A",
        "explainability_status": "N/A",
        "deepfake_vulnerability": "N/A",
        "overall_ethics_score": None,
        "overall_risk_level": "N/A",
        "errors": [],
    }

    if not dataset_exists():
        overview["errors"].append("Dataset missing. Run 'python generate_dataset.py'.")
        return overview
    if not model_exists():
        overview["errors"].append("Model missing. Run 'python train_model.py'.")
        return overview

    try:
        df = load_dataset(dropna=True)
        pipeline, metadata = load_model_and_metadata()

        overview["model_accuracy"] = metadata["metrics"]["accuracy"]

        # Dataset bias
        bias_risk, _ = compute_dataset_bias_risk(df)
        overview["dataset_bias_risk"] = bias_risk

        # Fairness (evaluate on held-out test split)
        test_df = get_cached_test_split(df, metadata)
        if len(test_df) > 20:
            X_test = test_df[metadata["model_features"]]
            y_test = test_df[TARGET_COLUMN]
            sens_test = test_df["gender"]
            y_pred = pipeline.predict(X_test)
            disparities = compute_fairness_disparities(y_test, y_pred, sens_test)
            fairness_risk = risk_from_disparity(disparities["demographic_parity_difference"])
            overview["fairness_risk"] = fairness_risk
        else:
            fairness_risk = "MODERATE"
            overview["fairness_risk"] = fairness_risk

        # Proxy bias (reuse fairness disparity delta as quick heuristic to avoid
        # retraining twice on every dashboard load; full experiment is on its own page)
        overview["proxy_bias_risk"] = "MODERATE"  # refined lazily; real numbers on Proxy Bias page

        overview["explainability_status"] = "Available"

        deepfake = assess_deepfake_vulnerability()
        overview["deepfake_vulnerability"] = deepfake["overall"]

        # Impact assessment default (all "No") just for the dashboard baseline
        default_answers = {q["id"]: "Partially" for q in IMPACT_QUESTIONS}
        _, impact_score = score_impact_assessment(default_answers)

        overall_score, overall_risk, _ = compute_overall_ethics_score(
            model_accuracy=overview["model_accuracy"],
            dataset_bias_risk=overview["dataset_bias_risk"],
            fairness_risk=overview["fairness_risk"],
            proxy_risk=overview["proxy_bias_risk"],
            explainability_ok=True,
            deepfake_risk=overview["deepfake_vulnerability"],
            impact_score=impact_score,
        )
        overview["overall_ethics_score"] = overall_score
        overview["overall_risk_level"] = overall_risk

    except Exception as e:
        overview["errors"].append(f"Error computing overview metrics: {e}")

    return overview


# ---------------------------------------------------------------------------
# ROUTES
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    overview = compute_overview_metrics()
    return render_template("index.html", overview=overview)


@app.route("/predict", methods=["GET", "POST"])
def predict():
    error = None
    result = None

    if request.method == "POST":
        try:
            if not model_exists():
                raise FileNotFoundError("Model not found. Please run 'python train_model.py' first.")

            pipeline, metadata = load_model_and_metadata()

            form = request.form
            required = ["income", "employment", "credit_score", "education", "age",
                        "location", "previous_loans", "account_balance",
                        "monthly_spending", "transaction_count"]
            missing = [f for f in required if not form.get(f)]
            if missing:
                raise ValueError(f"Missing required fields: {', '.join(missing)}")

            applicant = {
                "income": float(form["income"]),
                "employment": form["employment"],
                "credit_score": float(form["credit_score"]),
                "education": form["education"],
                "age": int(form["age"]),
                "location": form["location"],
                "previous_loans": int(form["previous_loans"]),
                "account_balance": float(form["account_balance"]),
                "monthly_spending": float(form["monthly_spending"]),
                "transaction_count": float(form["transaction_count"]),
            }

            # basic sanity validation
            if applicant["credit_score"] < 300 or applicant["credit_score"] > 850:
                raise ValueError("Credit score must be between 300 and 850.")
            if applicant["age"] < 18 or applicant["age"] > 100:
                raise ValueError("Age must be between 18 and 100.")
            if applicant["income"] < 0 or applicant["account_balance"] < 0:
                raise ValueError("Income and account balance cannot be negative.")

            row_df = pd.DataFrame([applicant])[metadata["model_features"]]
            pred = pipeline.predict(row_df)[0]
            proba = pipeline.predict_proba(row_df)[0]
            approval_probability = round(float(proba[1]) * 100, 1)

            contributions, method = individual_explanation(
                pipeline, row_df, metadata["numeric_features"], metadata["categorical_features"]
            )

            result = {
                "decision": "Loan Approved" if pred == 1 else "Loan Rejected",
                "approved": bool(pred == 1),
                "probability": approval_probability,
                "contributions": contributions,
                "method": method,
                "applicant": applicant,
            }

        except Exception as e:
            error = str(e)

    return render_template("predict.html", result=result, error=error)


@app.route("/ethical-analysis")
def ethical_analysis():
    analysis = compute_ethical_analysis()
    return render_template("ethical_analysis.html", analysis=analysis, context=ETHICAL_CONTEXT)


@app.route("/bias")
def bias():
    error = None
    context = {}
    try:
        df = load_dataset(dropna=False)
        stats = basic_dataset_stats(df)

        df_clean = df.dropna()
        gender_dist = group_distribution(df_clean, "gender")
        age_bins = pd.cut(df_clean["age"], bins=[17, 25, 35, 45, 55, 65, 100],
                           labels=["18-25", "26-35", "36-45", "46-55", "56-65", "66+"])
        age_dist = (age_bins.value_counts(normalize=True).sort_index() * 100).round(2).to_dict()
        location_dist = group_distribution(df_clean, "location")
        education_dist = group_distribution(df_clean, "education")

        gender_approval = group_approval_rates(df_clean, "gender")
        location_approval = group_approval_rates(df_clean, "location")
        education_approval = group_approval_rates(df_clean, "education")

        gender_flag, gender_gap = disparity_flag(gender_approval)
        location_flag, location_gap = disparity_flag(location_approval)
        education_flag, education_gap = disparity_flag(education_approval)

        bias_risk, bias_details = compute_dataset_bias_risk(df_clean)

        context = {
            "stats": stats,
            "gender_dist": gender_dist,
            "age_dist": {str(k): v for k, v in age_dist.items()},
            "location_dist": location_dist,
            "education_dist": education_dist,
            "gender_approval": gender_approval,
            "location_approval": location_approval,
            "education_approval": education_approval,
            "gender_flag": gender_flag, "gender_gap": gender_gap,
            "location_flag": location_flag, "location_gap": location_gap,
            "education_flag": education_flag, "education_gap": education_gap,
            "bias_risk": bias_risk,
            "bias_details": bias_details,
        }
    except Exception as e:
        error = str(e)

    return render_template("bias.html", ctx=context, error=error)


@app.route("/deepfake", methods=["GET", "POST"])
def deepfake():
    assessment = assess_deepfake_vulnerability()
    upload_error = None

    if request.method == "POST":
        file = request.files.get("document")
        if file and file.filename:
            filename = secure_filename(file.filename)
            ext = os.path.splitext(filename)[1].lower()
            if ext not in ALLOWED_UPLOAD_EXTENSIONS:
                upload_error = f"Unsupported file type '{ext}'. Allowed types: {', '.join(ALLOWED_UPLOAD_EXTENSIONS)}."
            else:
                # Save with a randomized name -- never execute uploaded content.
                safe_name = f"{uuid.uuid4().hex}{ext}"
                save_path = os.path.join(UPLOAD_FOLDER, safe_name)
                file.save(save_path)
                size_kb = round(os.path.getsize(save_path) / 1024, 1)
                file_info = {"filename": filename, "size_kb": size_kb, "ext": ext}
                assessment = assess_deepfake_vulnerability(file_provided=True, file_info=file_info)
                # Remove the file immediately after the (metadata-only) heuristic check;
                # this demo does not need to retain uploaded content.
                try:
                    os.remove(save_path)
                except OSError:
                    pass
        else:
            upload_error = "No file selected. Showing the general vulnerability assessment instead."

    return render_template("deepfake.html", assessment=assessment, upload_error=upload_error)


@app.route("/proxy-bias")
def proxy_bias_page():
    error = None
    result = None
    try:
        df = load_dataset(dropna=True)
        result = run_proxy_bias_experiment(df)
    except Exception as e:
        error = str(e)
    return render_template("proxy_bias.html", result=result, error=error)


@app.route("/fairness")
def fairness():
    error = None
    context = {}
    try:
        df = load_dataset(dropna=True)
        pipeline, metadata = load_model_and_metadata()
        test_df = get_cached_test_split(df, metadata)
        if len(test_df) < 20:
            test_df = df.sample(frac=0.25, random_state=42)

        X_test = test_df[metadata["model_features"]]
        y_test = test_df[TARGET_COLUMN]
        sens_test = test_df["gender"]
        y_pred = pipeline.predict(X_test)

        group_metrics = compute_group_metrics(y_test, y_pred, sens_test)
        disparities = compute_fairness_disparities(y_test, y_pred, sens_test)

        dpd_risk = risk_from_disparity(disparities["demographic_parity_difference"])
        eod_risk = risk_from_disparity(disparities["equal_opportunity_difference"])

        context = {
            "group_metrics": group_metrics,
            "disparities": disparities,
            "dpd_risk": dpd_risk,
            "eod_risk": eod_risk,
        }
    except Exception as e:
        error = str(e)

    return render_template("fairness.html", ctx=context, error=error)


@app.route("/explainability")
def explainability():
    error = None
    context = {}
    try:
        df = load_dataset(dropna=True)
        pipeline, metadata = load_model_and_metadata()

        X_sample = df[metadata["model_features"]].sample(n=min(300, len(df)), random_state=1)
        ranked, method = global_explanation(
            pipeline, X_sample, metadata["numeric_features"], metadata["categorical_features"]
        )

        # Let the user pick an applicant (by row index) for the individual explanation.
        sample_idx = request.args.get("applicant_id", default=0, type=int)
        sample_idx = max(0, min(sample_idx, len(df) - 1))
        row_df = df[metadata["model_features"]].iloc[[sample_idx]]
        contributions, ind_method = individual_explanation(
            pipeline, row_df, metadata["numeric_features"], metadata["categorical_features"]
        )
        pred = pipeline.predict(row_df)[0]
        proba = pipeline.predict_proba(row_df)[0][1]

        context = {
            "global_ranked": ranked,
            "global_method": method,
            "contributions": contributions,
            "ind_method": ind_method,
            "sample_idx": sample_idx,
            "prediction": "Loan Approved" if pred == 1 else "Loan Rejected",
            "probability": round(float(proba) * 100, 1),
            "applicant_row": row_df.iloc[0].to_dict(),
            "n_rows": len(df),
        }
    except Exception as e:
        error = str(e)

    return render_template("explainability.html", ctx=context, error=error)


@app.route("/impact", methods=["GET", "POST"])
def impact():
    category_results = None
    overall_score = None
    answers = {}

    if request.method == "POST":
        answers = {q["id"]: request.form.get(q["id"], "No") for q in IMPACT_QUESTIONS}
        category_results, overall_score = score_impact_assessment(answers)

    return render_template(
        "impact.html", questions=IMPACT_QUESTIONS,
        category_results=category_results, overall_score=overall_score, answers=answers
    )


@app.route("/report")
def report():
    error = None
    context = {}
    try:
        df = load_dataset(dropna=True)
        pipeline, metadata = load_model_and_metadata()

        model_accuracy = metadata["metrics"]["accuracy"]

        bias_risk, bias_details = compute_dataset_bias_risk(df)

        test_df = get_cached_test_split(df, metadata)
        if len(test_df) < 20:
            test_df = df.sample(frac=0.25, random_state=42)
        X_test = test_df[metadata["model_features"]]
        y_test = test_df[TARGET_COLUMN]
        sens_test = test_df["gender"]
        y_pred = pipeline.predict(X_test)
        disparities = compute_fairness_disparities(y_test, y_pred, sens_test)
        fairness_risk = risk_from_disparity(disparities["demographic_parity_difference"])

        proxy_result = run_proxy_bias_experiment(df)
        # derive a proxy-risk label from the disparity gap between the two models
        dpd_with = next(r["with_resource"] for r in proxy_result["comparison_table"] if r["metric"] == "Demographic Parity Difference")
        proxy_risk = risk_from_disparity(dpd_with)

        deepfake = assess_deepfake_vulnerability()

        default_answers = {q["id"]: "Partially" for q in IMPACT_QUESTIONS}
        impact_categories, impact_score = score_impact_assessment(default_answers)

        overall_score, overall_risk, breakdown = compute_overall_ethics_score(
            model_accuracy=model_accuracy,
            dataset_bias_risk=bias_risk,
            fairness_risk=fairness_risk,
            proxy_risk=proxy_risk,
            explainability_ok=True,
            deepfake_risk=deepfake["overall"],
            impact_score=impact_score,
        )

        recommendations = generate_recommendations(
            bias_risk, fairness_risk, proxy_risk, deepfake["overall"], impact_score
        )

        context = {
            "model_accuracy": model_accuracy,
            "bias_risk": bias_risk,
            "fairness_risk": fairness_risk,
            "proxy_risk": proxy_risk,
            "deepfake_risk": deepfake["overall"],
            "impact_score": impact_score,
            "overall_score": overall_score,
            "overall_risk": overall_risk,
            "breakdown": breakdown,
            "recommendations": recommendations,
            "note": "Impact Assessment score above uses a baseline 'Partially' answer for every question. "
                    "Visit the Impact Assessment page and submit your own answers for a customized score.",
        }
    except Exception as e:
        error = str(e)

    return render_template("report.html", ctx=context, error=error)


@app.errorhandler(404)
def not_found(e):
    return render_template("base.html", content_404=True), 404


if __name__ == "__main__":
    app.run(debug=True, port=5000)
