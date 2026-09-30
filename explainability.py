"""
explainability.py
------------------
Provides global and individual (local) explanations for the trained model.

Tries to use SHAP's TreeExplainer (fast + exact for tree ensembles). If SHAP
is unavailable or raises a compatibility error in this environment, falls
back to the model's built-in feature_importances_ so the page never breaks.
"""

import numpy as np
import pandas as pd

SHAP_AVAILABLE = True
try:
    import shap
except Exception:
    SHAP_AVAILABLE = False


def get_feature_names(preprocessor, numeric_features, categorical_features):
    """Recover human-readable feature names after a ColumnTransformer with
    a OneHotEncoder on categorical columns and passthrough numeric columns."""
    try:
        cat_encoder = preprocessor.named_transformers_["cat"]
        cat_names = list(cat_encoder.get_feature_names_out(categorical_features))
    except Exception:
        cat_names = categorical_features
    return cat_names + numeric_features  # matches ColumnTransformer(remainder="passthrough") order


def global_explanation(pipeline, X_sample, numeric_features, categorical_features, top_n=8):
    """Return a ranked list of {feature, importance} for global explanation."""
    preprocessor = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["classifier"]
    feature_names = get_feature_names(preprocessor, numeric_features, categorical_features)

    used_shap = False
    if SHAP_AVAILABLE:
        try:
            X_trans = preprocessor.transform(X_sample)
            if hasattr(X_trans, "toarray"):
                X_trans = X_trans.toarray()
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_trans)
            # for binary classifiers shap_values may be a list [class0, class1]
            if isinstance(shap_values, list):
                sv = shap_values[1]
            else:
                sv = shap_values
                if sv.ndim == 3:
                    sv = sv[:, :, 1]
            mean_abs = np.abs(sv).mean(axis=0)
            used_shap = True
        except Exception:
            used_shap = False

    if not used_shap:
        mean_abs = model.feature_importances_

    order = np.argsort(mean_abs)[::-1][:top_n]
    ranked = [
        {"feature": feature_names[i] if i < len(feature_names) else f"feature_{i}",
         "importance": round(float(mean_abs[i]), 5)}
        for i in order
    ]
    method = "SHAP (TreeExplainer)" if used_shap else "Random Forest built-in feature importance (SHAP fallback)"
    return ranked, method


def individual_explanation(pipeline, single_row_df, numeric_features, categorical_features, top_n=5):
    """Explain one applicant's prediction. Falls back to global feature
    importance weighted by the applicant's own (scaled) feature values if
    SHAP is unavailable."""
    preprocessor = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["classifier"]
    feature_names = get_feature_names(preprocessor, numeric_features, categorical_features)

    X_trans = preprocessor.transform(single_row_df)
    if hasattr(X_trans, "toarray"):
        X_trans = X_trans.toarray()

    used_shap = False
    if SHAP_AVAILABLE:
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_trans)
            if isinstance(shap_values, list):
                sv = shap_values[1][0]
            else:
                sv = shap_values[0]
                if sv.ndim == 2:
                    sv = sv[:, 1]
            used_shap = True
        except Exception:
            used_shap = False

    if not used_shap:
        # fallback: importance * normalized feature value magnitude (not a true
        # causal attribution -- this distinction is stated to the user in the UI)
        sv = model.feature_importances_ * X_trans[0]

    order = np.argsort(np.abs(sv))[::-1][:top_n]
    contributions = [
        {
            "feature": feature_names[i] if i < len(feature_names) else f"feature_{i}",
            "contribution": round(float(sv[i]), 5),
            "direction": "increases approval likelihood" if sv[i] > 0 else "decreases approval likelihood",
        }
        for i in order
    ]
    method = "SHAP (TreeExplainer)" if used_shap else "Feature importance x value (SHAP fallback)"
    return contributions, method
