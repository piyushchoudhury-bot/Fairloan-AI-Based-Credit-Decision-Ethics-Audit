# FairLoan: AI-Based Credit Decision Ethics Audit

> **⚠️ Academic / Educational Simulation Only**
> FairLoan is a mini-project built for a course in *Ethics in AI & Data Science*.
> It simulates an AI-based loan approval system purely for teaching purposes.
> **It is not a real banking or credit-scoring product** and must never be
> used to make actual financial decisions about real people.

---

## 1. Project Objective

FairLoan builds a small end-to-end AI system (a synthetic loan-approval
classifier) and then **audits that system** across the dimensions that matter
most in responsible-AI practice: dataset bias, fairness, explainability,
proxy discrimination through resource/cost features, deepfake-related
identity risk, and alignment with UNESCO/IEEE ethical AI principles.

## 2. Problem Statement

AI-based loan decision systems can improve efficiency, but they can also
introduce bias, proxy discrimination, lack of transparency, and other
ethical risks. FairLoan provides an educational framework for evaluating
these risks on a fully synthetic, reproducible dataset.

### Objectives
1. Analyze ethical risks in AI loan approval.
2. Detect dataset bias.
3. Assess deepfake-related vulnerabilities.
4. Investigate cost-as-a-proxy resource bias.
5. Evaluate fairness and explainability.
6. Operationalize responsible AI principles through an algorithmic impact assessment.

### Expected Outcomes
- Identify dataset disparities.
- Measure group-level fairness.
- Identify possible proxy features.
- Explain model predictions.
- Assess deepfake-related risks.
- Generate an overall ethical assessment.

## 3. Experiments Covered

| # | Page | Description |
|---|------|-------------|
| 1 | Ethical Analysis of AI Applications | Rule-based scoring across fairness, privacy, transparency, accountability, human oversight, and security. |
| 2 | Dataset Bias Detection | Representation, class balance, and group-wise outcome disparity analysis. |
| 3 | Deepfake Vulnerability Assessment | Simulated risk assessment of the identity/document verification pipeline. |
| 4 | Cost-as-a-Proxy Resource Bias | Compares a model trained with vs. without spending/balance/transaction features. |
| 5 | Fairness Dashboard | Fairlearn-based group metrics and disparity measures (demographic parity, equal opportunity, equalized odds). |
| 6 | Model Explainability | SHAP-based global and individual prediction explanations (with a documented fallback). |
| 7 | UNESCO & IEEE Algorithmic Impact Assessment | Educational questionnaire inspired by responsible-AI principles. |
| 8 | Final Ethics Report | Combines every module into one transparent, weighted overall ethics score. |

## 4. Technology Stack

**Backend:** Python 3, Flask
**ML:** pandas, numpy, scikit-learn, fairlearn, shap, joblib
**Frontend:** HTML5, CSS3 (custom), Bootstrap 5, JavaScript
**Visualization:** Chart.js
**Storage:** CSV (`data/loan_data.csv`)

## 5. Project Structure

```
FairLoan/
├── app.py                  # Flask application & routes
├── train_model.py          # Trains and saves the RandomForest model
├── generate_dataset.py     # Generates the reproducible synthetic dataset
├── requirements.txt
├── README.md
├── data/
│   └── loan_data.csv
├── model/
│   ├── loan_model.pkl
│   └── metadata.pkl
├── templates/
│   ├── base.html, index.html, predict.html, ethical_analysis.html,
│   ├── bias.html, deepfake.html, proxy_bias.html, fairness.html,
│   └── explainability.html, impact.html, report.html
├── static/
│   ├── css/style.css
│   └── js/dashboard.js
├── utils/
│   ├── data_utils.py       # Dataset loading & basic statistics
│   ├── fairness_utils.py   # Fairlearn wrappers
│   ├── ethics_utils.py     # Ethical analysis / deepfake / impact / report scoring
│   ├── proxy_bias.py       # Model A vs Model B experiment
│   └── explainability.py   # SHAP + fallback explanations
└── uploads/                # Temporary storage for deepfake-module uploads (auto-cleared)
```

## 6. Installation

```bash
python -m venv venv

# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
```

## 7. Generate the Dataset

```bash
python generate_dataset.py
```

This creates `data/loan_data.csv` — a synthetic, reproducible dataset
(fixed random seed = 42) of ~8,000 records.

## 8. Train the Model

```bash
python train_model.py
```

This trains a `RandomForestClassifier` on a 75/25 train/test split (seed=42)
using all features **except** `gender`, and saves:
- `model/loan_model.pkl` — the fitted scikit-learn pipeline
- `model/metadata.pkl` — feature lists, metrics, and the exact test-set
  indices (so every other page evaluates fairness/explainability on the
  same held-out data used for the headline accuracy figure).

## 9. Run the Application

```bash
python app.py
```

Then open **http://127.0.0.1:5000** in your browser.

## 10. Dataset Description

The dataset is entirely **synthetic** — no real individual's financial data
is used. Features:

`income, employment, credit_score, education, age, location, previous_loans,
account_balance, monthly_spending, transaction_count, gender, loan_approved`

`gender` is retained **only** for post-hoc fairness auditing; it is **never**
passed to the model as an input feature. The target (`loan_approved`) is
generated from a documented, transparent scoring formula (see
`generate_dataset.py`) plus random noise, with a deliberate, moderate
historical income gap by gender/location to make fairness auditing
meaningful — not to force an extreme or artificial result.

## 11. ML Methodology

- Model: `RandomForestClassifier` (300 trees, max depth 12), chosen for its
  strong tabular performance and straightforward compatibility with SHAP's
  `TreeExplainer`.
- Categorical features are one-hot encoded via a scikit-learn
  `ColumnTransformer`; numeric features pass through unchanged.
- Fixed random seed (42) throughout for reproducibility.
- Reported metrics: accuracy, precision, recall, F1, confusion matrix — all
  computed live from `train_model.py`, never hardcoded.

## 12. Fairness Methodology

Implemented in `utils/fairness_utils.py` using **Fairlearn**:
- Per-group accuracy, precision, recall, F1, selection rate, TPR, FPR, FNR
  via `fairlearn.metrics.MetricFrame`.
- Headline disparity measures: demographic parity difference/ratio,
  equalized odds difference, and equal opportunity difference (TPR gap).
- Risk labels (LOW/MODERATE/HIGH) use fixed, documented cutoffs
  (≤0.05 / ≤0.15 / >0.15) applied to the absolute disparity value.

## 13. Cost-as-Proxy Methodology

Implemented in `utils/proxy_bias.py`. Two identical RandomForest pipelines
are trained on the **same** train/test split:
- **Model A:** all standard features, including `monthly_spending`,
  `account_balance`, `transaction_count`.
- **Model B:** identical, minus those three resource-related features.

Performance and fairness metrics for both are computed live and compared,
with an automatically generated interpretation describing the accuracy vs.
fairness trade-off actually observed in that run.

## 14. Explainability Methodology

Implemented in `utils/explainability.py`. Uses **SHAP's `TreeExplainer`**
for both a global (mean absolute SHAP value across a sample) and individual
(per-applicant SHAP values) explanation. If SHAP raises any compatibility
error in a given environment, the code automatically falls back to the
RandomForest's built-in `feature_importances_` and clearly labels the
result as using the fallback method, so no explanation is ever silently
fabricated.

## 15. Deepfake Vulnerability Methodology

This module performs a **simulated, educational vulnerability assessment**
of the loan-approval identity/document verification ecosystem — it does
**not** run real forensic deepfake detection. Threat categories (identity
manipulation, document manipulation, video verification, synthetic
profiles) are scored using a fixed, documented rubric. If a user uploads a
file, only basic, explainable metadata heuristics (file size, extension)
are checked; the file is deleted immediately after the check and never
stored or executed. This distinction is stated clearly in the UI.

## 16. UNESCO/IEEE Assessment Methodology

`utils/ethics_utils.py` defines a 15-question checklist spanning fairness,
human rights, privacy, transparency, explainability, accountability, human
oversight, safety, security, human agency, and sustainability — categories
drawn from UNESCO's *Recommendation on the Ethics of Artificial
Intelligence* and IEEE's responsible-AI principles. Each answer
(Yes/Partially/No) is scored (1 / 0.5 / 0) and averaged per category and
overall. **This is an educational approximation, not an official
certification.** Official references:
- UNESCO: https://www.unesco.org/en/artificial-intelligence/recommendation-ethics
- IEEE: https://ethicsinaction.ieee.org/

## 17. Limitations

- The dataset is synthetic; real-world lending data would show different
  (and likely more complex) bias patterns.
- Fairness metrics are demonstrated only for `gender`, as a single example
  sensitive attribute — a production audit would examine multiple
  attributes and their intersections.
- The deepfake module is a simulated heuristic, not a validated forensic
  detector.
- The final "Overall Ethics Score" is one transparent, documented weighting
  scheme, not a universally standardized formula.
- This is a local, single-user Flask app without authentication — not
  suitable for deployment beyond a classroom demo.

## 18. Ethical Disclaimer

FairLoan is provided strictly for educational use in an Ethics in AI &
Data Science course context. It must not be used, adapted, or represented
as a real credit-decision system. Any resemblance between its synthetic
data and real individuals is coincidental — no real personal or financial
data is used anywhere in this project.

## 19. Future Enhancements

- Support additional sensitive attributes (age group, location) in the
  Fairness Dashboard with intersectional analysis.
- Add a bias-mitigation module (e.g., Fairlearn's `ExponentiatedGradient`)
  to demonstrate in-processing fairness interventions.
- Persist impact-assessment answers and generate a downloadable PDF report.
- Add a real (opt-in, consent-based) image-forensics model for the
  deepfake module in a future, non-classroom iteration.

---

### Demonstrating This Project in Class

1. Run `python generate_dataset.py` then `python train_model.py` live to
   show reproducibility (fixed seed → same accuracy every time).
2. Start with the **Dashboard** to give an overview of every metric.
3. Walk through **Dataset Bias → Fairness Dashboard → Cost-as-Proxy** in
   that order — it tells a coherent story: "here's a disparity, here's how
   it shows up in fairness metrics, here's a feature-level explanation for
   why."
4. Show **Explainability** on one specific rejected applicant to make the
   audit concrete.
5. Finish on the **Final Ethics Report** to show how every module rolls up
   into one transparent, documented score.
