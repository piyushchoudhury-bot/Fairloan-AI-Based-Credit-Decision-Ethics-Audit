# FairLoan: AI-Based Credit Decision Ethics Audit

## Project Report

**Project type:** Educational web application and responsible-AI audit demonstration  
**Course:** [Course name]  
**Prepared by:** [Student name(s)]  
**Institution:** [Institution name]  
**Date:** [Submission date]

> **Important notice:** FairLoan is an academic simulation using synthetic data. It is not a lender, validated credit-scoring product, forensic deepfake detector, or official ethics certification. It must not be used to make decisions about real applicants.

## Abstract

FairLoan is a local web application that demonstrates how a machine-learning loan-approval workflow can be examined through responsible-AI and ethics concepts. It generates or loads a reproducible synthetic dataset, trains a Random Forest classifier, and provides interactive pages for individual predictions, dataset disparity analysis, group fairness metrics, feature explanations, resource-feature proxy comparisons, identity-verification risk, ethical analysis, an algorithmic impact questionnaire, and a combined ethics report.

The main model excludes gender from its prediction inputs and retains it for post-hoc fairness evaluation. The application demonstrates that excluding a sensitive attribute does not itself guarantee equal outcomes: group disparities can still arise through correlated features and historical patterns in the data. The project is intended to support learning and discussion. Its metrics, heuristics, and score aggregation are educational approximations and are not substitutes for legal review, domain expertise, real-world validation, or professional security testing.

## 1. Introduction

Automated credit systems can process applications consistently and at scale, but they can also reproduce patterns embedded in their data, rely on proxy variables, make errors that are difficult to contest, or provide inadequate explanations. A technically accurate model can therefore still raise fairness, transparency, privacy, accountability, and oversight concerns.

FairLoan makes these concerns observable in a small end-to-end example. A synthetic loan dataset feeds a classifier, while separate audit modules examine its data, outputs, feature use, and surrounding verification workflow. The user can move from a single prediction to aggregate metrics and then to a combined ethics summary.

## 2. Problem Statement and Objectives

The project addresses the question: **How can an educational loan-approval model be inspected for predictive performance and selected ethical risks before anyone considers a real-world system?**

The objectives are to:

1. Build a reproducible synthetic loan-approval dataset and a working classifier.
2. Demonstrate how data representation and outcome differences can be measured.
3. Compare model performance across gender groups using fairness metrics.
4. Explain global model behavior and individual predictions.
5. investigate whether resource-related features change model performance or group disparities.
6. Illustrate identity and document verification risks without claiming to detect deepfakes.
7. Organize responsible-AI questions into an educational impact assessment.
8. Summarize selected outputs in a transparent, weighted ethics score and recommendations.

## 3. Scope and Intended Users

FairLoan is designed for students, instructors, and project reviewers studying machine learning and AI ethics. It provides a local, single-user Flask interface and a fixed synthetic example. It is not designed for applicants, banks, production operations, or regulatory certification.

The system includes a loan prediction workflow and eight audit/reporting areas: ethical analysis, dataset bias, simulated deepfake vulnerability, cost-as-proxy bias, fairness, explainability, UNESCO/IEEE-inspired impact assessment, and the final ethics report. The home page links these activities in one dashboard.

## 4. System Design and Technology

### 4.1 Architecture

The application follows a compact server-rendered web architecture:

1. `generate_dataset.py` creates the CSV training and audit data.
2. `train_model.py` preprocesses features, fits the main classifier, and saves a model pipeline and metadata.
3. `app.py` serves Flask routes, validates submitted form values, loads model/data artifacts, and calls analysis utilities.
4. The `utils` modules implement dataset statistics, fairness metrics, explainability, proxy comparisons, ethical scoring, and impact assessment.
5. Jinja templates render each page; custom CSS, Bootstrap, JavaScript, and Chart.js provide the interface and visualizations.

### 4.2 Technology Stack

- **Language and server:** Python 3 and Flask
- **Data handling:** pandas and NumPy
- **Machine learning:** scikit-learn and joblib
- **Fairness metrics:** Fairlearn
- **Explainability:** SHAP with a documented fallback
- **Interface:** HTML, Jinja, custom CSS, Bootstrap 5, and JavaScript
- **Charts:** Chart.js
- **Persistent project data:** CSV dataset and serialized model/metadata files

### 4.3 Main Application Routes

| Route | Function |
|---|---|
| `/` | Project dashboard and navigation |
| `/predict` | Applicant-form prediction and individual explanation |
| `/ethical-analysis` | Rule-based ethical dimension scores and context |
| `/bias` | Dataset statistics, distributions, and approval-rate comparisons |
| `/deepfake` | Fixed system-risk rubric and optional metadata-only file check |
| `/proxy-bias` | Comparison of models with and without resource-related features |
| `/fairness` | Overall and gender-group performance and disparity measures |
| `/explainability` | Global feature influence and per-applicant explanation |
| `/impact` | UNESCO/IEEE-inspired questionnaire |
| `/report` | Combined weighted ethics summary and recommendations |

## 5. Dataset and Preparation

### 5.1 Dataset Characteristics

The dataset is generated locally with a fixed random seed of 42. The generator creates 8,000 base records, sets account-balance values to missing for approximately 1% of those records, and appends a 0.5% sample of duplicate records. The resulting CSV contains 8,040 rows and 12 columns in the observed project data.

The fields are:

- **Numeric predictors:** income, credit score, age, previous loans, account balance, monthly spending, and transaction count
- **Categorical predictors:** employment, education, and location
- **Audit attribute:** gender
- **Target:** loan approved (binary)

Gender is retained for post-hoc group analysis but is excluded from the main model’s input features. The outcome is constructed from a documented scoring formula using credit score, income, previous loans, account balance, monthly spending, employment, education, and random noise. The generator includes a mild synthetic income difference by gender and location so that students can investigate how historical patterns may propagate. No actual applicant or financial records are used.

### 5.2 Data Handling

The main training and several analysis routes load cleaned data with missing rows removed. Dataset-bias analysis also reports the original missing and duplicate counts. Categorical model features are one-hot encoded; numeric features pass through the preprocessing pipeline. The fixed seed and stored test indices support repeatable comparisons across modules.

## 6. Machine-Learning Methodology

The main model is a scikit-learn `RandomForestClassifier` with 300 trees, maximum depth 12, and random state 42. Data is split into training and test portions using a stratified 75/25 split. A `ColumnTransformer` applies `OneHotEncoder` to categorical features and passes numerical features through unchanged. The preprocessing and model are saved together as a pipeline.

The training script calculates accuracy, precision, recall, F1 score, and a confusion matrix on the held-out set. It also saves model feature lists, metrics, random seed, and test-set indices as metadata. The prediction page accepts applicant attributes, returns a predicted approval/rejection and an approval probability, and provides an individual explanation. Gender is not requested as a prediction input.

These predictions demonstrate a synthetic model’s behavior only. They are not estimates of the likelihood that any real person should receive credit.

## 7. Audit Modules and Methods

### 7.1 Ethical Analysis

This page presents six fixed rubric dimensions: fairness, privacy, transparency, accountability, human oversight, and security. It also describes stakeholders, benefits, potential harms, concerns, and recommended safeguards. The dimension values and explanatory text are educational rubric content; they are not calculated from a formal external audit or automatically updated from each model run.

### 7.2 Dataset Bias Detection

The dataset page reports row and column counts, missing values, duplicate rows, target distribution, and demographic distributions. It compares loan approval rates across gender, location, and education groups and displays warnings when configured disparity thresholds are met. Its aggregate bias label is a project heuristic, not a determination of unlawful discrimination. A low aggregate label does not establish that every group-level difference is unimportant.

### 7.3 Deepfake and Identity-Verification Vulnerability

This module rates risks in a hypothetical identity/document verification workflow across identity manipulation, document manipulation, video verification, and synthetic profiles. Those category ratings form a fixed system-level assessment. Optional uploads receive only basic file-size and file-type checks; the application does not inspect pixels, document content, or authenticity and does not detect deepfakes. Uploaded content is removed after the metadata check. The interface explicitly distinguishes this file result from the fixed system-level risk rating.

### 7.4 Cost-as-Proxy Resource Bias

The proxy experiment trains two Random Forest pipelines on the same seeded train/test split. Model A includes the standard predictors; Model B omits monthly spending, account balance, and transaction count. The page compares predictive metrics and selected group fairness measures, including demographic parity and equal opportunity, to illustrate potential performance/fairness trade-offs. This comparison is exploratory: a change in metrics does not prove that a feature caused discrimination or that removing it is always the correct intervention.

### 7.5 Fairness Dashboard

Fairlearn’s `MetricFrame` is used to calculate overall and gender-group accuracy, precision, recall, F1, selection rate, true-positive rate, false-positive rate, and false-negative rate. Headline measures include demographic parity difference and ratio, equalized-odds difference, and equal-opportunity difference represented as the gap in true-positive rates. The displayed LOW/MODERATE/HIGH disparity labels use fixed cutoffs: at most 0.05 is LOW, at most 0.15 is MODERATE, and larger values are HIGH.

A disparity metric is a signal for investigation, not by itself proof of bias. Different fairness definitions can conflict, and domain context, data quality, legal requirements, and error consequences must also be considered.

### 7.6 Explainability

The explainability module attempts SHAP TreeExplainer for global feature influence and individual prediction contributions. If SHAP cannot run in the environment, the application uses a fallback based on the model’s built-in feature importance; individual fallback contributions are approximated from feature importance and transformed feature values. The interface identifies the fallback. Neither SHAP values nor the fallback establish causation or guarantee that an explanation captures every reason for a prediction.

### 7.7 UNESCO/IEEE-Inspired Impact Assessment

The impact page presents 15 questions spanning fairness, human rights, privacy, transparency, explainability, accountability, human oversight, safety, security, human agency, and sustainability. Answers are scored Yes = 1, Partially = 0.5, and No = 0; results are summarized by category and overall. The questionnaire is inspired by responsible-AI principles and is not an official UNESCO or IEEE assessment or certification.

### 7.8 Final Ethics Report

The combined report presents model accuracy, dataset-bias risk, fairness risk, proxy risk, explainability availability, deepfake vulnerability, an impact score, an overall ethics score, a weighted breakdown, and recommendations. Its component weights are:

| Component | Weight |
|---|---:|
| Model performance | 10% |
| Dataset bias | 15% |
| Fairness | 20% |
| Proxy bias | 15% |
| Explainability | 10% |
| Deepfake vulnerability | 10% |
| Algorithmic impact assessment | 20% |

Risk labels are converted to project-defined numbers and risk-related components are inverted before aggregation. The overall label is LOW at 75 or above, MODERATE from 50 to below 75, and HIGH below 50. This is a transparent project scoring scheme, not a standardized ethics index. The combined report currently uses a baseline “Partially” response for each impact question rather than persisting a user’s submitted questionnaire answers.

## 8. Observed Results

The following values are from the project’s current local dashboard run. They illustrate the output; they should be regenerated if the dataset, model artifacts, or implementation changes.

### 8.1 Dataset and Group Outcomes

| Measure | Observed result |
|---|---:|
| Dataset rows | 8,040 |
| Dataset columns | 12 |
| Missing values | 80 |
| Duplicate rows | 40 |
| Current aggregate dataset-bias label | LOW |

Observed approval rates were 56.28% for women and 59.48% for men, a 3.20 percentage-point difference. Rates by location were 51.42% Rural, 58.51% Suburban, and 60.38% Urban. Rates by education were 50.36% High School, 56.03% Bachelor, 68.17% Master, and 69.92% PhD. The highest-to-lowest education approval-rate gap was 19.56 percentage points. This illustrates why individual group results should be examined even when an aggregate heuristic label is LOW.

### 8.2 Main Model and Fairness

| Metric | Observed result |
|---|---:|
| Accuracy | 0.8236 |
| Precision | 0.8203 |
| Recall | 0.8907 |
| F1 score | 0.8541 |
| Demographic parity difference | 0.0326 (LOW by project cutoff) |
| Demographic parity ratio | 0.9495 |
| Equal opportunity difference | 0.0313 (LOW by project cutoff) |
| Equalized odds difference | 0.0313 |

The observed test-set accuracy is approximately 82.36%. The displayed gender-group accuracy was 0.8135 for women and 0.8335 for men. These statistics describe one synthetic test split and should not be interpreted as real-world model validation.

### 8.3 Resource-Feature Comparison

| Measure | With resource features | Without resource features |
|---|---:|---:|
| Accuracy | 0.8236 | 0.8141 |
| Demographic parity difference | 0.0282 | 0.0230 |
| Equal opportunity difference | 0.0255 | 0.0169 |

In this observed run, omitting the three resource features slightly reduced accuracy while also reducing the displayed parity and opportunity gaps. These values are descriptive results from this experiment, not evidence that the removed features caused the original disparities. The overall result depends on the synthetic data-generation process and chosen model.

## 9. Ethical Considerations and Safeguards

The project uses synthetic data and excludes gender from the main prediction inputs, reducing the risk of exposing real personal information and allowing a controlled demonstration of post-hoc auditing. It also makes its educational purpose and deepfake-module limitations visible in the interface.

Important remaining concerns include the synthetic historical pattern deliberately encoded in income, the possibility of indirect proxies in otherwise permitted features, the limited single-attribute fairness analysis, the use of probabilistic model outputs in a high-impact domain, and the absence of appeal or human-review workflows in the prediction demo. Any real lending application would require lawful data governance, privacy protections, validated performance, ongoing subgroup monitoring, documented human oversight, accessible explanations, and a meaningful contest/appeal process.

## 10. Limitations and Implementation Caveats

1. **Synthetic data:** Results do not establish performance or fairness on real populations, institutions, or lending decisions.
2. **Limited subgroup coverage:** The fairness demonstration primarily uses gender and does not provide intersectional or broader protected-group analysis.
3. **Rubric-based judgments:** Ethical-analysis scores and deepfake system-risk categories are fixed project rules, not learned measurements or independently validated assessments.
4. **Metadata-only file check:** File size and extension cannot establish whether an image or document is authentic or manipulated.
5. **Combined report baseline:** The final report uses “Partially” for all impact questions; custom questionnaire submissions are not currently persisted into that report.
6. **Proxy risk summary:** The combined report derives its proxy-risk label from the with-resource model’s demographic parity result rather than directly scoring the change between the two models. The A/B page should be consulted for the actual comparison.
7. **Ethical-analysis label semantics:** The current rule-based ethical-analysis code labels low numeric scores as LOW risk and high numeric scores as HIGH risk, although the dimension text reads like higher scores represent stronger practice. This mapping should be reviewed before interpreting that page.
8. **Deployment security:** The application is a local demo without authentication and runs Flask in debug mode. It is not configured for public or production deployment.
9. **Explanations are not causal:** SHAP and the fallback describe model behavior, not causal effects or a complete justification for a credit decision.
10. **Fairness trade-offs:** No single metric captures fairness in every context. A full assessment needs stakeholder participation and domain/legal review.

## 11. Future Work

- Persist impact-assessment answers and use them in the combined report.
- Correct and test risk-label semantics in the ethical-analysis page.
- Compute the combined proxy-risk label from the measured change between the two model variants and document the chosen comparison rule.
- Add confidence intervals, repeated seeds, and subgroup sample counts to make metric uncertainty visible.
- Expand fairness analysis to multiple protected attributes and intersectional groups where appropriate and lawful.
- Add documented mitigation experiments, then re-evaluate both predictive performance and group outcomes.
- Improve data validation, audit logging, access controls, secret management, and deployment configuration before any broader use.
- Add report export, such as PDF generation, while preserving the methodology and limitations.
- Only consider forensic file analysis as a separate feature using a validated, consent-based system with clear privacy and accuracy disclosures; metadata heuristics must not be presented as detection.

## 12. Conclusion

FairLoan provides a practical classroom workflow for seeing how model development and ethics auditing can be connected. Its synthetic classifier produces measurable predictions, and its dashboards expose data distributions, group performance, feature contributions, and selected system-level risks. The observed run demonstrates that reasonable aggregate fairness values can coexist with larger differences for particular groups, and that removing resource-related predictors can alter both performance and disparities.

The project’s main value is educational: it helps users ask better questions about data, metrics, features, explanations, and governance. Its results are not evidence that a real credit system is fair, safe, or suitable for deployment. Any real-world use would require substantially broader validation, oversight, security, and stakeholder review.

## 13. Installation and Execution

From the `FairLoan` project directory, create and activate a virtual environment, install dependencies, generate the dataset if needed, train the model, and start the server:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python generate_dataset.py
python train_model.py
python app.py
```

Then open `http://127.0.0.1:5000` in a browser. On systems where PowerShell script activation is restricted, use the environment’s Python executable directly or activate the environment using the appropriate shell command. The model and metadata artifacts are created under `model/` by the training script.

## 14. Project Files

- `app.py` — Flask routes, prediction workflow, and report assembly
- `generate_dataset.py` — reproducible synthetic dataset generation
- `train_model.py` — model preprocessing, training, evaluation, and artifact saving
- `utils/data_utils.py` — dataset loading, statistics, and dataset-bias heuristics
- `utils/fairness_utils.py` — group metrics and disparity calculations
- `utils/proxy_bias.py` — with/without resource-feature comparison
- `utils/explainability.py` — SHAP explanations and fallback behavior
- `utils/ethics_utils.py` — ethical rubrics, simulated verification-risk assessment, impact scoring, and final report scoring
- `templates/` — dashboard, prediction, experiment, and report pages
- `static/` — shared styles and browser-side helpers
- `data/loan_data.csv` — synthetic dataset
- `model/` — trained model and metadata artifacts

## References

1. UNESCO, *Recommendation on the Ethics of Artificial Intelligence*. https://www.unesco.org/en/artificial-intelligence/recommendation-ethics
2. IEEE, *Ethics in Action in the Age of AI*. https://ethicsinaction.ieee.org/
3. Fairlearn documentation. https://fairlearn.org/
4. SHAP documentation. https://shap.readthedocs.io/
5. scikit-learn documentation. https://scikit-learn.org/
