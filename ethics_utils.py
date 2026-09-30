"""
ethics_utils.py
----------------
Houses the scoring logic for:
  1. Experiment 1: Ethical Analysis of AI Applications (rule-based scoring)
  2. Experiment 3: Deepfake Vulnerability Assessment (heuristic risk engine)
  3. Experiment 6: UNESCO / IEEE Algorithmic Impact Assessment (questionnaire scoring)
  4. Final Ethics Report (transparent aggregation of all module outputs)

None of these produce an "official" UNESCO/IEEE certification. They are
educational, rule-based approximations designed to teach the underlying
concepts, and this is stated explicitly in the UI.
"""

# ---------------------------------------------------------------------------
# 1. ETHICAL ANALYSIS (Experiment 1)
# ---------------------------------------------------------------------------

ETHICAL_CATEGORIES = {
    "fairness": {
        "score": 60,
        "explanation": (
            "The system relies on historical loan data that may embed past "
            "societal inequities. Group-level disparities were observed in the "
            "Fairness Dashboard, which is why this score is moderate rather than high."
        ),
    },
    "privacy": {
        "score": 78,
        "explanation": (
            "The demo does not store real personal financial data and uses a "
            "synthetic dataset. In a real deployment, applicant financial data "
            "would require strict access control and data minimization, which "
            "keeps this from being scored as fully addressed."
        ),
    },
    "transparency": {
        "score": 68,
        "explanation": (
            "The Explainability module provides global and per-prediction "
            "feature contributions, which supports transparency. The score is "
            "not higher because explanations describe correlation-based feature "
            "importance rather than a full causal account of the decision."
        ),
    },
    "accountability": {
        "score": 62,
        "explanation": (
            "There is no automated appeals process or designated human reviewer "
            "role implemented in this educational demo, which limits accountability "
            "compared to a production-grade credit system."
        ),
    },
    "human_oversight": {
        "score": 55,
        "explanation": (
            "Predictions in this demo are fully automated with no human-in-the-loop "
            "review step before a decision is finalized, which is a meaningful gap "
            "for any high-stakes decision such as credit access."
        ),
    },
    "security": {
        "score": 70,
        "explanation": (
            "Basic file validation and upload restrictions are implemented for the "
            "deepfake module. However, this is a local demo without production-grade "
            "authentication, encryption at rest, or audit logging."
        ),
    },
}


def compute_ethical_analysis():
    overall = round(sum(v["score"] for v in ETHICAL_CATEGORIES.values()) / len(ETHICAL_CATEGORIES), 1)
    if overall < 40:
        risk_level = "LOW"
    elif overall < 70:
        risk_level = "MODERATE"
    else:
        risk_level = "HIGH"
    return {
        "categories": ETHICAL_CATEGORIES,
        "overall_score": overall,
        "risk_level": risk_level,
    }


ETHICAL_CONTEXT = {
    "stakeholders": [
        "Loan applicants (esp. from historically underrepresented groups)",
        "Bank / lending institution",
        "Regulators and auditors",
        "Data scientists and engineers maintaining the model",
        "Society at large (systemic access to credit)",
    ],
    "potential_benefits": [
        "Faster, more consistent loan processing",
        "Reduced manual workload for loan officers",
        "Potential to apply uniform criteria rather than individual human bias",
    ],
    "potential_harms": [
        "Automated denial of credit to historically disadvantaged groups",
        "Opaque decisions that applicants cannot meaningfully contest",
        "Reinforcement of existing socioeconomic inequities via proxy features",
    ],
    "ethical_concerns": [
        "Fairness across demographic groups",
        "Transparency and the right to an explanation",
        "Accountability when an automated decision causes harm",
        "Appropriate human oversight for high-stakes decisions",
    ],
    "recommended_safeguards": [
        "Regular fairness and bias audits (see Fairness Dashboard)",
        "Mandatory human review for borderline / high-impact decisions",
        "Clear applicant-facing explanations of adverse decisions",
        "A documented appeals process",
        "Ongoing monitoring for feature drift and proxy discrimination",
    ],
}


# ---------------------------------------------------------------------------
# 3. DEEPFAKE VULNERABILITY ASSESSMENT (Experiment 3)
# ---------------------------------------------------------------------------

def assess_deepfake_vulnerability(file_provided=False, file_info=None):
    """A controlled, educational vulnerability-assessment heuristic. This is
    NOT real forensic deepfake detection -- it is a simulated risk-scoring
    engine meant to teach students about the categories of risk in an
    identity-verification pipeline attached to an AI loan system."""

    categories = {
        "Identity Manipulation": {
            "level": "HIGH",
            "explanation": (
                "Many loan platforms accept photos of ID documents without live "
                "verification, making them vulnerable to fabricated or digitally "
                "altered identity documents."
            ),
            "consequence": "Approval of loans to fraudulent or synthetic identities.",
            "mitigation": "Use certified liveness detection and cross-check with government ID databases.",
        },
        "Document Manipulation": {
            "level": "HIGH",
            "explanation": (
                "Income statements, payslips, and bank statements can be edited "
                "with common photo/PDF editing tools or generated synthetically."
            ),
            "consequence": "Loans approved based on falsified income or financial standing.",
            "mitigation": "Cross-verify documents directly with issuing institutions (e.g., payroll, banks).",
        },
        "Video Verification": {
            "level": "MEDIUM",
            "explanation": (
                "Real-time video verification is harder to spoof than static images, "
                "but is still vulnerable to advanced deepfake video / voice-cloning attacks."
            ),
            "consequence": "A fraudulent applicant could impersonate a legitimate person in a live call.",
            "mitigation": "Combine video verification with device/behavioral biometrics and challenge-response prompts.",
        },
        "Synthetic Profile": {
            "level": "MEDIUM",
            "explanation": (
                "AI image generators can produce plausible human faces and "
                "consistent fake profiles that are difficult to distinguish visually.",
            ),
            "consequence": "Creation of entirely fictitious applicant identities at scale.",
            "mitigation": "Apply deepfake-detection classifiers and require multi-factor identity proofing.",
        },
    }

    # If the user actually uploaded a file, add a lightweight simulated
    # per-file signal (based only on simple, explainable heuristics such as
    # file size/type -- NOT real forensic analysis) so the demo feels
    # interactive without pretending to run a real detector.
    file_note = None
    if file_provided and file_info:
        size_kb = file_info.get("size_kb", 0)
        ext = file_info.get("ext", "").lower()
        flags = []
        if size_kb < 15:
            flags.append("Unusually small file size for a photo/document (possible re-compression or tampering).")
        if ext not in [".jpg", ".jpeg", ".png", ".pdf"]:
            flags.append("Unexpected file type for an identity document upload.")
        file_note = {
            "filename": file_info.get("filename"),
            "size_kb": size_kb,
            "flags": flags if flags else ["No basic heuristic flags triggered."],
            "status": "NO BASIC FLAGS" if not flags else "REVIEW METADATA",
            "status_class": "low" if not flags else "moderate",
            "disclaimer": (
                "This is a SIMULATED heuristic check based only on file metadata "
                "(size/type), not an actual forensic deepfake or tamper detector."
            ),
        }

    levels = [c["level"] for c in categories.values()]
    if levels.count("HIGH") >= 2:
        overall = "HIGH"
    elif "HIGH" in levels:
        overall = "MODERATE"
    else:
        overall = "LOW"

    harms_explained = {
        "identity_fraud": "Criminals impersonate real or synthetic people to gain unauthorized access to financial products.",
        "financial_fraud": "Falsified income/asset documents lead to loans that would not otherwise be approved.",
        "incorrect_loan_decisions": "The model makes decisions on fabricated inputs, corrupting both approvals and denials.",
        "privacy_risks": "Collecting biometric/video data for verification introduces its own privacy and data-protection risks.",
        "reputational_harm": "A lending institution that approves fraudulent loans faces regulatory and reputational damage.",
    }

    return {
        "categories": categories,
        "overall": overall,
        "file_note": file_note,
        "harms_explained": harms_explained,
    }


# ---------------------------------------------------------------------------
# 6. UNESCO / IEEE ALGORITHMIC IMPACT ASSESSMENT (Experiment 6)
# ---------------------------------------------------------------------------

IMPACT_QUESTIONS = [
    {"id": "q1", "category": "Fairness", "question": "Is the training data evaluated for representativeness across demographic groups?"},
    {"id": "q2", "category": "Fairness", "question": "Are group-level outcome disparities regularly measured?"},
    {"id": "q3", "category": "Human Rights", "question": "Has the system been assessed for potential discriminatory impact on protected groups?"},
    {"id": "q4", "category": "Privacy", "question": "Are privacy risks of collected applicant data formally assessed?"},
    {"id": "q5", "category": "Privacy", "question": "Is data collection limited to what is necessary for the credit decision?"},
    {"id": "q6", "category": "Transparency", "question": "Are automated decisions explainable to affected users?"},
    {"id": "q7", "category": "Transparency", "question": "Is it disclosed to applicants that an AI system is involved in the decision?"},
    {"id": "q8", "category": "Explainability", "question": "Can the system provide the top factors behind an individual decision?"},
    {"id": "q9", "category": "Accountability", "question": "Can an applicant challenge or appeal an automated decision?"},
    {"id": "q10", "category": "Accountability", "question": "Is there a clear, named party responsible for model outcomes?"},
    {"id": "q11", "category": "Human Oversight", "question": "Does the system provide meaningful human oversight before high-impact decisions?"},
    {"id": "q12", "category": "Safety", "question": "Is the model tested for robustness against unusual or adversarial inputs?"},
    {"id": "q13", "category": "Security", "question": "Are identity-verification documents protected against tampering and forgery?"},
    {"id": "q14", "category": "Human Agency", "question": "Do applicants retain meaningful agency (e.g. can they opt for manual review)?"},
    {"id": "q15", "category": "Sustainability", "question": "Is the computational cost of retraining/maintaining the model considered sustainable?"},
]

ANSWER_SCORES = {"Yes": 1.0, "Partially": 0.5, "No": 0.0}


def score_impact_assessment(answers):
    """answers: dict of {question_id: 'Yes'|'Partially'|'No'}. Returns
    per-category scoring and an overall educational score out of 100."""
    from collections import defaultdict

    cat_scores = defaultdict(list)
    for q in IMPACT_QUESTIONS:
        ans = answers.get(q["id"], "No")
        cat_scores[q["category"]].append(ANSWER_SCORES.get(ans, 0.0))

    category_results = []
    all_scores = []
    for cat, scores in cat_scores.items():
        pct = round(sum(scores) / len(scores) * 100, 1)
        all_scores.extend(scores)
        if pct >= 70:
            status, rec = "Good", "Maintain current practice; monitor periodically."
        elif pct >= 40:
            status, rec = "Needs Improvement", "Prioritize concrete action items in this category."
        else:
            status, rec = "Weak", "Urgent attention recommended before any real-world use."
        category_results.append({
            "category": cat, "score": pct, "status": status, "recommendation": rec
        })

    overall_score = round(sum(all_scores) / len(all_scores) * 100, 1) if all_scores else 0.0
    return category_results, overall_score


# ---------------------------------------------------------------------------
# 4. FINAL ETHICS REPORT
# ---------------------------------------------------------------------------

def risk_label_to_number(label):
    return {"LOW": 20, "MODERATE": 55, "HIGH": 85}.get(label, 50)


def compute_overall_ethics_score(model_accuracy, dataset_bias_risk, fairness_risk,
                                  proxy_risk, explainability_ok, deepfake_risk,
                                  impact_score):
    """
    Transparent weighted aggregation:
      - Model performance (10%): accuracy contributes positively.
      - Dataset bias risk (15%): inverted (lower risk -> higher sub-score).
      - Fairness risk (20%): inverted.
      - Proxy bias risk (15%): inverted.
      - Explainability (10%): 100 if SHAP/feature-importance explanations available, else 40.
      - Deepfake vulnerability (10%): inverted.
      - Algorithmic impact assessment score (20%): used directly.

    All weights and the formula are documented here and in the README so the
    final number is fully explainable, not a black box.
    """
    weights = {
        "model_perf": 0.10,
        "dataset_bias": 0.15,
        "fairness": 0.20,
        "proxy_bias": 0.15,
        "explainability": 0.10,
        "deepfake": 0.10,
        "impact": 0.20,
    }

    model_perf_sub = model_accuracy * 100
    dataset_bias_sub = 100 - risk_label_to_number(dataset_bias_risk)
    fairness_sub = 100 - risk_label_to_number(fairness_risk)
    proxy_sub = 100 - risk_label_to_number(proxy_risk)
    explainability_sub = 100 if explainability_ok else 40
    deepfake_sub = 100 - risk_label_to_number(deepfake_risk)
    impact_sub = impact_score

    overall = (
        weights["model_perf"] * model_perf_sub
        + weights["dataset_bias"] * dataset_bias_sub
        + weights["fairness"] * fairness_sub
        + weights["proxy_bias"] * proxy_sub
        + weights["explainability"] * explainability_sub
        + weights["deepfake"] * deepfake_sub
        + weights["impact"] * impact_sub
    )
    overall = round(overall, 1)

    if overall >= 75:
        risk_level = "LOW"
    elif overall >= 50:
        risk_level = "MODERATE"
    else:
        risk_level = "HIGH"

    breakdown = [
        {"component": "Model Performance", "weight": weights["model_perf"], "sub_score": round(model_perf_sub, 1)},
        {"component": "Dataset Bias (inverted)", "weight": weights["dataset_bias"], "sub_score": round(dataset_bias_sub, 1)},
        {"component": "Fairness (inverted)", "weight": weights["fairness"], "sub_score": round(fairness_sub, 1)},
        {"component": "Proxy Bias (inverted)", "weight": weights["proxy_bias"], "sub_score": round(proxy_sub, 1)},
        {"component": "Explainability", "weight": weights["explainability"], "sub_score": round(explainability_sub, 1)},
        {"component": "Deepfake Vulnerability (inverted)", "weight": weights["deepfake"], "sub_score": round(deepfake_sub, 1)},
        {"component": "Algorithmic Impact Assessment", "weight": weights["impact"], "sub_score": round(impact_sub, 1)},
    ]

    return overall, risk_level, breakdown


def generate_recommendations(dataset_bias_risk, fairness_risk, proxy_risk,
                              deepfake_risk, impact_score):
    recs = []
    if dataset_bias_risk in ("MODERATE", "HIGH"):
        recs.append("Investigate demographic representation and historical bias in the training data.")
    if proxy_risk in ("MODERATE", "HIGH"):
        recs.append("Review resource-related features (spending, balance, transactions) for proxy discrimination effects.")
    if fairness_risk in ("MODERATE", "HIGH"):
        recs.append("Monitor group-level false positive and false negative rates on an ongoing basis.")
    recs.append("Provide clear, accessible explanations for every automated decision.")
    recs.append("Add mandatory human review for high-impact or borderline decisions.")
    if deepfake_risk in ("MODERATE", "HIGH"):
        recs.append("Strengthen identity and document verification against synthetic/manipulated evidence.")
    recs.append("Conduct periodic, independent fairness and bias audits.")
    recs.append("Maintain living documentation of model limitations and known risks.")
    if impact_score < 60:
        recs.append("Close gaps identified in the UNESCO/IEEE Algorithmic Impact Assessment before considering real-world use.")
    return recs
