import os
import pandas as pd
import numpy as np


# ============================================================
# TRUSTWORTHY AI - SOCIAL IMPACT ANALYSIS
# ============================================================

print("=" * 70)
print("TRUSTWORTHY AI - SOCIAL IMPACT ANALYSIS")
print("=" * 70)


# ============================================================
# 1. PROJECT PATH
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "../../.."
    )
)

DATA_PROCESSED = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed"
)

print("\nProject root:")
print(PROJECT_ROOT)


# ============================================================
# 2. FILE PATHS
# ============================================================

DIABETES_X_TEST = os.path.join(
    DATA_PROCESSED,
    "diabetes_X_test.csv"
)

DIABETES_Y_TEST = os.path.join(
    DATA_PROCESSED,
    "diabetes_y_test.csv"
)

HYPERTENSION_X_TEST = os.path.join(
    DATA_PROCESSED,
    "hypertension_X_test.csv"
)

HYPERTENSION_Y_TEST = os.path.join(
    DATA_PROCESSED,
    "hypertension_y_test.csv"
)


# ============================================================
# 3. CHECK FILES
# ============================================================

print("\n")
print("=" * 70)
print("CHECKING REQUIRED FILES")
print("=" * 70)

required_files = {
    "Diabetes X_test": DIABETES_X_TEST,
    "Diabetes y_test": DIABETES_Y_TEST,
    "Hypertension X_test": HYPERTENSION_X_TEST,
    "Hypertension y_test": HYPERTENSION_Y_TEST
}

for name, path in required_files.items():

    if os.path.exists(path):
        print(f"[OK] {name}: {path}")

    else:
        print(f"[MISSING] {name}: {path}")


# ============================================================
# 4. LOAD DATA
# ============================================================

diabetes_X = pd.read_csv(DIABETES_X_TEST)
diabetes_y = pd.read_csv(
    DIABETES_Y_TEST
).squeeze()

hypertension_X = pd.read_csv(HYPERTENSION_X_TEST)
hypertension_y = pd.read_csv(
    HYPERTENSION_Y_TEST
).squeeze()


# ============================================================
# 5. BASIC DATA INFORMATION
# ============================================================

print("\n")
print("=" * 70)
print("DATASET INFORMATION")
print("=" * 70)

print("\nDiabetes:")
print("X_test:", diabetes_X.shape)
print("y_test:", diabetes_y.shape)

print("\nHypertension:")
print("X_test:", hypertension_X.shape)
print("y_test:", hypertension_y.shape)


# ============================================================
# 6. MODEL PERFORMANCE FROM FINAL EVALUATION
# ============================================================

# Final values obtained from the model evaluation stage.

diabetes_metrics = {
    "Accuracy": 0.7857,
    "Precision": 0.6479,
    "Recall": 0.8519,
    "F1": 0.7360,
    "ROC-AUC": 0.8753,
    "TN": 75,
    "FP": 25,
    "FN": 8,
    "TP": 46
}

hypertension_metrics = {
    "Accuracy": 0.8998,
    "Precision": 0.7890,
    "Recall": 0.9240,
    "F1": 0.8511,
    "ROC-AUC": 0.9535,
    "TN": 520,
    "FP": 65,
    "FN": 20,
    "TP": 243
}


# ============================================================
# 7. SOCIAL BENEFITS
# ============================================================

print("\n")
print("=" * 70)
print("1. POTENTIAL SOCIAL BENEFITS")
print("=" * 70)

benefits = [
    "Early identification of patients with elevated disease risk.",
    "Support for healthcare professionals during risk assessment.",
    "Potential reduction in missed high-risk cases.",
    "Consistent application of the same prediction rules.",
    "Potential support for healthcare screening and triage.",
    "Can improve access to preliminary risk assessment in large populations."
]

for i, benefit in enumerate(benefits, 1):
    print(f"{i}. {benefit}")


# ============================================================
# 8. DIABETES SOCIAL IMPACT
# ============================================================

print("\n")
print("=" * 70)
print("2. DIABETES SOCIAL IMPACT")
print("=" * 70)

print("\nModel performance:")

for key, value in diabetes_metrics.items():

    if key in ["TN", "FP", "FN", "TP"]:
        continue

    print(f"{key:<10}: {value:.4f}")


print("\nError analysis:")

print(
    f"False Positive : {diabetes_metrics['FP']}"
)

print(
    f"False Negative : {diabetes_metrics['FN']}"
)

print(
    f"True Positive  : {diabetes_metrics['TP']}"
)

print(
    f"True Negative  : {diabetes_metrics['TN']}"
)


print("\nSocial interpretation:")

print(
    "- The diabetes model has relatively high recall."
)

print(
    "- This is important because missing a patient with diabetes "
    "risk may delay further assessment."
)

print(
    "- However, the model produces False Positive predictions."
)

print(
    "- False Positive predictions may cause unnecessary follow-up "
    "testing or patient anxiety."
)

print(
    "- Therefore, the prediction should be treated as a risk signal "
    "rather than a medical diagnosis."
)


# ============================================================
# 9. HYPERTENSION SOCIAL IMPACT
# ============================================================

print("\n")
print("=" * 70)
print("3. HYPERTENSION SOCIAL IMPACT")
print("=" * 70)

print("\nModel performance:")

for key, value in hypertension_metrics.items():

    if key in ["TN", "FP", "FN", "TP"]:
        continue

    print(f"{key:<10}: {value:.4f}")


print("\nError analysis:")

print(
    f"False Positive : {hypertension_metrics['FP']}"
)

print(
    f"False Negative : {hypertension_metrics['FN']}"
)

print(
    f"True Positive  : {hypertension_metrics['TP']}"
)

print(
    f"True Negative  : {hypertension_metrics['TN']}"
)


print("\nSocial interpretation:")

print(
    "- The hypertension model achieves high recall."
)

print(
    "- High recall is valuable for identifying patients who may "
    "require further blood-pressure assessment."
)

print(
    "- False Positive predictions may increase unnecessary "
    "clinical follow-up."
)

print(
    "- False Negative predictions remain a safety concern because "
    "some high-risk patients may not be flagged."
)


# ============================================================
# 10. RISK OF OVER-RELIANCE ON AI
# ============================================================

print("\n")
print("=" * 70)
print("4. RISK OF OVER-RELIANCE ON AI")
print("=" * 70)

risks = [
    "Users may treat AI predictions as medical diagnoses.",
    "Healthcare professionals may become overly dependent on automated predictions.",
    "Patients may misunderstand probability estimates.",
    "Incorrect predictions may influence subsequent medical decisions.",
    "Model performance may decrease when deployed on populations different from the training data."
]

for i, risk in enumerate(risks, 1):
    print(f"{i}. {risk}")


# ============================================================
# 11. INEQUALITY AND ACCESS
# ============================================================

print("\n")
print("=" * 70)
print("5. INEQUALITY AND ACCESS")
print("=" * 70)

print(
    "The system may provide useful risk assessment support, "
    "but access to AI-assisted healthcare may differ between "
    "urban and rural healthcare environments."
)

print(
    "\nPotential inequality risks:"
)

inequality_risks = [
    "Different access to computing infrastructure.",
    "Different availability of healthcare professionals.",
    "Differences in data quality between populations.",
    "Under-representation of some demographic groups.",
    "Different health profiles between training and deployment populations."
]

for i, risk in enumerate(inequality_risks, 1):
    print(f"{i}. {risk}")


# ============================================================
# 12. FAIRNESS AND BIAS CONNECTION
# ============================================================

print("\n")
print("=" * 70)
print("6. FAIRNESS AND BIAS CONNECTION")
print("=" * 70)

print(
    "Previous fairness and bias analyses showed that model "
    "performance differs between demographic groups."
)

print(
    "\nThis means social impact should not be evaluated only "
    "using overall accuracy."
)

print(
    "\nGroup-level Recall, Precision and F1 should also be monitored."
)

print(
    "\nA model can achieve high overall performance while still "
    "performing differently for particular groups."
)


# ============================================================
# 13. PATIENT SAFETY
# ============================================================

print("\n")
print("=" * 70)
print("7. PATIENT SAFETY")
print("=" * 70)

safety_principles = [
    "AI predictions should support rather than replace healthcare professionals.",
    "High-risk predictions should trigger further clinical assessment.",
    "Low-risk predictions should not automatically exclude disease.",
    "Users should be informed that predictions are probabilistic.",
    "The system should communicate model limitations.",
    "Model performance should be continuously monitored after deployment."
]

for i, principle in enumerate(safety_principles, 1):
    print(f"{i}. {principle}")


# ============================================================
# 14. POSITIVE SOCIAL IMPACT
# ============================================================

print("\n")
print("=" * 70)
print("8. POSITIVE SOCIAL IMPACT")
print("=" * 70)

positive_impacts = [
    "Earlier identification of potential health risks.",
    "Additional decision-support information for healthcare professionals.",
    "Potential support for large-scale screening.",
    "Consistent automated risk estimation.",
    "Potential reduction in missed high-risk cases.",
    "Improved awareness of health risk factors."
]

for i, impact in enumerate(positive_impacts, 1):
    print(f"{i}. {impact}")


# ============================================================
# 15. NEGATIVE SOCIAL IMPACT
# ============================================================

print("\n")
print("=" * 70)
print("9. POTENTIAL NEGATIVE SOCIAL IMPACT")
print("=" * 70)

negative_impacts = [
    "False predictions may cause unnecessary anxiety.",
    "False negatives may delay further medical assessment.",
    "Different performance between demographic groups may contribute to inequality.",
    "Over-reliance on AI may reduce human judgment.",
    "Poor interpretation of probabilities may lead to inappropriate decisions.",
    "Unauthorized access to healthcare data may create privacy risks."
]

for i, impact in enumerate(negative_impacts, 1):
    print(f"{i}. {impact}")


# ============================================================
# 16. RESPONSIBLE AI RECOMMENDATIONS
# ============================================================

print("\n")
print("=" * 70)
print("10. RESPONSIBLE AI RECOMMENDATIONS")
print("=" * 70)

recommendations = [
    "Use AI as decision support rather than autonomous diagnosis.",
    "Require qualified healthcare professionals to review important predictions.",
    "Monitor False Negative and False Positive rates.",
    "Continuously evaluate fairness across demographic groups.",
    "Protect patient data and restrict access to model and dataset files.",
    "Maintain prediction logs for auditing.",
    "Monitor model performance after deployment.",
    "Retrain or recalibrate models when significant performance degradation occurs.",
    "Clearly communicate uncertainty and limitations to users.",
    "Validate the system on representative real-world populations before clinical deployment."
]

for i, recommendation in enumerate(recommendations, 1):
    print(f"{i}. {recommendation}")


# ============================================================
# 17. SOCIAL IMPACT SCORECARD
# ============================================================

print("\n")
print("=" * 70)
print("11. SOCIAL IMPACT SCORECARD")
print("=" * 70)

scorecard = pd.DataFrame({
    "Dimension": [
        "Early risk detection",
        "Patient safety",
        "Human oversight",
        "Fairness awareness",
        "Privacy awareness",
        "Bias awareness",
        "Transparency",
        "Continuous monitoring"
    ],

    "Status": [
        "Implemented",
        "Partially implemented",
        "Required",
        "Implemented",
        "Implemented",
        "Implemented",
        "Implemented",
        "Recommended"
    ]
})

print(scorecard.to_string(index=False))


# ============================================================
# 18. FINAL SOCIAL IMPACT SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("SOCIAL IMPACT SUMMARY")
print("=" * 70)

print(
    "\nThe AI Healthcare Assistant has potential positive social "
    "impact through early risk identification and decision-support."
)

print(
    "\nHowever, incorrect predictions, demographic performance "
    "differences, privacy risks and over-reliance on AI may "
    "create negative consequences."
)

print(
    "\nTherefore, the system should be deployed as a decision-support "
    "tool with human oversight rather than as an autonomous diagnostic system."
)

print(
    "\nThe previous Fairness, Bias, Robustness, Explainability, "
    "Privacy and Accountability analyses should be considered "
    "together when evaluating the overall social impact."
)


# ============================================================
# 19. COMPLETED
# ============================================================

print("\n")
print("=" * 70)
print("SOCIAL IMPACT ANALYSIS COMPLETED")
print("=" * 70)

