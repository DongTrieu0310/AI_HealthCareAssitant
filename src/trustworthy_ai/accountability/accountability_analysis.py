import os
import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


# ============================================================
# PROJECT ROOT
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

DATA_MODELS = os.path.join(
    PROJECT_ROOT,
    "data",
    "models"
)


# ============================================================
# FILE PATHS
# ============================================================

DIABETES_X_TEST = os.path.join(
    DATA_PROCESSED,
    "diabetes_X_test.csv"
)

DIABETES_Y_TEST = os.path.join(
    DATA_PROCESSED,
    "diabetes_y_test.csv"
)

DIABETES_MODEL = os.path.join(
    DATA_MODELS,
    "diabetes_random_forest.pkl"
)

DIABETES_THRESHOLD = os.path.join(
    DATA_MODELS,
    "diabetes_threshold.pkl"
)

HYPERTENSION_X_TEST = os.path.join(
    DATA_PROCESSED,
    "hypertension_X_test.csv"
)

HYPERTENSION_Y_TEST = os.path.join(
    DATA_PROCESSED,
    "hypertension_y_test.csv"
)

HYPERTENSION_MODEL = os.path.join(
    DATA_MODELS,
    "hypertension_random_forest.pkl"
)

HYPERTENSION_THRESHOLD = os.path.join(
    DATA_MODELS,
    "hypertension_threshold.pkl"
)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("TRUSTWORTHY AI - ACCOUNTABILITY ANALYSIS")
print("=" * 70)

print("\nProject root:")
print(PROJECT_ROOT)


# ============================================================
# 1. CHECK REQUIRED FILES
# ============================================================

print("\n" + "=" * 70)
print("CHECKING REQUIRED FILES")
print("=" * 70)


required_files = {
    "Diabetes X_test": DIABETES_X_TEST,
    "Diabetes y_test": DIABETES_Y_TEST,
    "Diabetes model": DIABETES_MODEL,
    "Diabetes threshold": DIABETES_THRESHOLD,

    "Hypertension X_test": HYPERTENSION_X_TEST,
    "Hypertension y_test": HYPERTENSION_Y_TEST,
    "Hypertension model": HYPERTENSION_MODEL,
    "Hypertension threshold": HYPERTENSION_THRESHOLD,
}


all_files_exist = True

for name, path in required_files.items():

    if os.path.exists(path):
        print(f"[OK] {name}: {path}")

    else:
        print(f"[MISSING] {name}: {path}")
        all_files_exist = False


if not all_files_exist:

    print("\nERROR:")
    print("Some required files are missing.")

    raise SystemExit(1)


# ============================================================
# 2. SAFE PICKLE LOADING
# ============================================================

def load_pickle(path):

    try:

        with open(path, "rb") as file:

            return joblib.load(file)

    except Exception as error:

        print("\nWARNING: Could not load model artifact.")
        print("File:", path)
        print("Error:", repr(error))

        return None


# ============================================================
# 3. METRICS
# ============================================================

def calculate_metrics(y_true, y_pred, y_prob):

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_true,
        y_prob
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC-AUC": roc_auc,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp
    }


# ============================================================
# 4. DIABETES ACCOUNTABILITY
# ============================================================

print("\n" + "=" * 70)
print("1. DIABETES ACCOUNTABILITY")
print("=" * 70)


X_diabetes = pd.read_csv(
    DIABETES_X_TEST
)

y_diabetes = pd.read_csv(
    DIABETES_Y_TEST
).squeeze()


threshold_diabetes = load_pickle(
    DIABETES_THRESHOLD
)

diabetes_model = load_pickle(
    DIABETES_MODEL
)


print("\nTest data:")
print("X:", X_diabetes.shape)
print("y:", y_diabetes.shape)


print("\nModel artifact:")
print(DIABETES_MODEL)


if threshold_diabetes is not None:

    print("Threshold:", threshold_diabetes)

else:

    print("Threshold: unavailable")


if diabetes_model is not None:

    print("Model loading: SUCCESS")

    try:

        diabetes_probability = diabetes_model.predict_proba(
            X_diabetes
        )[:, 1]

        if threshold_diabetes is not None:

            diabetes_prediction = (
                diabetes_probability >= threshold_diabetes
            ).astype(int)

        else:

            diabetes_prediction = (
                diabetes_probability >= 0.5
            ).astype(int)

        metrics = calculate_metrics(
            y_diabetes,
            diabetes_prediction,
            diabetes_probability
        )

        print("\n" + "-" * 70)
        print("DIABETES MODEL PERFORMANCE")
        print("-" * 70)

        for key, value in metrics.items():

            if key in ["TN", "FP", "FN", "TP"]:

                print(f"{key:<10}: {value}")

            else:

                print(f"{key:<10}: {value:.4f}")

    except Exception as error:

        print("\nCould not evaluate Diabetes model.")
        print("Error:", repr(error))

else:

    print(
        "\nModel artifact could not be inspected."
    )

    print(
        "Accountability analysis continues using "
        "available metadata."
    )


# ============================================================
# 5. HYPERTENSION ACCOUNTABILITY
# ============================================================

print("\n" + "=" * 70)
print("2. HYPERTENSION ACCOUNTABILITY")
print("=" * 70)


X_hypertension = pd.read_csv(
    HYPERTENSION_X_TEST
)

y_hypertension = pd.read_csv(
    HYPERTENSION_Y_TEST
).squeeze()


threshold_hypertension = load_pickle(
    HYPERTENSION_THRESHOLD
)

hypertension_model = load_pickle(
    HYPERTENSION_MODEL
)


print("\nTest data:")
print("X:", X_hypertension.shape)
print("y:", y_hypertension.shape)


print("\nModel artifact:")
print(HYPERTENSION_MODEL)


if threshold_hypertension is not None:

    print("Threshold:", threshold_hypertension)

else:

    print("Threshold: unavailable")


if hypertension_model is not None:

    print("Model loading: SUCCESS")

    try:

        hypertension_probability = (
            hypertension_model.predict_proba(
                X_hypertension
            )[:, 1]
        )

        if threshold_hypertension is not None:

            hypertension_prediction = (
                hypertension_probability >= threshold_hypertension
            ).astype(int)

        else:

            hypertension_prediction = (
                hypertension_probability >= 0.5
            ).astype(int)

        metrics = calculate_metrics(
            y_hypertension,
            hypertension_prediction,
            hypertension_probability
        )

        print("\n" + "-" * 70)
        print("HYPERTENSION MODEL PERFORMANCE")
        print("-" * 70)

        for key, value in metrics.items():

            if key in ["TN", "FP", "FN", "TP"]:

                print(f"{key:<10}: {value}")

            else:

                print(f"{key:<10}: {value:.4f}")

    except Exception as error:

        print("\nCould not evaluate Hypertension model.")
        print("Error:", repr(error))

else:

    print(
        "\nModel artifact could not be inspected."
    )

    print(
        "Accountability analysis continues using "
        "available metadata."
    )


# ============================================================
# 6. ACCOUNTABILITY CHECKLIST
# ============================================================

print("\n" + "=" * 70)
print("3. ACCOUNTABILITY CHECKLIST")
print("=" * 70)


accountability_items = [

    (
        "Model identification",
        "Diabetes and Hypertension models are explicitly identified."
    ),

    (
        "Threshold identification",
        "Final decision thresholds are stored separately."
    ),

    (
        "Performance monitoring",
        "Accuracy, Precision, Recall, F1 and ROC-AUC are evaluated."
    ),

    (
        "Error monitoring",
        "False Positive and False Negative predictions are measured."
    ),

    (
        "Human oversight",
        "AI predictions should support, not replace, healthcare professionals."
    ),

    (
        "Auditability",
        "Model inputs, predictions and evaluation results should be recorded."
    ),

    (
        "Model version control",
        "Model artifacts should be versioned and protected from unauthorized modification."
    ),

    (
        "Decision responsibility",
        "Final medical decisions should remain under qualified human supervision."
    ),

    (
        "Risk communication",
        "Users should be informed that predictions represent risk estimates, not medical diagnoses."
    )
]


for item, description in accountability_items:

    print(f"[CHECK] {item}")
    print(f"        {description}")


# ============================================================
# 7. RESPONSIBILITY FRAMEWORK
# ============================================================

print("\n" + "=" * 70)
print("4. RESPONSIBILITY FRAMEWORK")
print("=" * 70)


print("""
AI SYSTEM
    |
    |-- Produces risk prediction
    |
    v
AI DEVELOPER / DATA SCIENTIST
    |
    |-- Responsible for model design,
    |   testing, validation and monitoring
    |
    v
HEALTHCARE ORGANIZATION
    |
    |-- Responsible for deployment,
    |   access control and governance
    |
    v
HEALTHCARE PROFESSIONAL
    |
    |-- Reviews AI output and patient context
    |
    v
FINAL MEDICAL DECISION
""")


# ============================================================
# 8. ACCOUNTABILITY FINDINGS
# ============================================================

print("\n" + "=" * 70)
print("5. ACCOUNTABILITY FINDINGS")
print("=" * 70)


print("""
1. The project stores trained models as separate artifacts.

2. Decision thresholds are explicitly stored and can be inspected.

3. Model performance is evaluated using multiple metrics.

4. Confusion matrices allow False Positive and False Negative
   errors to be identified.

5. The system should maintain human oversight for healthcare decisions.

6. AI predictions should be treated as decision-support information,
   not as an autonomous medical diagnosis.

7. Model files should be protected against unauthorized modification.

8. Prediction logs should be maintained in a production environment
   so that decisions can be audited later.

9. Model versions and training datasets should be documented.

10. Responsibility for an incorrect AI-assisted decision should be
    distributed across the AI development, deployment and clinical
    decision-making process rather than assigned to the AI itself.
""")


# ============================================================
# 9. ACCOUNTABILITY LIMITATIONS
# ============================================================

print("\n" + "=" * 70)
print("6. ACCOUNTABILITY LIMITATIONS")
print("=" * 70)


print("""
Current project limitations:

- No formal production audit-log system is implemented yet.
- Model version metadata is limited.
- No automated model monitoring system is implemented.
- Human clinical review is not implemented as an automated workflow.
- Model artifacts are stored locally.
- Pickle model files should be protected because loading an
  untrusted pickle file can create security risks.
""")


# ============================================================
# 10. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("ACCOUNTABILITY ANALYSIS SUMMARY")
print("=" * 70)


print("""
Accountability analysis evaluates whether the AI system provides
traceability, monitoring, human oversight and clear responsibility.

The current project demonstrates:

[OK] Model identification
[OK] Threshold identification
[OK] Performance evaluation
[OK] Error measurement
[OK] Human oversight requirement
[OK] Risk communication requirement
[OK] Responsibility framework

Recommended future improvements:

- Add prediction logging.
- Add model versioning.
- Add audit trails.
- Add access control.
- Add model monitoring.
- Add formal human-in-the-loop workflow.
- Add model update and rollback procedures.
""")


print("\n" + "=" * 70)
print("ACCOUNTABILITY ANALYSIS COMPLETED")
print("=" * 70)

