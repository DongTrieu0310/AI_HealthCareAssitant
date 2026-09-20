import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


print("=" * 70)
print("TRUSTWORTHY AI - FAIRNESS ANALYSIS")
print("=" * 70)


# ============================================================
# 1. PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_MODELS = PROJECT_ROOT / "data" / "models"

FAIRNESS_OUTPUT = DATA_MODELS / "fairness"
FAIRNESS_OUTPUT.mkdir(
    parents=True,
    exist_ok=True
)


print("\nProject root:")
print(PROJECT_ROOT)

print("\nFairness output directory:")
print(FAIRNESS_OUTPUT)


# ============================================================
# 2. FILE PATHS
# ============================================================

DIABETES_X_TEST = DATA_PROCESSED / "diabetes_X_test.csv"
DIABETES_Y_TEST = DATA_PROCESSED / "diabetes_y_test.csv"

HYPERTENSION_X_TEST = DATA_PROCESSED / "hypertension_X_test.csv"
HYPERTENSION_Y_TEST = DATA_PROCESSED / "hypertension_y_test.csv"

DIABETES_MODEL = DATA_MODELS / "diabetes_random_forest.pkl"
DIABETES_THRESHOLD = DATA_MODELS / "diabetes_threshold.pkl"

HYPERTENSION_MODEL = DATA_MODELS / "hypertension_random_forest.pkl"
HYPERTENSION_THRESHOLD = DATA_MODELS / "hypertension_threshold.pkl"


# ============================================================
# 3. CHECK FILES
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
    "Hypertension threshold": HYPERTENSION_THRESHOLD
}


missing_files = []


for name, path in required_files.items():

    if path.exists():

        print(f"[OK] {name}: {path}")

    else:

        print(f"[MISSING] {name}: {path}")

        missing_files.append(name)


if missing_files:

    print("\nERROR: Required files are missing.")

    for item in missing_files:
        print("-", item)

    raise FileNotFoundError(
        "Required files are missing."
    )


# ============================================================
# 4. METRIC FUNCTION
# ============================================================

def calculate_metrics(y_true, y_pred):

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if len(y_true) == 0:

        return {
            "Accuracy": np.nan,
            "Precision": np.nan,
            "Recall": np.nan,
            "F1": np.nan,
            "Prediction_Rate": np.nan,
            "False_Positive_Rate": np.nan,
            "False_Negative_Rate": np.nan
        }

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

    prediction_rate = np.mean(
        y_pred == 1
    )

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()

    if (fp + tn) > 0:

        false_positive_rate = fp / (fp + tn)

    else:

        false_positive_rate = np.nan

    if (fn + tp) > 0:

        false_negative_rate = fn / (fn + tp)

    else:

        false_negative_rate = np.nan

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Prediction_Rate": prediction_rate,
        "False_Positive_Rate": false_positive_rate,
        "False_Negative_Rate": false_negative_rate
    }


# ============================================================
# 5. FAIRNESS GAP FUNCTION
# ============================================================

def calculate_fairness_gaps(results):

    columns = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "Prediction_Rate",
        "False_Positive_Rate",
        "False_Negative_Rate"
    ]

    gaps = {}

    for column in columns:

        values = results[column].dropna()

        if len(values) >= 2:

            gaps[column] = (
                values.max() -
                values.min()
            )

        else:

            gaps[column] = np.nan

    return gaps


# ============================================================
# 6. PRINT FAIRNESS RESULTS
# ============================================================

def print_fairness_results(
        results,
        gaps,
        title
):

    print("\n")
    print("-" * 70)
    print(title)
    print("-" * 70)

    display_columns = [
        "Group",
        "Samples",
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "Prediction_Rate",
        "False_Positive_Rate",
        "False_Negative_Rate"
    ]

    print(
        results[display_columns].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    print("\nFairness gaps:")

    print(
        f"Accuracy Gap             : "
        f"{gaps['Accuracy']:.4f}"
    )

    print(
        f"Precision Gap            : "
        f"{gaps['Precision']:.4f}"
    )

    print(
        f"Recall Gap               : "
        f"{gaps['Recall']:.4f}"
    )

    print(
        f"F1 Gap                   : "
        f"{gaps['F1']:.4f}"
    )

    print(
        f"Prediction Rate Gap      : "
        f"{gaps['Prediction_Rate']:.4f}"
    )

    print(
        f"False Positive Rate Gap  : "
        f"{gaps['False_Positive_Rate']:.4f}"
    )

    print(
        f"False Negative Rate Gap  : "
        f"{gaps['False_Negative_Rate']:.4f}"
    )


# ============================================================
# 7. SAVE RESULTS
# ============================================================

def save_fairness_results(
        results,
        gaps,
        filename
):

    output_path = FAIRNESS_OUTPUT / filename

    results.to_csv(
        output_path,
        index=False
    )

    gap_path = FAIRNESS_OUTPUT / (
        filename.replace(
            ".csv",
            "_gaps.csv"
        )
    )

    gap_df = pd.DataFrame(
        [{
            "Metric": key,
            "Gap": value
        }
        for key, value in gaps.items()]
    )

    gap_df.to_csv(
        gap_path,
        index=False
    )

    print(
        f"\nResults saved: {output_path}"
    )

    print(
        f"Gaps saved   : {gap_path}"
    )


# ============================================================
# 8. DIABETES AGE GROUP
# ============================================================

def create_diabetes_age_groups(age_series):

    """
    Diabetes X_test is standardized.

    Therefore:
    We must NOT use raw thresholds such as
    age < 30 or age < 50.

    Instead, use test-set quantiles.
    """

    age_series = pd.to_numeric(
        age_series,
        errors="coerce"
    )

    if age_series.isna().all():

        raise ValueError(
            "Diabetes Age column contains no valid numeric values."
        )

    try:

        groups = pd.qcut(
            age_series,
            q=3,
            labels=[
                "Youngest tertile",
                "Middle tertile",
                "Oldest tertile"
            ],
            duplicates="drop"
        )

    except ValueError:

        groups = pd.cut(
            age_series,
            bins=3,
            labels=[
                "Youngest group",
                "Middle group",
                "Oldest group"
            ]
        )

    return groups


# ============================================================
# 9. DIABETES FAIRNESS
# ============================================================

print("\n")
print("=" * 70)
print("1. DIABETES FAIRNESS")
print("=" * 70)


X_diabetes = pd.read_csv(
    DIABETES_X_TEST
)

y_diabetes = pd.read_csv(
    DIABETES_Y_TEST
).squeeze()


print("\nDiabetes test data:")
print("X:", X_diabetes.shape)
print("y:", y_diabetes.shape)


# ------------------------------------------------------------
# Load model
# ------------------------------------------------------------

diabetes_model = joblib.load(
    DIABETES_MODEL
)

diabetes_threshold = float(
    joblib.load(
        DIABETES_THRESHOLD
    )
)


print(
    f"Diabetes threshold: "
    f"{diabetes_threshold:.4f}"
)


# ------------------------------------------------------------
# Validate Age
# ------------------------------------------------------------

if "Age" not in X_diabetes.columns:

    raise ValueError(
        "Diabetes X_test does not contain 'Age'."
    )


# ------------------------------------------------------------
# Prediction
# ------------------------------------------------------------

diabetes_probability = (
    diabetes_model
    .predict_proba(X_diabetes)[:, 1]
)

diabetes_prediction = (
    diabetes_probability >=
    diabetes_threshold
).astype(int)


# ------------------------------------------------------------
# Build dataframe
# ------------------------------------------------------------

diabetes_fairness = X_diabetes.copy()

diabetes_fairness["Actual"] = (
    y_diabetes.values
)

diabetes_fairness["Prediction"] = (
    diabetes_prediction
)


# ------------------------------------------------------------
# IMPORTANT:
# Age is standardized.
# Use quantile-based groups.
# ------------------------------------------------------------

diabetes_fairness["AgeGroup"] = (
    create_diabetes_age_groups(
        diabetes_fairness["Age"]
    )
)


print(
    "\n[INFO] Diabetes Age is standardized."
)

print(
    "[INFO] Age fairness uses test-set tertiles "
    "instead of raw age thresholds."
)


# ------------------------------------------------------------
# Diabetes Age Fairness
# ------------------------------------------------------------

diabetes_results = []


for group in diabetes_fairness[
    "AgeGroup"
].dropna().unique():

    group_data = diabetes_fairness[
        diabetes_fairness["AgeGroup"] == group
    ]

    print(
        f"\nDiabetes age group: {group}"
    )

    print(
        f"Samples: {len(group_data)}"
    )

    if len(group_data) < 10:

        print(
            "WARNING: Group contains fewer than 10 samples."
        )

    metrics = calculate_metrics(
        group_data["Actual"],
        group_data["Prediction"]
    )

    diabetes_results.append({
        "Group": str(group),
        "Samples": len(group_data),
        **metrics
    })


diabetes_results = pd.DataFrame(
    diabetes_results
)


if len(diabetes_results) >= 2:

    diabetes_gaps = calculate_fairness_gaps(
        diabetes_results
    )

    print_fairness_results(
        diabetes_results,
        diabetes_gaps,
        "DIABETES - AGE FAIRNESS"
    )

    save_fairness_results(
        diabetes_results,
        diabetes_gaps,
        "diabetes_age_fairness.csv"
    )

else:

    diabetes_gaps = {}

    print(
        "\n[WARNING] Insufficient Diabetes age groups."
    )


# ============================================================
# 10. HYPERTENSION FAIRNESS
# ============================================================

print("\n")
print("=" * 70)
print("2. HYPERTENSION FAIRNESS")
print("=" * 70)


X_hypertension = pd.read_csv(
    HYPERTENSION_X_TEST
)

y_hypertension = pd.read_csv(
    HYPERTENSION_Y_TEST
).squeeze()


print("\nHypertension test data:")
print("X:", X_hypertension.shape)
print("y:", y_hypertension.shape)


# ------------------------------------------------------------
# Load model
# ------------------------------------------------------------

hypertension_model = joblib.load(
    HYPERTENSION_MODEL
)

hypertension_threshold = float(
    joblib.load(
        HYPERTENSION_THRESHOLD
    )
)


print(
    f"Hypertension threshold: "
    f"{hypertension_threshold:.4f}"
)


# ------------------------------------------------------------
# Prediction
# ------------------------------------------------------------

hypertension_probability = (
    hypertension_model
    .predict_proba(X_hypertension)[:, 1]
)

hypertension_prediction = (
    hypertension_probability >=
    hypertension_threshold
).astype(int)


# ------------------------------------------------------------
# Build dataframe
# ------------------------------------------------------------

hypertension_fairness = (
    X_hypertension.copy()
)

hypertension_fairness["Actual"] = (
    y_hypertension.values
)

hypertension_fairness["Prediction"] = (
    hypertension_prediction
)


# ------------------------------------------------------------
# Validate demographic columns
# ------------------------------------------------------------

if "male" not in hypertension_fairness.columns:

    raise ValueError(
        "Hypertension X_test does not contain 'male'."
    )

if "age" not in hypertension_fairness.columns:

    raise ValueError(
        "Hypertension X_test does not contain 'age'."
    )


# ------------------------------------------------------------
# Create age groups
# ------------------------------------------------------------

hypertension_fairness["AgeGroup"] = (
    pd.cut(
        hypertension_fairness["age"],
        bins=[-np.inf, 29, 49, np.inf],
        labels=[
            "Under 30",
            "30-49",
            "50+"
        ]
    )
)


# ============================================================
# 11. HYPERTENSION GENDER FAIRNESS
# ============================================================

print("\n")
print("-" * 70)
print("HYPERTENSION - GENDER FAIRNESS")
print("-" * 70)


gender_results = []


gender_mapping = {
    0: "Female",
    1: "Male"
}


for gender_value, gender_name in gender_mapping.items():

    group_data = hypertension_fairness[
        hypertension_fairness["male"] ==
        gender_value
    ]

    print(
        f"\nGroup: {gender_name}"
    )

    print(
        f"Samples: {len(group_data)}"
    )

    if len(group_data) == 0:

        print(
            "No samples in this group. Skipped."
        )

        continue

    metrics = calculate_metrics(
        group_data["Actual"],
        group_data["Prediction"]
    )

    gender_results.append({
        "Group": gender_name,
        "Samples": len(group_data),
        **metrics
    })


gender_results = pd.DataFrame(
    gender_results
)


if len(gender_results) >= 2:

    gender_gaps = calculate_fairness_gaps(
        gender_results
    )

    print_fairness_results(
        gender_results,
        gender_gaps,
        "HYPERTENSION - GENDER FAIRNESS RESULTS"
    )

    save_fairness_results(
        gender_results,
        gender_gaps,
        "hypertension_gender_fairness.csv"
    )

else:

    gender_gaps = {}


# ============================================================
# 12. HYPERTENSION AGE FAIRNESS
# ============================================================

print("\n")
print("-" * 70)
print("HYPERTENSION - AGE FAIRNESS")
print("-" * 70)


hypertension_age_results = []


for group in [
    "Under 30",
    "30-49",
    "50+"
]:

    group_data = hypertension_fairness[
        hypertension_fairness["AgeGroup"]
        .astype(str) == group
    ]

    print(
        f"\nAge group: {group}"
    )

    print(
        f"Samples: {len(group_data)}"
    )

    if len(group_data) == 0:

        print(
            "No samples in this group. Skipped."
        )

        continue

    metrics = calculate_metrics(
        group_data["Actual"],
        group_data["Prediction"]
    )

    hypertension_age_results.append({
        "Group": group,
        "Samples": len(group_data),
        **metrics
    })


hypertension_age_results = pd.DataFrame(
    hypertension_age_results
)


if len(hypertension_age_results) >= 2:

    hypertension_age_gaps = (
        calculate_fairness_gaps(
            hypertension_age_results
        )
    )

    print_fairness_results(
        hypertension_age_results,
        hypertension_age_gaps,
        "HYPERTENSION - AGE FAIRNESS RESULTS"
    )

    save_fairness_results(
        hypertension_age_results,
        hypertension_age_gaps,
        "hypertension_age_fairness.csv"
    )

else:

    hypertension_age_gaps = {}


# ============================================================
# 13. FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("FAIRNESS ANALYSIS SUMMARY")
print("=" * 70)


print(
    """
Diabetes:
- Gender fairness: NOT AVAILABLE
- Reason: Diabetes dataset does not contain a gender attribute.
- Age-group fairness: Evaluated using test-set age tertiles.
- Raw age thresholds were NOT used because Diabetes X_test
  contains standardized age values.

Hypertension:
- Gender fairness: Evaluated.
- Age-group fairness: Evaluated using real age values.

Fairness metrics:
- Accuracy
- Precision
- Recall
- F1
- Prediction Rate
- False Positive Rate
- False Negative Rate

Fairness gap:
Difference between the highest and lowest
group-level metric.

Important:
A fairness gap does not automatically prove
that a model is unfair or discriminatory.

It indicates that model performance or error
rates differ between groups.

Fairness results should be interpreted together
with Bias, Robustness, Explainability, and
overall model performance.
"""
)


# ============================================================
# 14. FINAL STATUS
# ============================================================

print("=" * 70)
print("FAIRNESS ANALYSIS COMPLETED")
print("=" * 70)

print("\nOutput files:")

for file in sorted(
    FAIRNESS_OUTPUT.glob("*.csv")
):

    print(
        f"[OK] {file.name}"
    )

print(
    "\nProcess finished successfully."
)