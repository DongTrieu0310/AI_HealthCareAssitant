import os
import joblib
import pandas as pd
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


print("=" * 70)
print("TRUSTWORTHY AI - BIAS ANALYSIS")
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

DATA_MODELS = os.path.join(
    PROJECT_ROOT,
    "data",
    "models"
)

BIAS_OUTPUT = os.path.join(
    DATA_MODELS,
    "bias"
)

os.makedirs(
    BIAS_OUTPUT,
    exist_ok=True
)


print("\nProject root:")
print(PROJECT_ROOT)

print("\nBias output directory:")
print(BIAS_OUTPUT)


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
# 3. CHECK REQUIRED FILES
# ============================================================

print("\n")
print("=" * 70)
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

    if os.path.exists(path):

        print(f"[OK] {name}: {path}")

    else:

        print(f"[MISSING] {name}: {path}")

        missing_files.append(path)


if missing_files:

    print("\nERROR: Required files are missing.")

    raise FileNotFoundError(
        "Please check the paths above."
    )


# ============================================================
# 4. LOAD PICKLE
# ============================================================

def load_pickle(path):

    try:

        return joblib.load(path)

    except Exception as e:

        print("\nERROR: Cannot load serialized file.")
        print("Path:", path)
        print("Reason:", e)

        raise


# ============================================================
# 5. GENERAL BIAS METRICS
# ============================================================

def calculate_bias_metrics(
        y_true,
        y_pred
):

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if len(y_true) == 0:

        return None

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

    actual_positive_rate = np.mean(
        y_true == 1
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    false_positive_rate = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else 0.0
    )

    false_negative_rate = (
        fn / (fn + tp)
        if (fn + tp) > 0
        else 0.0
    )

    return {
        "Samples": len(y_true),
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Prediction Rate": prediction_rate,
        "Actual Positive Rate": actual_positive_rate,
        "False Positive Rate": false_positive_rate,
        "False Negative Rate": false_negative_rate
    }


# ============================================================
# 6. CLASS DISTRIBUTION
# ============================================================

def analyze_class_distribution(
        y,
        dataset_name
):

    y = pd.Series(y)

    counts = (
        y.value_counts()
        .sort_index()
    )

    percentages = (
        y.value_counts(
            normalize=True
        )
        .sort_index()
        * 100
    )

    result = pd.DataFrame({
        "Class": counts.index,
        "Count": counts.values,
        "Percentage": percentages.values
    })

    print("\n")
    print("-" * 70)
    print(
        f"{dataset_name.upper()} - CLASS DISTRIBUTION"
    )
    print("-" * 70)

    print(
        result.to_string(
            index=False
        )
    )

    imbalance_ratio = (
        counts.max() / counts.min()
        if counts.min() > 0
        else np.inf
    )

    print(
        f"\nClass imbalance ratio: "
        f"{imbalance_ratio:.2f}"
    )

    if imbalance_ratio < 1.5:

        print(
            "Interpretation: "
            "Relatively balanced classes."
        )

    elif imbalance_ratio < 3:

        print(
            "Interpretation: "
            "Moderate class imbalance."
        )

    else:

        print(
            "Interpretation: "
            "Strong class imbalance."
        )

    return result


# ============================================================
# 7. FEATURE DISTRIBUTION
# ============================================================

def analyze_feature_distribution(
        X,
        dataset_name
):

    print("\n")
    print("-" * 70)
    print(
        f"{dataset_name.upper()} - FEATURE DISTRIBUTION"
    )
    print("-" * 70)

    numeric_columns = X.select_dtypes(
        include=np.number
    ).columns

    statistics = X[
        numeric_columns
    ].describe().T

    statistics = statistics[
        [
            "mean",
            "std",
            "min",
            "25%",
            "50%",
            "75%",
            "max"
        ]
    ]

    print(
        statistics.round(3).to_string()
    )

    return statistics


# ============================================================
# 8. PREDICTION BIAS
# ============================================================

def analyze_predictions(
        X,
        y,
        model,
        threshold,
        dataset_name
):

    probability = model.predict_proba(
        X
    )[:, 1]

    prediction = (
        probability >= threshold
    ).astype(int)

    metrics = calculate_bias_metrics(
        y,
        prediction
    )

    print("\n")
    print("-" * 70)
    print(
        f"{dataset_name.upper()} - PREDICTION BIAS"
    )
    print("-" * 70)

    print(
        f"Prediction threshold : "
        f"{threshold:.2f}"
    )

    print(
        f"Actual positive rate  : "
        f"{metrics['Actual Positive Rate']:.4f}"
    )

    print(
        f"Predicted positive rate: "
        f"{metrics['Prediction Rate']:.4f}"
    )

    print(
        f"False Positive Rate    : "
        f"{metrics['False Positive Rate']:.4f}"
    )

    print(
        f"False Negative Rate    : "
        f"{metrics['False Negative Rate']:.4f}"
    )

    print("\nPerformance:")

    print(
        f"Accuracy : "
        f"{metrics['Accuracy']:.4f}"
    )

    print(
        f"Precision: "
        f"{metrics['Precision']:.4f}"
    )

    print(
        f"Recall   : "
        f"{metrics['Recall']:.4f}"
    )

    print(
        f"F1       : "
        f"{metrics['F1']:.4f}"
    )

    prediction_distribution = pd.DataFrame({
        "Prediction": [0, 1],
        "Count": [
            np.sum(prediction == 0),
            np.sum(prediction == 1)
        ]
    })

    prediction_distribution[
        "Percentage"
    ] = (
        prediction_distribution["Count"]
        / len(prediction)
        * 100
    )

    print("\nPrediction distribution:")

    print(
        prediction_distribution.to_string(
            index=False
        )
    )

    return prediction, probability


# ============================================================
# 9. GROUP BIAS ANALYSIS
# ============================================================

def analyze_group_bias(
        X,
        y,
        prediction,
        group_column,
        group_labels,
        dataset_name,
        output_name
):

    if group_column not in X.columns:

        print(
            f"\n[SKIPPED] "
            f"{group_column} not available."
        )

        return None, None

    data = X.copy()

    data["Actual"] = np.asarray(y)

    data["Prediction"] = np.asarray(
        prediction
    )

    print("\n")
    print("-" * 70)
    print(
        f"{dataset_name.upper()} - "
        f"{group_column.upper()} BIAS"
    )
    print("-" * 70)

    rows = []

    for group_value, group_name in group_labels:

        group_data = data[
            data[group_column] == group_value
        ]

        print(
            f"\nGroup: {group_name}"
        )

        print(
            f"Samples: {len(group_data)}"
        )

        if len(group_data) == 0:

            print(
                "No samples. Skipped."
            )

            continue

        metrics = calculate_bias_metrics(
            group_data["Actual"],
            group_data["Prediction"]
        )

        metrics["Group"] = group_name

        rows.append(metrics)

    if not rows:

        print(
            "No valid groups available."
        )

        return None, None

    result = pd.DataFrame(rows)

    result = result[
        [
            "Group",
            "Samples",
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
            "Prediction Rate",
            "Actual Positive Rate",
            "False Positive Rate",
            "False Negative Rate"
        ]
    ]

    print("\nGroup-level results:")

    print(
        result.round(4).to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Calculate gaps
    # --------------------------------------------------------

    gap_rows = []

    print("\nBias gaps:")

    for metric in [
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "Prediction Rate",
        "Actual Positive Rate",
        "False Positive Rate",
        "False Negative Rate"
    ]:

        gap = (
            result[metric].max()
            - result[metric].min()
        )

        print(
            f"{metric} Gap: "
            f"{gap:.4f}"
        )

        gap_rows.append({
            "Metric": metric,
            "Gap": gap
        })

    gaps = pd.DataFrame(
        gap_rows
    )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    result_path = os.path.join(
        BIAS_OUTPUT,
        f"{output_name}.csv"
    )

    gap_path = os.path.join(
        BIAS_OUTPUT,
        f"{output_name}_gaps.csv"
    )

    result.to_csv(
        result_path,
        index=False
    )

    gaps.to_csv(
        gap_path,
        index=False
    )

    print(
        f"\nResults saved: {result_path}"
    )

    print(
        f"Gaps saved   : {gap_path}"
    )

    return result, gaps


# ============================================================
# 10. DIABETES BIAS ANALYSIS
# ============================================================

print("\n")
print("=" * 70)
print("1. DIABETES BIAS ANALYSIS")
print("=" * 70)


X_diabetes = pd.read_csv(
    DIABETES_X_TEST
)

y_diabetes = pd.read_csv(
    DIABETES_Y_TEST
).squeeze()


diabetes_model = load_pickle(
    DIABETES_MODEL
)

diabetes_threshold = float(
    load_pickle(
        DIABETES_THRESHOLD
    )
)


print("\nDiabetes data:")

print(
    f"X_test: {X_diabetes.shape}"
)

print(
    f"y_test: {y_diabetes.shape}"
)

print(
    f"Threshold: {diabetes_threshold:.4f}"
)


# ============================================================
# Diabetes class distribution
# ============================================================

analyze_class_distribution(
    y_diabetes,
    "Diabetes"
)


# ============================================================
# Diabetes feature distribution
# ============================================================

analyze_feature_distribution(
    X_diabetes,
    "Diabetes"
)


# ============================================================
# Diabetes prediction bias
# ============================================================

(
    diabetes_prediction,
    diabetes_probability
) = analyze_predictions(
    X_diabetes,
    y_diabetes,
    diabetes_model,
    diabetes_threshold,
    "Diabetes"
)


# ============================================================
# Diabetes AGE BIAS
# IMPORTANT:
# Diabetes Age is standardized.
# Therefore raw age boundaries such as 30/50
# MUST NOT be used.
#
# We use test-set tertiles instead.
# ============================================================

if "Age" in X_diabetes.columns:

    print("\n")
    print("-" * 70)
    print("DIABETES - AGE GROUP BIAS")
    print("-" * 70)

    print(
        "[INFO] Diabetes Age is standardized."
    )

    print(
        "[INFO] Age bias uses test-set tertiles."
    )

    diabetes_age = X_diabetes["Age"]

    # --------------------------------------------------------
    # Create tertiles
    # --------------------------------------------------------

    try:

        age_group = pd.qcut(
            diabetes_age,
            q=3,
            labels=[
                "Youngest tertile",
                "Middle tertile",
                "Oldest tertile"
            ],
            duplicates="drop"
        )

    except ValueError:

        print(
            "[WARNING] Unable to create age tertiles."
        )

        age_group = None


    if age_group is not None:

        X_diabetes_bias = X_diabetes.copy()

        X_diabetes_bias[
            "AgeGroup"
        ] = age_group

        # ----------------------------------------------------
        # Group labels
        # ----------------------------------------------------

        diabetes_age_labels = [
            (
                "Youngest tertile",
                "Youngest tertile"
            ),
            (
                "Middle tertile",
                "Middle tertile"
            ),
            (
                "Oldest tertile",
                "Oldest tertile"
            )
        ]

        diabetes_age_result, diabetes_age_gaps = (
            analyze_group_bias(
                X_diabetes_bias,
                y_diabetes,
                diabetes_prediction,
                "AgeGroup",
                diabetes_age_labels,
                "Diabetes",
                "diabetes_age_bias"
            )
        )

else:

    print(
        "\n[SKIPPED] "
        "Diabetes dataset does not contain Age."
    )


# ============================================================
# 11. HYPERTENSION BIAS ANALYSIS
# ============================================================

print("\n")
print("=" * 70)
print("2. HYPERTENSION BIAS ANALYSIS")
print("=" * 70)


X_hypertension = pd.read_csv(
    HYPERTENSION_X_TEST
)

y_hypertension = pd.read_csv(
    HYPERTENSION_Y_TEST
).squeeze()


hypertension_model = load_pickle(
    HYPERTENSION_MODEL
)

hypertension_threshold = float(
    load_pickle(
        HYPERTENSION_THRESHOLD
    )
)


print("\nHypertension data:")

print(
    f"X_test: {X_hypertension.shape}"
)

print(
    f"y_test: {y_hypertension.shape}"
)

print(
    f"Threshold: "
    f"{hypertension_threshold:.4f}"
)


# ============================================================
# Hypertension class distribution
# ============================================================

analyze_class_distribution(
    y_hypertension,
    "Hypertension"
)


# ============================================================
# Hypertension feature distribution
# ============================================================

analyze_feature_distribution(
    X_hypertension,
    "Hypertension"
)


# ============================================================
# Hypertension prediction bias
# ============================================================

(
    hypertension_prediction,
    hypertension_probability
) = analyze_predictions(
    X_hypertension,
    y_hypertension,
    hypertension_model,
    hypertension_threshold,
    "Hypertension"
)


# ============================================================
# Hypertension GENDER BIAS
# ============================================================

hypertension_gender_result, hypertension_gender_gaps = (
    analyze_group_bias(
        X_hypertension,
        y_hypertension,
        hypertension_prediction,
        "male",
        [
            (0, "Female"),
            (1, "Male")
        ],
        "Hypertension",
        "hypertension_gender_bias"
    )
)


# ============================================================
# Hypertension AGE BIAS
# ============================================================

if "age" in X_hypertension.columns:

    print("\n")
    print("-" * 70)
    print("HYPERTENSION - AGE GROUP BIAS")
    print("-" * 70)

    print(
        "[INFO] Hypertension Age uses real age values."
    )

    X_hypertension_bias = (
        X_hypertension.copy()
    )

    X_hypertension_bias[
        "AgeGroup"
    ] = pd.cut(
        X_hypertension["age"],
        bins=[
            -np.inf,
            29,
            49,
            np.inf
        ],
        labels=[
            "Under 30",
            "30-49",
            "50+"
        ]
    )

    hypertension_age_labels = [
        ("Under 30", "Under 30"),
        ("30-49", "30-49"),
        ("50+", "50+")
    ]

    (
        hypertension_age_result,
        hypertension_age_gaps
    ) = analyze_group_bias(
        X_hypertension_bias,
        y_hypertension,
        hypertension_prediction,
        "AgeGroup",
        hypertension_age_labels,
        "Hypertension",
        "hypertension_age_bias"
    )

else:

    print(
        "\n[SKIPPED] "
        "Hypertension dataset does not contain age."
    )


# ============================================================
# 12. FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("BIAS ANALYSIS SUMMARY")
print("=" * 70)


print(
"""
Diabetes:
- Class distribution analyzed.
- Prediction distribution analyzed.
- Feature distribution analyzed.
- Age-group bias evaluated using test-set tertiles.
- Raw age thresholds were NOT used because Diabetes Age
  is standardized.
- Gender bias: NOT AVAILABLE because the dataset has no
  gender attribute.

Hypertension:
- Class distribution analyzed.
- Prediction distribution analyzed.
- Feature distribution analyzed.
- Gender bias analyzed.
- Age-group bias analyzed using real age values.

Bias metrics:
- Accuracy
- Precision
- Recall
- F1
- Prediction Rate
- Actual Positive Rate
- False Positive Rate
- False Negative Rate

Bias gap:
Difference between the highest and lowest
group-level metric.

Important:
Bias indicators do not automatically prove that
a model is unfair or discriminatory.

They identify potential differences in:
- Data distribution
- Prediction rates
- Error rates
- Model performance between groups

These results should be interpreted together with
the Fairness Analysis, Robustness Analysis,
Explainability Analysis, and overall model performance.
"""
)


# ============================================================
# 13. OUTPUT CHECK
# ============================================================

print("=" * 70)
print("BIAS OUTPUT FILES")
print("=" * 70)


expected_outputs = [
    "diabetes_age_bias.csv",
    "diabetes_age_bias_gaps.csv",
    "hypertension_gender_bias.csv",
    "hypertension_gender_bias_gaps.csv",
    "hypertension_age_bias.csv",
    "hypertension_age_bias_gaps.csv"
]


for filename in expected_outputs:

    path = os.path.join(
        BIAS_OUTPUT,
        filename
    )

    if os.path.exists(path):

        print(
            f"[OK] {filename}"
        )

    else:

        print(
            f"[WARNING] {filename} not generated"
        )


print("\n")
print("=" * 70)
print("BIAS ANALYSIS COMPLETED")
print("=" * 70)

print(
    "\nProcess finished successfully."
)