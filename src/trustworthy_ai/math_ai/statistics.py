import os
import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


print("=" * 70)
print("TRUSTWORTHY AI - MATH AI STATISTICS")
print("=" * 70)


# ============================================================
# 1. PATHS
# ============================================================

BASE_PROCESSED = "../../data/processed"
BASE_MODELS = "../../data/models"


# -------------------------
# CARDIO
# -------------------------

CARDIO_X_TEST = os.path.join(
    BASE_PROCESSED,
    "cardio_X_test.csv"
)

CARDIO_Y_TEST = os.path.join(
    BASE_PROCESSED,
    "cardio_y_test.csv"
)

CARDIO_MODEL = os.path.join(
    BASE_MODELS,
    "cardio_random_forest.pkl"
)

CARDIO_THRESHOLD = os.path.join(
    BASE_MODELS,
    "cardio_threshold.pkl"
)


# -------------------------
# DIABETES
# -------------------------

DIABETES_X_TEST = os.path.join(
    BASE_PROCESSED,
    "diabetes_X_test.csv"
)

DIABETES_Y_TEST = os.path.join(
    BASE_PROCESSED,
    "diabetes_y_test.csv"
)

DIABETES_MODEL = os.path.join(
    BASE_MODELS,
    "diabetes_random_forest.pkl"
)

DIABETES_THRESHOLD = os.path.join(
    BASE_MODELS,
    "diabetes_threshold.pkl"
)


# -------------------------
# HYPERTENSION
# -------------------------

HYPERTENSION_X_TEST = os.path.join(
    BASE_PROCESSED,
    "hypertension_X_test.csv"
)

HYPERTENSION_Y_TEST = os.path.join(
    BASE_PROCESSED,
    "hypertension_y_test.csv"
)

HYPERTENSION_MODEL = os.path.join(
    BASE_MODELS,
    "hypertension_random_forest.pkl"
)

HYPERTENSION_THRESHOLD = os.path.join(
    BASE_MODELS,
    "hypertension_threshold.pkl"
)


# ============================================================
# 2. CHECK REQUIRED FILES
# ============================================================

print("\n")
print("=" * 70)
print("CHECKING REQUIRED FILES")
print("=" * 70)


required_files = {

    "Cardio X_test": CARDIO_X_TEST,
    "Cardio y_test": CARDIO_Y_TEST,
    "Cardio model": CARDIO_MODEL,
    "Cardio threshold": CARDIO_THRESHOLD,

    "Diabetes X_test": DIABETES_X_TEST,
    "Diabetes y_test": DIABETES_Y_TEST,
    "Diabetes model": DIABETES_MODEL,
    "Diabetes threshold": DIABETES_THRESHOLD,

    "Hypertension X_test": HYPERTENSION_X_TEST,
    "Hypertension y_test": HYPERTENSION_Y_TEST,
    "Hypertension model": HYPERTENSION_MODEL,
    "Hypertension threshold": HYPERTENSION_THRESHOLD
}


all_files_available = True


for name, path in required_files.items():

    if os.path.exists(path):

        print(
            f"[OK] {name}: {os.path.abspath(path)}"
        )

    else:

        print(
            f"[MISSING] {name}: {os.path.abspath(path)}"
        )

        all_files_available = False


if not all_files_available:

    raise FileNotFoundError(
        "\nSome required files are missing."
    )


print("\nAll required files are available.")


# ============================================================
# 3. HELPER FUNCTION
# ============================================================

def analyze_dataset(
    dataset_name,
    x_path,
    y_path,
    model_path,
    threshold_path
):

    print("\n")
    print("=" * 70)
    print(f"{dataset_name.upper()} MODEL DATA LOADED")
    print("=" * 70)


    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    X_test = pd.read_csv(
        x_path
    )

    y_test = pd.read_csv(
        y_path
    ).squeeze()


    model = joblib.load(
        model_path
    )

    threshold = joblib.load(
        threshold_path
    )


    print(
        f"X_test   : {X_test.shape}"
    )

    print(
        f"y_test   : {y_test.shape}"
    )

    print(
        f"Threshold: {threshold}"
    )


    # --------------------------------------------------------
    # MISSING VALUES
    # --------------------------------------------------------

    missing_x = X_test.isnull().sum().sum()
    missing_y = y_test.isnull().sum()


    print(
        f"Missing values X_test: {missing_x}"
    )

    print(
        f"Missing values y_test : {missing_y}"
    )


    if missing_x > 0 or missing_y > 0:

        raise ValueError(
            f"{dataset_name} contains missing values."
        )


    # --------------------------------------------------------
    # DESCRIPTIVE STATISTICS
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print(
        f"{dataset_name.upper()} - DESCRIPTIVE STATISTICS"
    )
    print("=" * 70)


    print("\nDataset shape:")

    print(
        f"Rows    : {X_test.shape[0]}"
    )

    print(
        f"Features: {X_test.shape[1]}"
    )


    # --------------------------------------------------------
    # TARGET DISTRIBUTION
    # --------------------------------------------------------

    print("\nTarget distribution:")


    target_distribution = pd.DataFrame({

        "Count": y_test.value_counts().sort_index(),

        "Percentage":
            y_test.value_counts(
                normalize=True
            ).sort_index() * 100
    })


    print(
        target_distribution.round(2)
    )


    # --------------------------------------------------------
    # NUMERICAL DESCRIPTIVE STATISTICS
    # --------------------------------------------------------

    print("\n")
    print("-" * 70)
    print(
        "NUMERICAL DESCRIPTIVE STATISTICS"
    )
    print("-" * 70)


    descriptive = X_test.describe().T


    print(
        descriptive.round(4)
    )


    # --------------------------------------------------------
    # FEATURE VARIABILITY
    # --------------------------------------------------------

    print("\n")
    print("-" * 70)
    print(
        f"{dataset_name.upper()} - FEATURE VARIABILITY"
    )
    print("-" * 70)


    variability = pd.DataFrame({

        "Feature": X_test.columns,

        "Mean": X_test.mean().values,

        "Variance": X_test.var().values,

        "Std": X_test.std().values
    })


    variability = variability.sort_values(
        by="Variance",
        ascending=False
    )


    print(
        variability.round(4).to_string(
            index=False
        )
    )


    # --------------------------------------------------------
    # CORRELATION ANALYSIS
    # --------------------------------------------------------

    print("\n")
    print("-" * 70)
    print(
        f"{dataset_name.upper()} - CORRELATION ANALYSIS"
    )
    print("-" * 70)


    correlation = X_test.corr()


    print(
        correlation.round(3)
    )


    # --------------------------------------------------------
    # STRONGEST FEATURE CORRELATION
    # --------------------------------------------------------

    correlation_matrix = correlation.copy()


    # Remove diagonal
    for column in correlation_matrix.columns:

        correlation_matrix.loc[
            column,
            column
        ] = 0


    strongest_pair = (
        correlation_matrix.abs()
        .stack()
        .idxmax()
    )


    feature_1 = strongest_pair[0]
    feature_2 = strongest_pair[1]


    strongest_correlation = correlation.loc[
        feature_1,
        feature_2
    ]


    print("\n")
    print("-" * 70)
    print(
        "STRONGEST FEATURE CORRELATION"
    )
    print("-" * 70)


    print(
        f"Feature 1 : {feature_1}"
    )

    print(
        f"Feature 2 : {feature_2}"
    )

    print(
        f"Correlation: {strongest_correlation:.4f}"
    )


    # --------------------------------------------------------
    # MODEL PREDICTION
    # --------------------------------------------------------

    probability = model.predict_proba(
        X_test
    )[:, 1]


    prediction = (
        probability >= threshold
    ).astype(int)


    # --------------------------------------------------------
    # MODEL STATISTICS
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        prediction
    )

    precision = precision_score(
        y_test,
        prediction,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        prediction,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        prediction,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        probability
    )


    print("\n")
    print("=" * 70)
    print(
        f"{dataset_name.upper()} - MODEL STATISTICS"
    )
    print("=" * 70)


    print("\nModel performance:")

    print(
        f"Accuracy : {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall   : {recall:.4f}"
    )

    print(
        f"F1-score : {f1:.4f}"
    )

    print(
        f"ROC-AUC  : {roc_auc:.4f}"
    )


    # --------------------------------------------------------
    # PREDICTED PROBABILITY STATISTICS
    # --------------------------------------------------------

    print("\n")
    print("-" * 70)
    print(
        "PREDICTED PROBABILITY STATISTICS"
    )
    print("-" * 70)


    print(
        f"Mean   : {probability.mean():.4f}"
    )

    print(
        f"Median : {pd.Series(probability).median():.4f}"
    )

    print(
        f"Std    : {probability.std():.4f}"
    )

    print(
        f"Minimum: {probability.min():.4f}"
    )

    print(
        f"Maximum: {probability.max():.4f}"
    )

    print(
        f"Threshold: {threshold:.4f}"
    )


    # --------------------------------------------------------
    # RETURN RESULTS
    # --------------------------------------------------------

    return {

        "Dataset": dataset_name,

        "Accuracy": accuracy,

        "Precision": precision,

        "Recall": recall,

        "F1": f1,

        "ROC-AUC": roc_auc,

        "Threshold": threshold
    }


# ============================================================
# 4. CARDIO ANALYSIS
# ============================================================

cardio_results = analyze_dataset(

    "Cardio",

    CARDIO_X_TEST,

    CARDIO_Y_TEST,

    CARDIO_MODEL,

    CARDIO_THRESHOLD
)


# ============================================================
# 5. DIABETES ANALYSIS
# ============================================================

diabetes_results = analyze_dataset(

    "Diabetes",

    DIABETES_X_TEST,

    DIABETES_Y_TEST,

    DIABETES_MODEL,

    DIABETES_THRESHOLD
)


# ============================================================
# 6. HYPERTENSION ANALYSIS
# ============================================================

hypertension_results = analyze_dataset(

    "Hypertension",

    HYPERTENSION_X_TEST,

    HYPERTENSION_Y_TEST,

    HYPERTENSION_MODEL,

    HYPERTENSION_THRESHOLD
)


# ============================================================
# 7. FINAL STATISTICS SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("STATISTICS ANALYSIS SUMMARY")
print("=" * 70)


summary = pd.DataFrame([

    cardio_results,

    diabetes_results,

    hypertension_results
])


print(
    summary[
        [
            "Dataset",
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
            "ROC-AUC",
            "Threshold"
        ]
    ].round(4).to_string(
        index=False
    )
)


# ============================================================
# 8. BEST MODEL BY F1
# ============================================================

best_f1 = summary.loc[
    summary["F1"].idxmax()
]


print("\n")
print("-" * 70)
print("BEST MODEL BY F1-SCORE")
print("-" * 70)


print(
    f"Dataset   : {best_f1['Dataset']}"
)

print(
    f"F1-score  : {best_f1['F1']:.4f}"
)

print(
    f"ROC-AUC   : {best_f1['ROC-AUC']:.4f}"
)


# ============================================================
# 9. BEST MODEL BY ROC-AUC
# ============================================================

best_auc = summary.loc[
    summary["ROC-AUC"].idxmax()
]


print("\n")
print("-" * 70)
print("BEST MODEL BY ROC-AUC")
print("-" * 70)


print(
    f"Dataset   : {best_auc['Dataset']}"
)

print(
    f"ROC-AUC   : {best_auc['ROC-AUC']:.4f}"
)


# ============================================================
# 10. HEALTHCARE RECALL COMPARISON
# ============================================================

print("\n")
print("-" * 70)
print("HEALTHCARE-ORIENTED RECALL COMPARISON")
print("-" * 70)


for _, row in summary.iterrows():

    print(
        f"{row['Dataset']:<15} "
        f"Recall: {row['Recall']:.4f}"
    )


# ============================================================
# 11. FINAL CHECK
# ============================================================

print("\n")
print("=" * 70)
print("STATISTICS MODULE CHECK")
print("=" * 70)


if len(summary) == 3:

    print(
        "[PASSED] All three disease models analyzed."
    )

else:

    raise ValueError(
        "Statistics summary does not contain all three models."
    )


print(
    "[PASSED] Cardio statistics available."
)

print(
    "[PASSED] Diabetes statistics available."
)

print(
    "[PASSED] Hypertension statistics available."
)


print("\n")
print("=" * 70)
print(
    "MATH AI STATISTICS ANALYSIS COMPLETED SUCCESSFULLY"
)
print("=" * 70)