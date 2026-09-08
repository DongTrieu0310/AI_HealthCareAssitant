import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier


print("=" * 70)
print("TRUSTWORTHY AI - EXPLAINABILITY ANALYSIS")
print("=" * 70)


# ============================================================
# 1. PROJECT ROOT
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "../../.."
    )
)

print("\nProject root:")
print(PROJECT_ROOT)


# ============================================================
# 2. PATH HELPER
# ============================================================

def project_path(*parts):

    return os.path.join(
        PROJECT_ROOT,
        *parts
    )


# ============================================================
# 3. FILE PATHS
# ============================================================

DIABETES_X_TEST = project_path(
    "data",
    "processed",
    "diabetes_X_test.csv"
)

DIABETES_MODEL = project_path(
    "data",
    "models",
    "diabetes_random_forest.pkl"
)

DIABETES_THRESHOLD = project_path(
    "data",
    "models",
    "diabetes_threshold.pkl"
)


HYPERTENSION_X_TEST = project_path(
    "data",
    "processed",
    "hypertension_X_test.csv"
)

HYPERTENSION_MODEL = project_path(
    "data",
    "models",
    "hypertension_random_forest.pkl"
)

HYPERTENSION_THRESHOLD = project_path(
    "data",
    "models",
    "hypertension_threshold.pkl"
)


# ============================================================
# 4. CHECK FILES
# ============================================================

def check_file(path, name):

    if os.path.exists(path):

        print(f"[OK] {name}")

        return True

    print(f"[ERROR] Missing: {name}")
    print(f"Path: {path}")

    return False


print("\n")
print("=" * 70)
print("CHECKING REQUIRED FILES")
print("=" * 70)


required_files = [

    (DIABETES_X_TEST, "Diabetes X_test"),
    (DIABETES_MODEL, "Diabetes model"),
    (DIABETES_THRESHOLD, "Diabetes threshold"),

    (HYPERTENSION_X_TEST, "Hypertension X_test"),
    (HYPERTENSION_MODEL, "Hypertension model"),
    (HYPERTENSION_THRESHOLD, "Hypertension threshold")

]


all_files_exist = True


for path, name in required_files:

    if not check_file(path, name):

        all_files_exist = False


if not all_files_exist:

    raise FileNotFoundError(
        "\nSome required files are missing."
    )


# ============================================================
# 5. LOAD MODEL
# ============================================================

def load_model(path):

    model = joblib.load(
        path
    )

    return model


# ============================================================
# 6. GLOBAL FEATURE IMPORTANCE
# ============================================================

def analyze_feature_importance(
    dataset_name,
    X,
    model
):

    print("\n")
    print("=" * 70)
    print(
        f"{dataset_name.upper()} - GLOBAL EXPLAINABILITY"
    )
    print("=" * 70)


    # --------------------------------------------------------
    # Feature importance
    # --------------------------------------------------------

    importance = model.feature_importances_


    feature_importance = pd.DataFrame({

        "Feature": X.columns,

        "Importance": importance

    })


    feature_importance = (
        feature_importance
        .sort_values(
            by="Importance",
            ascending=False
        )
        .reset_index(drop=True)
    )


    print("\n")
    print("-" * 70)
    print("FEATURE IMPORTANCE")
    print("-" * 70)


    print(
        feature_importance.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )


    # --------------------------------------------------------
    # Top 5
    # --------------------------------------------------------

    top5 = feature_importance.head(5)


    print("\n")
    print("-" * 70)
    print("TOP 5 FEATURES")
    print("-" * 70)


    print(
        top5.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )


    # --------------------------------------------------------
    # Most important feature
    # --------------------------------------------------------

    top_feature = feature_importance.iloc[0]


    print("\n")
    print("-" * 70)
    print("MOST IMPORTANT FEATURE")
    print("-" * 70)


    print(
        f"Feature   : {top_feature['Feature']}"
    )

    print(
        f"Importance: {top_feature['Importance']:.4f}"
    )


    # --------------------------------------------------------
    # Visualization
    # --------------------------------------------------------

    plt.figure(
        figsize=(10, 6)
    )


    plt.barh(
        feature_importance["Feature"],
        feature_importance["Importance"]
    )


    plt.xlabel(
        "Importance"
    )

    plt.ylabel(
        "Feature"
    )

    plt.title(
        f"{dataset_name} - Random Forest Feature Importance"
    )


    plt.gca().invert_yaxis()

    plt.tight_layout()

    plt.show()


    return feature_importance


# ============================================================
# 7. LOCAL EXPLAINABILITY
# ============================================================

def local_explanation(
    dataset_name,
    X,
    model,
    threshold,
    patient_index=0
):

    print("\n")
    print("=" * 70)
    print(
        f"{dataset_name.upper()} - LOCAL EXPLAINABILITY"
    )
    print("=" * 70)


    # --------------------------------------------------------
    # Select patient
    # --------------------------------------------------------

    patient = X.iloc[
        patient_index
    ]


    patient_df = pd.DataFrame(
        [patient]
    )


    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    probability = model.predict_proba(
        patient_df
    )[0, 1]


    prediction = int(
        probability >= threshold
    )


    print("\n")
    print("-" * 70)
    print("PATIENT INFORMATION")
    print("-" * 70)


    print(
        patient.to_string()
    )


    print("\n")
    print("-" * 70)
    print("MODEL PREDICTION")
    print("-" * 70)


    print(
        f"Probability : {probability:.4f}"
    )

    print(
        f"Probability : {probability * 100:.2f}%"
    )

    print(
        f"Threshold   : {threshold:.2f}"
    )


    if prediction == 1:

        print(
            "Prediction  : POSITIVE RISK"
        )

    else:

        print(
            "Prediction  : NEGATIVE RISK"
        )


    # --------------------------------------------------------
    # Compare patient with test-set median
    # --------------------------------------------------------

    medians = X.median()


    explanation = pd.DataFrame({

        "Patient Value":
            patient,

        "Dataset Median":
            medians

    })


    explanation[
        "Difference"
    ] = (
        explanation["Patient Value"]
        - explanation["Dataset Median"]
    )


    # --------------------------------------------------------
    # Attach feature importance
    # --------------------------------------------------------

    feature_importance = pd.Series(

        model.feature_importances_,

        index=X.columns

    )


    explanation[
        "Importance"
    ] = feature_importance


    explanation = explanation.sort_values(
        by="Importance",
        ascending=False
    )


    print("\n")
    print("-" * 70)
    print("LOCAL FEATURE ANALYSIS")
    print("-" * 70)


    print(
        explanation.to_string(
            float_format=lambda x: f"{x:.4f}"
        )
    )


    # --------------------------------------------------------
    # Top influential features
    # --------------------------------------------------------

    print("\n")
    print("-" * 70)
    print("TOP FEATURES FOR THIS PATIENT")
    print("-" * 70)


    for feature, row in explanation.head(5).iterrows():

        patient_value = row[
            "Patient Value"
        ]

        median_value = row[
            "Dataset Median"
        ]

        importance = row[
            "Importance"
        ]


        direction = (
            "above"
            if patient_value > median_value
            else "below"
            if patient_value < median_value
            else "equal to"
        )


        print(
            f"{feature}: "
            f"patient={patient_value:.4f}, "
            f"median={median_value:.4f}, "
            f"{direction} median, "
            f"importance={importance:.4f}"
        )


    return explanation


# ============================================================
# 8. DIABETES EXPLAINABILITY
# ============================================================

print("\n")
print("=" * 70)
print("1. DIABETES EXPLAINABILITY")
print("=" * 70)


X_diabetes = pd.read_csv(
    DIABETES_X_TEST
)


diabetes_model = load_model(
    DIABETES_MODEL
)


diabetes_threshold = float(
    joblib.load(
        DIABETES_THRESHOLD
    )
)


print("\nDiabetes data:")
print(
    "X_test:",
    X_diabetes.shape
)

print(
    "Threshold:",
    diabetes_threshold
)


diabetes_importance = analyze_feature_importance(

    dataset_name="Diabetes",

    X=X_diabetes,

    model=diabetes_model

)


diabetes_local = local_explanation(

    dataset_name="Diabetes",

    X=X_diabetes,

    model=diabetes_model,

    threshold=diabetes_threshold,

    patient_index=0

)


# ============================================================
# 9. HYPERTENSION EXPLAINABILITY
# ============================================================

print("\n")
print("=" * 70)
print("2. HYPERTENSION EXPLAINABILITY")
print("=" * 70)


X_hypertension = pd.read_csv(
    HYPERTENSION_X_TEST
)


hypertension_model = load_model(
    HYPERTENSION_MODEL
)


hypertension_threshold = float(
    joblib.load(
        HYPERTENSION_THRESHOLD
    )
)


print("\nHypertension data:")

print(
    "X_test:",
    X_hypertension.shape
)

print(
    "Threshold:",
    hypertension_threshold
)


hypertension_importance = analyze_feature_importance(

    dataset_name="Hypertension",

    X=X_hypertension,

    model=hypertension_model

)


hypertension_local = local_explanation(

    dataset_name="Hypertension",

    X=X_hypertension,

    model=hypertension_model,

    threshold=hypertension_threshold,

    patient_index=0

)


# ============================================================
# 10. SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("EXPLAINABILITY SUMMARY")
print("=" * 70)


print("\nDiabetes:")

print(
    f"- Top feature: "
    f"{diabetes_importance.iloc[0]['Feature']}"
)

print(
    f"- Importance: "
    f"{diabetes_importance.iloc[0]['Importance']:.4f}"
)


print("\nHypertension:")

print(
    f"- Top feature: "
    f"{hypertension_importance.iloc[0]['Feature']}"
)

print(
    f"- Importance: "
    f"{hypertension_importance.iloc[0]['Importance']:.4f}"
)


print("\n")
print("=" * 70)
print("EXPLAINABILITY ANALYSIS COMPLETED")
print("=" * 70)