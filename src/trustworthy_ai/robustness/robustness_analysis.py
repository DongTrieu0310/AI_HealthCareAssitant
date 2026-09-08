import os
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


print("=" * 70)
print("TRUSTWORTHY AI - ROBUSTNESS ANALYSIS")
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
# 2. OUTPUT DIRECTORY
# ============================================================

ROBUSTNESS_OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "models",
    "robustness"
)

os.makedirs(
    ROBUSTNESS_OUTPUT_DIR,
    exist_ok=True
)

print("\nRobustness output directory:")
print(ROBUSTNESS_OUTPUT_DIR)


# ============================================================
# 3. PATH HELPER
# ============================================================

def project_path(*parts):

    return os.path.join(
        PROJECT_ROOT,
        *parts
    )


# ============================================================
# 4. FILE PATHS
# ============================================================

DIABETES_X_TEST = project_path(
    "data",
    "processed",
    "diabetes_X_test.csv"
)

DIABETES_Y_TEST = project_path(
    "data",
    "processed",
    "diabetes_y_test.csv"
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

HYPERTENSION_Y_TEST = project_path(
    "data",
    "processed",
    "hypertension_y_test.csv"
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
# 5. CHECK FILES
# ============================================================

def check_file(path, name):

    if os.path.exists(path):

        print(f"[OK] {name}: {path}")

        return True

    print(f"[MISSING] {name}: {path}")

    return False


print("\n")
print("=" * 70)
print("CHECKING REQUIRED FILES")
print("=" * 70)


required_files = [

    (DIABETES_X_TEST, "Diabetes X_test"),
    (DIABETES_Y_TEST, "Diabetes y_test"),
    (DIABETES_MODEL, "Diabetes model"),
    (DIABETES_THRESHOLD, "Diabetes threshold"),

    (HYPERTENSION_X_TEST, "Hypertension X_test"),
    (HYPERTENSION_Y_TEST, "Hypertension y_test"),
    (HYPERTENSION_MODEL, "Hypertension model"),
    (HYPERTENSION_THRESHOLD, "Hypertension threshold")

]


missing_files = []


for path, name in required_files:

    if not check_file(path, name):

        missing_files.append(name)


if missing_files:

    print("\nERROR: Required files are missing:")

    for name in missing_files:

        print("-", name)

    raise FileNotFoundError(
        "Some required files are missing."
    )


# ============================================================
# 6. LOAD MODEL
# ============================================================

def load_model(path):

    print("\nLoading model:")
    print(path)

    model = joblib.load(
        path
    )

    print("Model loaded successfully.")

    return model


# ============================================================
# 7. LOAD THRESHOLD
# ============================================================

def load_threshold(path):

    threshold = joblib.load(
        path
    )

    threshold = float(
        threshold
    )

    print(
        f"Threshold loaded: {threshold:.4f}"
    )

    return threshold


# ============================================================
# 8. METRICS
# ============================================================

def calculate_metrics(
    y_true,
    y_pred,
    y_prob
):

    return {

        "Accuracy": accuracy_score(
            y_true,
            y_pred
        ),

        "Precision": precision_score(
            y_true,
            y_pred,
            zero_division=0
        ),

        "Recall": recall_score(
            y_true,
            y_pred,
            zero_division=0
        ),

        "F1": f1_score(
            y_true,
            y_pred,
            zero_division=0
        ),

        "ROC-AUC": roc_auc_score(
            y_true,
            y_prob
        )

    }


# ============================================================
# 9. ADD CONTROLLED NOISE
# ============================================================

def add_noise(
    X,
    noise_level,
    dataset_name,
    random_seed=42
):
    """
    Add controlled Gaussian noise only to continuous features.

    Binary features are kept unchanged because adding Gaussian
    noise to binary demographic/clinical indicators would create
    unrealistic values.

    noise_level:
        0.01 = 1% of feature standard deviation
        0.05 = 5%
        0.10 = 10%
    """

    X_noisy = X.copy()

    rng = np.random.default_rng(
        random_seed
    )

    # --------------------------------------------------------
    # Diabetes
    # --------------------------------------------------------

    if dataset_name.lower() == "diabetes":

        continuous_columns = [
            column
            for column in X_noisy.columns
            if column not in []
        ]

        print(
            "\n[INFO] Diabetes:"
        )

        print(
            "Gaussian noise applied to "
            "continuous standardized features."
        )

    # --------------------------------------------------------
    # Hypertension
    # --------------------------------------------------------

    elif dataset_name.lower() == "hypertension":

        binary_columns = [
            "male",
            "currentSmoker",
            "BPMeds",
            "diabetes"
        ]

        continuous_columns = [
            column
            for column in X_noisy.columns
            if column not in binary_columns
        ]

        print(
            "\n[INFO] Hypertension:"
        )

        print(
            "Gaussian noise applied only to "
            "continuous clinical features."
        )

        print(
            "Binary features kept unchanged:"
        )

        print(
            ", ".join(binary_columns)
        )

    else:

        raise ValueError(
            f"Unknown dataset: {dataset_name}"
        )

    # --------------------------------------------------------
    # Apply Gaussian noise
    # --------------------------------------------------------

    for column in continuous_columns:

        if not pd.api.types.is_numeric_dtype(
            X_noisy[column]
        ):

            continue

        std = X_noisy[column].std()

        if pd.isna(std) or std == 0:

            continue

        noise = rng.normal(
            loc=0.0,
            scale=std * noise_level,
            size=len(X_noisy)
        )

        X_noisy[column] = (
            X_noisy[column] + noise
        )

    return X_noisy


# ============================================================
# 10. PERFORMANCE DROP
# ============================================================

def calculate_performance_drop(
    baseline,
    results_df
):

    drop_rows = []

    for _, row in results_df.iterrows():

        result = {
            "Noise": row["Noise"]
        }

        for metric in [
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
            "ROC-AUC"
        ]:

            result[
                f"{metric} Drop"
            ] = (
                baseline[metric]
                - row[metric]
            )

        drop_rows.append(
            result
        )

    return pd.DataFrame(
        drop_rows
    )


# ============================================================
# 11. ROBUSTNESS TEST
# ============================================================

def robustness_test(
    dataset_name,
    X_path,
    y_path,
    model_path,
    threshold_path
):

    print("\n")
    print("=" * 70)

    print(
        f"{dataset_name.upper()} - ROBUSTNESS TEST"
    )

    print("=" * 70)


    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    X = pd.read_csv(
        X_path
    )

    y = pd.read_csv(
        y_path
    ).squeeze()


    print("\nOriginal data:")
    print(
        "X:",
        X.shape
    )

    print(
        "y:",
        y.shape
    )


    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    model = load_model(
        model_path
    )

    threshold = load_threshold(
        threshold_path
    )


    # --------------------------------------------------------
    # BASELINE
    # --------------------------------------------------------

    probability = model.predict_proba(
        X
    )[:, 1]

    prediction = (
        probability >= threshold
    ).astype(int)


    baseline = calculate_metrics(
        y,
        prediction,
        probability
    )


    print("\n")
    print("-" * 70)
    print("BASELINE PERFORMANCE")
    print("-" * 70)


    for metric, value in baseline.items():

        print(
            f"{metric:<10}: {value:.4f}"
        )


    # --------------------------------------------------------
    # NOISE TEST
    # --------------------------------------------------------

    noise_levels = [
        0.01,
        0.05,
        0.10
    ]


    results = []


    for noise_level in noise_levels:

        print("\n")
        print(
            f"Testing noise level: "
            f"{noise_level * 100:.0f}%"
        )


        X_noisy = add_noise(
            X=X,
            noise_level=noise_level,
            dataset_name=dataset_name,
            random_seed=42
        )


        noisy_probability = (
            model.predict_proba(
                X_noisy
            )[:, 1]
        )


        noisy_prediction = (
            noisy_probability >= threshold
        ).astype(int)


        metrics = calculate_metrics(
            y,
            noisy_prediction,
            noisy_probability
        )


        results.append({

            "Noise": noise_level,

            "Accuracy":
                metrics["Accuracy"],

            "Precision":
                metrics["Precision"],

            "Recall":
                metrics["Recall"],

            "F1":
                metrics["F1"],

            "ROC-AUC":
                metrics["ROC-AUC"]

        })


    results_df = pd.DataFrame(
        results
    )


    # --------------------------------------------------------
    # DISPLAY RESULTS
    # --------------------------------------------------------

    print("\n")
    print("-" * 70)
    print("ROBUSTNESS RESULTS")
    print("-" * 70)


    print(
        results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )


    # --------------------------------------------------------
    # PERFORMANCE DROP
    # --------------------------------------------------------

    drop_df = calculate_performance_drop(
        baseline,
        results_df
    )


    print("\n")
    print("-" * 70)
    print("PERFORMANCE DROP FROM BASELINE")
    print("-" * 70)


    print(
        drop_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )


    # --------------------------------------------------------
    # WORST-CASE DROP
    # --------------------------------------------------------

    print("\n")
    print("-" * 70)
    print("WORST-CASE PERFORMANCE DROP")
    print("-" * 70)


    worst_case = {}


    for metric in [
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "ROC-AUC"
    ]:

        worst_drop = (
            drop_df[
                f"{metric} Drop"
            ].max()
        )

        worst_case[
            f"{metric} Drop"
        ] = worst_drop


        print(
            f"{metric:<10}: "
            f"{worst_drop:.4f}"
        )


    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    dataset_prefix = (
        dataset_name.lower()
    )


    results_path = os.path.join(
        ROBUSTNESS_OUTPUT_DIR,
        f"{dataset_prefix}_robustness.csv"
    )


    drop_path = os.path.join(
        ROBUSTNESS_OUTPUT_DIR,
        f"{dataset_prefix}_robustness_drop.csv"
    )


    results_df.to_csv(
        results_path,
        index=False
    )


    drop_df.to_csv(
        drop_path,
        index=False
    )


    print("\n")
    print(
        f"Results saved: {results_path}"
    )

    print(
        f"Drops saved  : {drop_path}"
    )


    return {
        "baseline": baseline,
        "results": results_df,
        "drops": drop_df,
        "worst_case": worst_case
    }


# ============================================================
# 12. DIABETES
# ============================================================

diabetes_results = robustness_test(

    dataset_name="Diabetes",

    X_path=DIABETES_X_TEST,

    y_path=DIABETES_Y_TEST,

    model_path=DIABETES_MODEL,

    threshold_path=DIABETES_THRESHOLD

)


# ============================================================
# 13. HYPERTENSION
# ============================================================

hypertension_results = robustness_test(

    dataset_name="Hypertension",

    X_path=HYPERTENSION_X_TEST,

    y_path=HYPERTENSION_Y_TEST,

    model_path=HYPERTENSION_MODEL,

    threshold_path=HYPERTENSION_THRESHOLD

)


# ============================================================
# 14. FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("ROBUSTNESS ANALYSIS SUMMARY")
print("=" * 70)


print("""
Robustness evaluation was performed by
introducing controlled Gaussian noise to
continuous input features.

Noise levels:
- 1%
- 5%
- 10%

Binary features in the Hypertension dataset
were kept unchanged to avoid unrealistic
feature values.

Performance was compared against the original
test-set baseline using:

- Accuracy
- Precision
- Recall
- F1
- ROC-AUC

Performance drop represents the decrease
from the original baseline performance.

A larger drop indicates greater sensitivity
to input perturbations.

These results should be interpreted together
with Fairness, Bias, Explainability,
Privacy, Accountability, and Social Impact.
""")


print("\n")
print("=" * 70)
print("ROBUSTNESS OUTPUT FILES")
print("=" * 70)


output_files = [

    "diabetes_robustness.csv",
    "diabetes_robustness_drop.csv",
    "hypertension_robustness.csv",
    "hypertension_robustness_drop.csv"

]


for filename in output_files:

    path = os.path.join(
        ROBUSTNESS_OUTPUT_DIR,
        filename
    )

    if os.path.exists(path):

        print(
            f"[OK] {filename}"
        )

    else:

        print(
            f"[MISSING] {filename}"
        )


print("\n")
print("=" * 70)
print("ROBUSTNESS ANALYSIS COMPLETED")
print("=" * 70)