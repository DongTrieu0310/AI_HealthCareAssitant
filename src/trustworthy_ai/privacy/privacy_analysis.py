import os
import joblib
import pandas as pd


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "../../.."
    )
)


DATA_RAW = os.path.join(
    PROJECT_ROOT,
    "data",
    "raw"
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
# DATASET PATHS
# ============================================================

DIABETES_RAW = os.path.join(
    DATA_RAW,
    "diabetes.csv"
)

HYPERTENSION_RAW = os.path.join(
    DATA_RAW,
    "hypertension.csv"
)

CARDIO_RAW = os.path.join(
    DATA_RAW,
    "cardio.csv"
)


# ============================================================
# MODEL PATHS
# ============================================================

DIABETES_MODEL = os.path.join(
    DATA_MODELS,
    "diabetes_random_forest.pkl"
)

HYPERTENSION_MODEL = os.path.join(
    DATA_MODELS,
    "hypertension_random_forest.pkl"
)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("TRUSTWORTHY AI - PRIVACY ANALYSIS")
print("=" * 70)

print("\nProject root:")
print(PROJECT_ROOT)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def check_file(path, name):
    """
    Check whether a required file exists.
    """

    if os.path.exists(path):

        print(f"[OK] {name}: {path}")
        return True

    print(f"[MISSING] {name}: {path}")
    return False


def analyze_dataset_privacy(
    dataset_name,
    file_path,
    identifier_columns=None
):

    print("\n")
    print("=" * 70)
    print(f"{dataset_name.upper()} - PRIVACY ANALYSIS")
    print("=" * 70)

    if not os.path.exists(file_path):

        print("Dataset not found.")
        return

    df = pd.read_csv(file_path)

    print("\nDataset information:")
    print("Rows    :", df.shape[0])
    print("Columns :", df.shape[1])

    print("\nColumns:")
    print(list(df.columns))

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    missing_total = df.isnull().sum().sum()

    print("\nMissing values:")
    print(missing_total)

    # --------------------------------------------------------
    # Duplicate records
    # --------------------------------------------------------

    duplicate_count = df.duplicated().sum()

    print("\nDuplicate rows:")
    print(duplicate_count)

    # --------------------------------------------------------
    # Identifier detection
    # --------------------------------------------------------

    if identifier_columns is None:
        identifier_columns = []

    detected_identifiers = []

    for column in df.columns:

        column_lower = column.lower()

        if (
            column_lower == "id"
            or column_lower.endswith("_id")
            or column_lower == "patient_id"
            or column_lower == "user_id"
        ):

            detected_identifiers.append(column)

    detected_identifiers = list(
        set(
            detected_identifiers +
            identifier_columns
        )
    )

    print("\nPotential direct identifiers:")

    if detected_identifiers:

        for column in detected_identifiers:
            print(f"- {column}")

    else:

        print("None detected.")

    # --------------------------------------------------------
    # Sensitive health-related columns
    # --------------------------------------------------------

    health_keywords = [
        "glucose",
        "diabetes",
        "bloodpressure",
        "sysbp",
        "diabp",
        "chol",
        "bmi",
        "insulin",
        "pregnancies",
        "smoker",
        "cigs",
        "heartrate",
        "cardio",
        "risk"
    ]

    sensitive_columns = []

    for column in df.columns:

        column_lower = column.lower()

        for keyword in health_keywords:

            if keyword in column_lower:

                sensitive_columns.append(column)
                break

    print("\nPotential health-sensitive attributes:")

    if sensitive_columns:

        for column in sensitive_columns:
            print(f"- {column}")

    else:

        print("None detected.")

    # --------------------------------------------------------
    # Unique-value analysis
    # --------------------------------------------------------

    print("\nHigh-cardinality columns:")

    high_cardinality_found = False

    for column in df.columns:

        unique_count = df[column].nunique()

        if unique_count > 0.9 * len(df):

            print(
                f"- {column}: "
                f"{unique_count} unique values "
                f"({unique_count / len(df):.2%})"
            )

            high_cardinality_found = True

    if not high_cardinality_found:

        print("None detected.")

    # --------------------------------------------------------
    # Privacy assessment
    # --------------------------------------------------------

    print("\nPrivacy assessment:")

    if detected_identifiers:

        print(
            "WARNING: Direct identifier(s) detected."
        )

        print(
            "Recommendation: remove identifiers "
            "before model training."
        )

    else:

        print(
            "No obvious direct identifier detected."
        )

    if sensitive_columns:

        print(
            "Health-sensitive attributes are present."
        )

        print(
            "These attributes should be handled "
            "with appropriate privacy protection."
        )

    print(
        "Privacy assessment completed for dataset."
    )


def inspect_model_file(
    model_name,
    model_path
):

    print("\n")
    print("-" * 70)
    print(f"{model_name.upper()} - MODEL PRIVACY CHECK")
    print("-" * 70)

    if not os.path.exists(model_path):

        print("Model file not found.")
        return

    file_size = os.path.getsize(model_path)

    print("\nModel file:")
    print(model_path)

    print(
        f"File size: "
        f"{file_size / 1024:.2f} KB"
    )

    try:

        model = joblib.load(
            model_path
        )

        print(
            "Model loaded successfully."
        )

        print(
            "Model type:",
            type(model).__name__
        )

        # ----------------------------------------------------
        # Check whether model exposes training data
        # ----------------------------------------------------

        suspicious_attributes = [
            "X_train_",
            "y_train_",
            "training_data",
            "train_data"
        ]

        found_training_data = []

        for attribute in suspicious_attributes:

            if hasattr(model, attribute):

                found_training_data.append(
                    attribute
                )

        print(
            "\nTraining-data attributes "
            "stored directly in model:"
        )

        if found_training_data:

            for attribute in found_training_data:

                print(
                    f"- {attribute}"
                )

            print(
                "WARNING: Model may contain "
                "training-related data."
            )

        else:

            print(
                "None detected."
            )

        print(
            "\nModel privacy assessment:"
        )

        print(
            "- Model file does not appear "
            "to intentionally store raw patient records."
        )

        print(
            "- Access to model files should "
            "still be restricted."
        )

    except Exception as error:

        print(
            "Could not inspect model:"
        )

        print(
            repr(error)
        )


# ============================================================
# 1. CHECK REQUIRED FILES
# ============================================================

print("\n")
print("=" * 70)
print("CHECKING REQUIRED FILES")
print("=" * 70)

check_file(
    DIABETES_RAW,
    "Diabetes raw dataset"
)

check_file(
    HYPERTENSION_RAW,
    "Hypertension raw dataset"
)

check_file(
    CARDIO_RAW,
    "Cardio raw dataset"
)

check_file(
    DIABETES_MODEL,
    "Diabetes model"
)

check_file(
    HYPERTENSION_MODEL,
    "Hypertension model"
)


# ============================================================
# 2. DIABETES
# ============================================================

analyze_dataset_privacy(
    dataset_name="Diabetes",
    file_path=DIABETES_RAW
)


# ============================================================
# 3. HYPERTENSION
# ============================================================

analyze_dataset_privacy(
    dataset_name="Hypertension",
    file_path=HYPERTENSION_RAW
)


# ============================================================
# 4. CARDIO
# ============================================================

analyze_dataset_privacy(
    dataset_name="Cardio",
    file_path=CARDIO_RAW,
    identifier_columns=["id"]
)


# ============================================================
# 5. MODEL PRIVACY
# ============================================================

print("\n")
print("=" * 70)
print("MODEL PRIVACY ANALYSIS")
print("=" * 70)

inspect_model_file(
    model_name="Diabetes Random Forest",
    model_path=DIABETES_MODEL
)

inspect_model_file(
    model_name="Hypertension Random Forest",
    model_path=HYPERTENSION_MODEL
)


# ============================================================
# 6. FINAL PRIVACY SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("PRIVACY ANALYSIS SUMMARY")
print("=" * 70)

print("""
Privacy analysis evaluates:

- Direct identifiers
- Health-sensitive attributes
- Duplicate records
- Missing values
- High-cardinality attributes
- Potential training-data exposure
- Model file privacy

Important findings:

1. Healthcare datasets contain sensitive health information.
2. Direct identifiers should be removed before model training.
3. The CARDIO dataset contains an 'id' column.
4. Diabetes and Hypertension datasets do not appear to
   contain an obvious direct identifier.
5. Model files should be protected from unauthorized access.
6. Privacy protection is not only a dataset issue;
   access control and secure deployment are also important.

IMPORTANT:

The absence of a direct identifier does NOT guarantee
complete privacy.

Re-identification can still be possible when multiple
attributes are combined.

Therefore, privacy analysis should be considered a
risk assessment rather than proof of complete privacy.
""")


# ============================================================
# 7. COMPLETED
# ============================================================

print("\n")
print("=" * 70)
print("PRIVACY ANALYSIS COMPLETED")
print("=" * 70)