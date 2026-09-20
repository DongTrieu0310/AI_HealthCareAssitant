import os
import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..", "..")
)

DATA_PROCESSED = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed"
)

DIABETES_X_TEST = os.path.join(
    DATA_PROCESSED,
    "diabetes_X_test.csv"
)

HYPERTENSION_X_TEST = os.path.join(
    DATA_PROCESSED,
    "hypertension_X_test.csv"
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_dataset(path):
    """
    Load processed feature dataset.
    """

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found:\n{path}"
        )

    data = pd.read_csv(path)

    return data


def analyze_matrix(X, dataset_name):
    """
    Perform basic linear algebra analysis
    on the feature matrix X.
    """

    print("\n" + "=" * 70)
    print(f"{dataset_name.upper()} - LINEAR ALGEBRA ANALYSIS")
    print("=" * 70)

    # --------------------------------------------------------
    # Matrix information
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("1. FEATURE MATRIX")
    print("-" * 70)

    print(f"Matrix shape : {X.shape}")
    print(f"Rows         : {X.shape[0]}")
    print(f"Columns      : {X.shape[1]}")

    print("\nFeature matrix X:")

    print(
        f"X ∈ R^({X.shape[0]} × {X.shape[1]})"
    )

    # --------------------------------------------------------
    # First patient vector
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("2. PATIENT FEATURE VECTOR")
    print("-" * 70)

    patient_vector = X.iloc[0].to_numpy(dtype=float)

    print(
        f"First patient vector shape: "
        f"{patient_vector.shape}"
    )

    print("\nPatient vector:")

    print(patient_vector)

    print(
        f"\nVector representation:"
        f"\nx ∈ R^{len(patient_vector)}"
    )

    # --------------------------------------------------------
    # Matrix transpose
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("3. MATRIX TRANSPOSE")
    print("-" * 70)

    X_matrix = X.to_numpy(dtype=float)

    X_transpose = X_matrix.T

    print(
        f"Original matrix X       : {X_matrix.shape}"
    )

    print(
        f"Transpose matrix X^T    : {X_transpose.shape}"
    )

    # --------------------------------------------------------
    # Vector norm
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("4. VECTOR NORM")
    print("-" * 70)

    vector_norm = np.linalg.norm(patient_vector)

    print(
        "Euclidean norm:"
    )

    print(
        f"||x|| = {vector_norm:.6f}"
    )

    print(
        "\nMathematical formula:"
    )

    print(
        "||x|| = sqrt(x1² + x2² + ... + xn²)"
    )

    # --------------------------------------------------------
    # Dot product
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("5. DOT PRODUCT")
    print("-" * 70)

    second_patient_vector = X.iloc[1].to_numpy(
        dtype=float
    )

    dot_product = np.dot(
        patient_vector,
        second_patient_vector
    )

    print(
        "Dot product between patient 1 and patient 2:"
    )

    print(
        f"x₁ · x₂ = {dot_product:.6f}"
    )

    print(
        "\nMathematical formula:"
    )

    print(
        "x₁ · x₂ = Σ xi * yi"
    )

    # --------------------------------------------------------
    # X transpose multiplied by X
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("6. MATRIX MULTIPLICATION: X^T X")
    print("-" * 70)

    XtX = X_transpose @ X_matrix

    print(
        f"X^T shape : {X_transpose.shape}"
    )

    print(
        f"X shape   : {X_matrix.shape}"
    )

    print(
        f"X^T X shape: {XtX.shape}"
    )

    print(
        "\nFirst 5 × 5 elements of X^T X:"
    )

    preview_size = min(5, XtX.shape[0])

    print(
        XtX[:preview_size, :preview_size]
    )

    # --------------------------------------------------------
    # Matrix rank
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("7. MATRIX RANK")
    print("-" * 70)

    matrix_rank = np.linalg.matrix_rank(
        X_matrix
    )

    print(
        f"Matrix rank: {matrix_rank}"
    )

    print(
        f"Maximum possible rank: "
        f"{min(X_matrix.shape)}"
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("8. LINEAR ALGEBRA SUMMARY")
    print("-" * 70)

    print(
        f"Dataset              : {dataset_name}"
    )

    print(
        f"Feature matrix       : "
        f"{X_matrix.shape[0]} × {X_matrix.shape[1]}"
    )

    print(
        f"Patient vector size  : "
        f"{len(patient_vector)}"
    )

    print(
        f"Vector norm          : "
        f"{vector_norm:.6f}"
    )

    print(
        f"Matrix rank          : "
        f"{matrix_rank}"
    )

    print(
        f"X^T X shape          : "
        f"{XtX.shape}"
    )

    print(
        "\nLinear algebra analysis completed."
    )


# ============================================================
# MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("MATHEMATICS FOR AI - LINEAR ALGEBRA")
    print("=" * 70)

    print("\nProject root:")
    print(PROJECT_ROOT)

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("CHECKING REQUIRED FILES")
    print("=" * 70)

    if os.path.exists(DIABETES_X_TEST):
        print(
            f"[OK] Diabetes X_test: "
            f"{DIABETES_X_TEST}"
        )
    else:
        print(
            f"[ERROR] Diabetes X_test not found: "
            f"{DIABETES_X_TEST}"
        )

    if os.path.exists(HYPERTENSION_X_TEST):
        print(
            f"[OK] Hypertension X_test: "
            f"{HYPERTENSION_X_TEST}"
        )
    else:
        print(
            f"[ERROR] Hypertension X_test not found: "
            f"{HYPERTENSION_X_TEST}"
        )

    # --------------------------------------------------------
    # Diabetes
    # --------------------------------------------------------

    if os.path.exists(DIABETES_X_TEST):

        diabetes = load_dataset(
            DIABETES_X_TEST
        )

        print("\nDiabetes data:")
        print(
            f"X_test: {diabetes.shape}"
        )

        analyze_matrix(
            diabetes,
            "Diabetes"
        )

    # --------------------------------------------------------
    # Hypertension
    # --------------------------------------------------------

    if os.path.exists(HYPERTENSION_X_TEST):

        hypertension = load_dataset(
            HYPERTENSION_X_TEST
        )

        print("\nHypertension data:")
        print(
            f"X_test: {hypertension.shape}"
        )

        analyze_matrix(
            hypertension,
            "Hypertension"
        )

    print("\n" + "=" * 70)
    print("LINEAR ALGEBRA ANALYSIS COMPLETED")
    print("=" * 70)