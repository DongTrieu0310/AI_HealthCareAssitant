import os
import numpy as np
import pandas as pd


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../..")
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
# HEADER
# ============================================================

print("=" * 70)
print("MATHEMATICS FOR AI - ADVANCED LINEAR ALGEBRA")
print("=" * 70)

print()
print("Project root:")
print(PROJECT_ROOT)


# ============================================================
# CHECK FILES
# ============================================================

print()
print("=" * 70)
print("CHECKING REQUIRED FILES")
print("=" * 70)

for name, path in [
    ("Diabetes X_test", DIABETES_X_TEST),
    ("Hypertension X_test", HYPERTENSION_X_TEST),
]:

    if os.path.exists(path):
        print(f"[OK] {name}: {path}")
    else:
        print(f"[ERROR] {name}: {path}")


# ============================================================
# LOAD DATA
# ============================================================

diabetes = pd.read_csv(DIABETES_X_TEST)
hypertension = pd.read_csv(HYPERTENSION_X_TEST)


# ============================================================
# HELPER FUNCTION
# ============================================================

def analyze_dataset(dataset_name, df):

    print()
    print("=" * 70)
    print(f"{dataset_name.upper()} - ADVANCED LINEAR ALGEBRA")
    print("=" * 70)

    X = df.to_numpy(dtype=float)

    print()
    print("-" * 70)
    print("1. FEATURE MATRIX")
    print("-" * 70)

    print(f"Matrix shape: {X.shape}")

    n_samples, n_features = X.shape

    print(f"Rows        : {n_samples}")
    print(f"Columns     : {n_features}")

    print(f"X ∈ R^({n_samples} × {n_features})")


    # ========================================================
    # 2. COVARIANCE MATRIX
    # ========================================================

    print()
    print("-" * 70)
    print("2. COVARIANCE MATRIX")
    print("-" * 70)

    covariance_matrix = np.cov(
        X,
        rowvar=False
    )

    print("Covariance matrix shape:")
    print(covariance_matrix.shape)

    print()
    print("First 5 × 5 elements:")

    print(
        np.round(
            covariance_matrix[:5, :5],
            4
        )
    )


    # ========================================================
    # 3. SYMMETRY CHECK
    # ========================================================

    print()
    print("-" * 70)
    print("3. COVARIANCE MATRIX SYMMETRY")
    print("-" * 70)

    symmetric = np.allclose(
        covariance_matrix,
        covariance_matrix.T
    )

    print(f"Symmetric: {symmetric}")

    print()
    print("Mathematical property:")
    print("Σ = Σ^T")


    # ========================================================
    # 4. EIGENVALUES AND EIGENVECTORS
    # ========================================================

    print()
    print("-" * 70)
    print("4. EIGENVALUE AND EIGENVECTOR")
    print("-" * 70)

    eigenvalues, eigenvectors = np.linalg.eigh(
        covariance_matrix
    )

    # Sort descending
    indices = np.argsort(
        eigenvalues
    )[::-1]

    eigenvalues = eigenvalues[indices]
    eigenvectors = eigenvectors[:, indices]

    print("Eigenvalues:")

    for i, value in enumerate(eigenvalues):
        print(
            f"λ{i + 1}: {value:.6f}"
        )


    # ========================================================
    # 5. EXPLAINED VARIANCE
    # ========================================================

    print()
    print("-" * 70)
    print("5. EXPLAINED VARIANCE")
    print("-" * 70)

    total_variance = np.sum(
        eigenvalues
    )

    explained_variance_ratio = (
        eigenvalues / total_variance
    )

    cumulative_variance = np.cumsum(
        explained_variance_ratio
    )

    variance_table = pd.DataFrame({
        "Component": np.arange(
            1,
            len(eigenvalues) + 1
        ),
        "Eigenvalue": eigenvalues,
        "Explained Variance": explained_variance_ratio,
        "Cumulative Variance": cumulative_variance
    })

    print(
        variance_table.to_string(
            index=False,
            formatters={
                "Eigenvalue": "{:.6f}".format,
                "Explained Variance": "{:.4f}".format,
                "Cumulative Variance": "{:.4f}".format
            }
        )
    )


    # ========================================================
    # 6. POSITIVE DEFINITENESS
    # ========================================================

    print()
    print("-" * 70)
    print("6. MATRIX DEFINITENESS")
    print("-" * 70)

    tolerance = 1e-10

    min_eigenvalue = np.min(
        eigenvalues
    )

    max_eigenvalue = np.max(
        eigenvalues
    )

    print(
        f"Minimum eigenvalue: {min_eigenvalue:.10f}"
    )

    print(
        f"Maximum eigenvalue: {max_eigenvalue:.10f}"
    )

    if min_eigenvalue > tolerance:

        definiteness = "POSITIVE DEFINITE"

    elif min_eigenvalue >= -tolerance:

        definiteness = "POSITIVE SEMIDEFINITE"

    elif max_eigenvalue < -tolerance:

        definiteness = "NEGATIVE DEFINITE"

    elif max_eigenvalue <= tolerance:

        definiteness = "NEGATIVE SEMIDEFINITE"

    else:

        definiteness = "INDEFINITE"

    print()
    print(
        f"Matrix classification: {definiteness}"
    )

    print()
    print("Decision rule:")
    print("All eigenvalues > 0  → Positive Definite")
    print("All eigenvalues >= 0 → Positive Semidefinite")
    print("All eigenvalues < 0  → Negative Definite")
    print("All eigenvalues <= 0 → Negative Semidefinite")
    print("Mixed signs          → Indefinite")


    # ========================================================
    # 7. CONDITION NUMBER
    # ========================================================

    print()
    print("-" * 70)
    print("7. CONDITION NUMBER")
    print("-" * 70)

    condition_number = np.linalg.cond(
        covariance_matrix
    )

    print(
        f"Condition number: {condition_number:.6f}"
    )

    if condition_number < 10:

        condition_status = "Well-conditioned"

    elif condition_number < 100:

        condition_status = "Moderately conditioned"

    elif condition_number < 1000:

        condition_status = "Poorly conditioned"

    else:

        condition_status = "Ill-conditioned"

    print(
        f"Matrix condition: {condition_status}"
    )


    # ========================================================
    # 8. RANK
    # ========================================================

    print()
    print("-" * 70)
    print("8. COVARIANCE MATRIX RANK")
    print("-" * 70)

    covariance_rank = np.linalg.matrix_rank(
        covariance_matrix
    )

    print(
        f"Rank: {covariance_rank}"
    )

    print(
        f"Maximum possible rank: {n_features}"
    )


    # ========================================================
    # 9. PCA INTERPRETATION
    # ========================================================

    print()
    print("-" * 70)
    print("9. PCA INTERPRETATION")
    print("-" * 70)

    for k in [1, 2, 3]:

        if k <= len(eigenvalues):

            explained = cumulative_variance[k - 1]

            print(
                f"First {k} principal component(s): "
                f"{explained * 100:.2f}% variance"
            )


    # ========================================================
    # 10. SUMMARY
    # ========================================================

    print()
    print("-" * 70)
    print("10. ADVANCED LINEAR ALGEBRA SUMMARY")
    print("-" * 70)

    print(
        f"Dataset                 : {dataset_name}"
    )

    print(
        f"Feature matrix          : {X.shape[0]} × {X.shape[1]}"
    )

    print(
        f"Covariance matrix       : {covariance_matrix.shape}"
    )

    print(
        f"Covariance rank         : {covariance_rank}"
    )

    print(
        f"Matrix definiteness     : {definiteness}"
    )

    print(
        f"Largest eigenvalue      : {max_eigenvalue:.6f}"
    )

    print(
        f"Condition number        : {condition_number:.6f}"
    )

    print(
        f"Variance PC1            : "
        f"{explained_variance_ratio[0] * 100:.2f}%"
    )

    if len(explained_variance_ratio) >= 2:

        print(
            f"Variance PC1-PC2        : "
            f"{cumulative_variance[1] * 100:.2f}%"
        )

    print()
    print(
        f"{dataset_name} advanced linear algebra analysis completed."
    )


# ============================================================
# RUN ANALYSIS
# ============================================================

analyze_dataset(
    "Diabetes",
    diabetes
)

analyze_dataset(
    "Hypertension",
    hypertension
)


print()
print("=" * 70)
print("ADVANCED LINEAR ALGEBRA ANALYSIS COMPLETED")
print("=" * 70)