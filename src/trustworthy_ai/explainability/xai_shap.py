import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap


# ============================================================
# TRUSTWORTHY AI - EXPLAINABLE AI
# SHAP ANALYSIS
# ============================================================

print("=" * 70)
print("TRUSTWORTHY AI - EXPLAINABLE AI (SHAP)")
print("=" * 70)


# ============================================================
# 1. PROJECT PATH
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "../../../"
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

XAI_OUTPUT = os.path.join(
    DATA_MODELS,
    "xai"
)

os.makedirs(
    XAI_OUTPUT,
    exist_ok=True
)


print("\nProject root:")
print(PROJECT_ROOT)

print("\nXAI output directory:")
print(XAI_OUTPUT)


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def load_model_object(path):
    """
    Load model / threshold object saved with joblib.
    """
    return joblib.load(path)


def get_positive_class_shap_values(model, X):
    """
    Calculate SHAP values for the positive class.

    Compatible with different SHAP versions.
    """

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X)

    # Older SHAP versions:
    # shap_values = [class_0, class_1]
    if isinstance(shap_values, list):
        return shap_values[1]

    # Newer SHAP versions may return:
    # samples x features x classes
    if isinstance(shap_values, np.ndarray):

        if shap_values.ndim == 3:
            return shap_values[:, :, 1]

        return shap_values

    raise ValueError(
        "Unsupported SHAP value format."
    )


def analyze_model(
    name,
    X_test_path,
    model_path,
    threshold_path,
    sample_size=300
):

    print("\n")
    print("=" * 70)
    print(f"{name.upper()} - SHAP ANALYSIS")
    print("=" * 70)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    X_test = pd.read_csv(X_test_path)

    model = load_model_object(model_path)

    threshold = load_model_object(threshold_path)

    print("\nData:")
    print("X_test:", X_test.shape)

    print("\nModel:")
    print(model)

    print("\nThreshold:")
    print(f"{threshold:.2f}")

    # --------------------------------------------------------
    # Sample data
    # --------------------------------------------------------

    sample_size = min(
        sample_size,
        len(X_test)
    )

    X_sample = X_test.sample(
        n=sample_size,
        random_state=42
    )

    print("\nSHAP sample:")
    print(X_sample.shape)

    # --------------------------------------------------------
    # Calculate SHAP values
    # --------------------------------------------------------

    print("\nCalculating SHAP values...")

    shap_values = get_positive_class_shap_values(
        model,
        X_sample
    )

    print("SHAP calculation completed.")

    # --------------------------------------------------------
    # Mean absolute SHAP importance
    # --------------------------------------------------------

    mean_abs_shap = np.abs(
        shap_values
    ).mean(axis=0)

    importance = pd.DataFrame({
        "Feature": X_sample.columns,
        "Mean_ABS_SHAP": mean_abs_shap
    })

    importance = importance.sort_values(
        by="Mean_ABS_SHAP",
        ascending=False
    ).reset_index(drop=True)

    importance["Rank"] = (
        importance.index + 1
    )

    importance = importance[
        [
            "Rank",
            "Feature",
            "Mean_ABS_SHAP"
        ]
    ]

    print("\n")
    print("-" * 70)
    print("FEATURE IMPORTANCE - SHAP")
    print("-" * 70)

    print(
        importance.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Save importance
    # --------------------------------------------------------

    importance_path = os.path.join(
        XAI_OUTPUT,
        f"{name.lower()}_shap_importance.csv"
    )

    importance.to_csv(
        importance_path,
        index=False
    )

    print("\nImportance saved:")
    print(importance_path)

    # ========================================================
    # SHAP BAR PLOT
    # ========================================================

    plt.figure()

    shap.summary_plot(
        shap_values,
        X_sample,
        plot_type="bar",
        show=False
    )

    plt.title(
        f"{name} - SHAP Feature Importance"
    )

    plt.tight_layout()

    bar_path = os.path.join(
        XAI_OUTPUT,
        f"{name.lower()}_shap_bar.png"
    )

    plt.savefig(
        bar_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("\nSHAP bar plot saved:")
    print(bar_path)

    # ========================================================
    # SHAP BEESWARM PLOT
    # ========================================================

    plt.figure()

    shap.summary_plot(
        shap_values,
        X_sample,
        show=False
    )

    plt.title(
        f"{name} - SHAP Summary Plot"
    )

    plt.tight_layout()

    beeswarm_path = os.path.join(
        XAI_OUTPUT,
        f"{name.lower()}_shap_summary.png"
    )

    plt.savefig(
        beeswarm_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("\nSHAP summary plot saved:")
    print(beeswarm_path)

    # --------------------------------------------------------
    # Top 5 features
    # --------------------------------------------------------

    print("\n")
    print("-" * 70)
    print("TOP 5 SHAP FEATURES")
    print("-" * 70)

    for _, row in importance.head(5).iterrows():

        print(
            f"{int(row['Rank'])}. "
            f"{row['Feature']} "
            f"({row['Mean_ABS_SHAP']:.6f})"
        )

    # --------------------------------------------------------
    # Return results
    # --------------------------------------------------------

    return importance


# ============================================================
# 3. FILE PATHS
# ============================================================

CARDIO_X_TEST = os.path.join(
    DATA_PROCESSED,
    "cardio_X_test.csv"
)

CARDIO_MODEL = os.path.join(
    DATA_MODELS,
    "cardio_random_forest.pkl"
)

CARDIO_THRESHOLD = os.path.join(
    DATA_MODELS,
    "cardio_threshold.pkl"
)


DIABETES_X_TEST = os.path.join(
    DATA_PROCESSED,
    "diabetes_X_test.csv"
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

HYPERTENSION_MODEL = os.path.join(
    DATA_MODELS,
    "hypertension_random_forest.pkl"
)

HYPERTENSION_THRESHOLD = os.path.join(
    DATA_MODELS,
    "hypertension_threshold.pkl"
)


# ============================================================
# 4. CHECK REQUIRED FILES
# ============================================================

print("\n")
print("=" * 70)
print("CHECKING REQUIRED FILES")
print("=" * 70)


required_files = {
    "Cardio X_test": CARDIO_X_TEST,
    "Cardio model": CARDIO_MODEL,
    "Cardio threshold": CARDIO_THRESHOLD,

    "Diabetes X_test": DIABETES_X_TEST,
    "Diabetes model": DIABETES_MODEL,
    "Diabetes threshold": DIABETES_THRESHOLD,

    "Hypertension X_test": HYPERTENSION_X_TEST,
    "Hypertension model": HYPERTENSION_MODEL,
    "Hypertension threshold": HYPERTENSION_THRESHOLD
}


for name, path in required_files.items():

    if os.path.exists(path):

        print(f"[OK] {name}: {path}")

    else:

        print(f"[ERROR] Missing: {name}")
        print(path)

        raise FileNotFoundError(
            f"Required file not found: {path}"
        )


print("\nAll required files are available.")


# ============================================================
# 5. CARDIO SHAP
# ============================================================

cardio_importance = analyze_model(
    name="Cardio",
    X_test_path=CARDIO_X_TEST,
    model_path=CARDIO_MODEL,
    threshold_path=CARDIO_THRESHOLD,
    sample_size=300
)


# ============================================================
# 6. DIABETES SHAP
# ============================================================

diabetes_importance = analyze_model(
    name="Diabetes",
    X_test_path=DIABETES_X_TEST,
    model_path=DIABETES_MODEL,
    threshold_path=DIABETES_THRESHOLD,
    sample_size=154
)


# ============================================================
# 7. HYPERTENSION SHAP
# ============================================================

hypertension_importance = analyze_model(
    name="Hypertension",
    X_test_path=HYPERTENSION_X_TEST,
    model_path=HYPERTENSION_MODEL,
    threshold_path=HYPERTENSION_THRESHOLD,
    sample_size=300
)


# ============================================================
# 8. FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("XAI - SHAP ANALYSIS SUMMARY")
print("=" * 70)


print("\nCardio - Top 5:")
print(
    cardio_importance.head(5).to_string(
        index=False
    )
)


print("\nDiabetes - Top 5:")
print(
    diabetes_importance.head(5).to_string(
        index=False
    )
)


print("\nHypertension - Top 5:")
print(
    hypertension_importance.head(5).to_string(
        index=False
    )
)


print("\n")
print("=" * 70)
print("XAI MODULE CHECK")
print("=" * 70)

print("[PASSED] Cardio SHAP analysis completed.")
print("[PASSED] Diabetes SHAP analysis completed.")
print("[PASSED] Hypertension SHAP analysis completed.")
print("[PASSED] SHAP importance files generated.")
print("[PASSED] SHAP visualization files generated.")

print("\n")
print("=" * 70)
print("EXPLAINABLE AI - SHAP COMPLETED SUCCESSFULLY")
print("=" * 70)
