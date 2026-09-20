"""
Model Input Inspection
----------------------

Inspect the exact feature structure expected by
the trained Random Forest models.

This file does NOT modify any model.
"""

from pathlib import Path

import joblib


# ============================================================
# 1. PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_MODELS = PROJECT_ROOT / "data" / "models"


# ============================================================
# 2. MODEL PATHS
# ============================================================

MODELS = {
    "Cardio": DATA_MODELS / "cardio_random_forest.pkl",
    "Diabetes": DATA_MODELS / "diabetes_random_forest.pkl",
    "Hypertension": DATA_MODELS / "hypertension_random_forest.pkl"
}


# ============================================================
# 3. INSPECT MODEL
# ============================================================

def inspect_model(name, path):

    print("\n" + "-" * 70)
    print(f"{name.upper()} MODEL")
    print("-" * 70)

    print(f"File: {path}")

    model = joblib.load(path)

    print(f"\nModel type:")
    print(type(model).__name__)

    print(f"\nNumber of features:")
    print(model.n_features_in_)

    if hasattr(model, "feature_names_in_"):

        print("\nExpected features:")

        for index, feature in enumerate(
            model.feature_names_in_,
            start=1
        ):
            print(f"{index:2}. {feature}")

    else:

        print(
            "\n[WARNING] Model does not contain "
            "feature_names_in_."
        )

    if hasattr(model, "classes_"):

        print("\nClasses:")
        print(model.classes_)

    if hasattr(model, "n_estimators"):

        print("\nNumber of trees:")
        print(model.n_estimators)


# ============================================================
# 4. MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("AI HEALTHCARE ASSISTANT - MODEL INPUT INSPECTION")
    print("=" * 70)

    print("\nProject root:")
    print(PROJECT_ROOT)

    for name, path in MODELS.items():

        if not path.exists():

            print(
                f"\n[MISSING] {name} model: {path}"
            )

            continue

        try:

            inspect_model(
                name,
                path
            )

            print(
                f"\n[OK] {name} inspection completed."
            )

        except Exception as error:

            print(
                f"\n[ERROR] Could not inspect "
                f"{name} model."
            )

            print(
                f"Reason: {repr(error)}"
            )

    print("\n" + "=" * 70)
    print("MODEL INPUT INSPECTION COMPLETED")
    print("=" * 70)