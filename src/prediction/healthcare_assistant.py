"""
AI Healthcare Assistant - Integration Layer
--------------------------------------------

Connects:
    Test data
        ↓
    Prediction Layer
        ↓
    Risk Decision Engine

This module does NOT train models.
"""

from pathlib import Path

import pandas as pd

from src.prediction.healthcare_predictor import (
    predict_all
)


# ============================================================
# 1. PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PROCESSED = (
    PROJECT_ROOT / "data" / "processed"
)


# ============================================================
# 2. TEST DATA
# ============================================================

CARDIO_X_TEST = (
    DATA_PROCESSED / "cardio_X_test.csv"
)

DIABETES_X_TEST = (
    DATA_PROCESSED / "diabetes_X_test.csv"
)

HYPERTENSION_X_TEST = (
    DATA_PROCESSED / "hypertension_X_test.csv"
)


# ============================================================
# 3. LOAD TEST PATIENT
# ============================================================

def load_test_patient(index=0):
    """
    Load one patient from each X_test dataset.

    This function is used only for integration testing.
    """

    cardio = pd.read_csv(
        CARDIO_X_TEST
    )

    diabetes = pd.read_csv(
        DIABETES_X_TEST
    )

    hypertension = pd.read_csv(
        HYPERTENSION_X_TEST
    )

    if index < 0:
        raise ValueError(
            "Patient index cannot be negative."
        )

    if index >= len(cardio):
        raise IndexError(
            "Cardio patient index is out of range."
        )

    if index >= len(diabetes):
        raise IndexError(
            "Diabetes patient index is out of range."
        )

    if index >= len(hypertension):
        raise IndexError(
            "Hypertension patient index is out of range."
        )

    return (
        cardio.iloc[[index]],
        diabetes.iloc[[index]],
        hypertension.iloc[[index]]
    )


# ============================================================
# 4. RUN HEALTHCARE ASSESSMENT
# ============================================================

def assess_test_patient(index=0):
    """
    Run the complete prediction pipeline
    using one patient from the test datasets.
    """

    (
        cardio_data,
        diabetes_data,
        hypertension_data
    ) = load_test_patient(index)

    result = predict_all(
        cardio_data,
        diabetes_data,
        hypertension_data
    )

    return result


# ============================================================
# 5. DISPLAY RESULT
# ============================================================

def print_result(result):

    print("\n")
    print("=" * 70)
    print("AI HEALTHCARE ASSISTANT - PREDICTION RESULT")
    print("=" * 70)

    print("\nDisease probabilities:")

    print(
        f"Cardiovascular Disease : "
        f"{result['cardio_probability']:.2%}"
    )

    print(
        f"Diabetes               : "
        f"{result['diabetes_probability']:.2%}"
    )

    print(
        f"Hypertension           : "
        f"{result['hypertension_probability']:.2%}"
    )

    print("\n" + "=" * 70)
    print("PREDICTION PIPELINE TEST COMPLETED")
    print("=" * 70)


# ============================================================
# 6. MAIN TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("AI HEALTHCARE ASSISTANT - INTEGRATION TEST")
    print("=" * 70)

    print("\nProject root:")
    print(PROJECT_ROOT)

    print("\nLoading test patient...")

    result = assess_test_patient(
        index=0
    )

    print_result(
        result
    )