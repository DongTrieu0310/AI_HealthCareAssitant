"""
Healthcare Prediction Layer
----------------------------

This module connects trained ML models with
the Risk Decision Engine.

Responsibilities:
1. Load trained models.
2. Load decision thresholds.
3. Receive processed patient features.
4. Generate disease probabilities.
5. Send probabilities to the Decision Engine.

This module does NOT train models.
"""

from pathlib import Path

import joblib
import pandas as pd

from src.decision_engine.decision_engine import make_decision


# ============================================================
# 1. PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_MODELS = PROJECT_ROOT / "data" / "models"


# ============================================================
# 2. MODEL FILES
# ============================================================

CARDIO_MODEL_PATH = (
    DATA_MODELS / "cardio_random_forest.pkl"
)

DIABETES_MODEL_PATH = (
    DATA_MODELS / "diabetes_random_forest.pkl"
)

HYPERTENSION_MODEL_PATH = (
    DATA_MODELS / "hypertension_random_forest.pkl"
)


# ============================================================
# 3. THRESHOLD FILES
# ============================================================

CARDIO_THRESHOLD_PATH = (
    DATA_MODELS / "cardio_threshold.pkl"
)

DIABETES_THRESHOLD_PATH = (
    DATA_MODELS / "diabetes_threshold.pkl"
)

HYPERTENSION_THRESHOLD_PATH = (
    DATA_MODELS / "hypertension_threshold.pkl"
)


# ============================================================
# 4. LOAD MODELS
# ============================================================

def load_models():

    cardio_model = joblib.load(
        CARDIO_MODEL_PATH
    )

    diabetes_model = joblib.load(
        DIABETES_MODEL_PATH
    )

    hypertension_model = joblib.load(
        HYPERTENSION_MODEL_PATH
    )

    return {
        "cardio": cardio_model,
        "diabetes": diabetes_model,
        "hypertension": hypertension_model
    }


# ============================================================
# 5. LOAD THRESHOLDS
# ============================================================

def load_thresholds():

    cardio_threshold = float(
        joblib.load(
            CARDIO_THRESHOLD_PATH
        )
    )

    diabetes_threshold = float(
        joblib.load(
            DIABETES_THRESHOLD_PATH
        )
    )

    hypertension_threshold = float(
        joblib.load(
            HYPERTENSION_THRESHOLD_PATH
        )
    )

    return {
        "cardio": cardio_threshold,
        "diabetes": diabetes_threshold,
        "hypertension": hypertension_threshold
    }


# ============================================================
# 6. PREDICT SINGLE MODEL
# ============================================================

def predict_probability(model, X):

    probability = model.predict_proba(
        X
    )[:, 1]

    return float(probability[0])


# ============================================================
# 7. PREDICT ALL DISEASES
# ============================================================

def predict_all(
    cardio_data,
    diabetes_data,
    hypertension_data
):
    """
    Generate probabilities for all three diseases.

    Parameters
    ----------
    cardio_data : pandas.DataFrame
    diabetes_data : pandas.DataFrame
    hypertension_data : pandas.DataFrame

    Returns
    -------
    dict
        Disease probabilities.
    """

    models = load_models()

    cardio_probability = predict_probability(
        models["cardio"],
        cardio_data
    )

    diabetes_probability = predict_probability(
        models["diabetes"],
        diabetes_data
    )

    hypertension_probability = predict_probability(
        models["hypertension"],
        hypertension_data
    )

    return {
        "cardio_probability": cardio_probability,
        "diabetes_probability": diabetes_probability,
        "hypertension_probability": hypertension_probability
    }


# ============================================================
# 8. COMPLETE HEALTHCARE ASSESSMENT
# ============================================================

def assess_patient(
    cardio_data,
    diabetes_data,
    hypertension_data
):
    """
    Generate complete multi-disease assessment.
    """

    probabilities = predict_all(
        cardio_data,
        diabetes_data,
        hypertension_data
    )

    decision = make_decision(
        cardio_probability=
            probabilities["cardio_probability"],

        diabetes_probability=
            probabilities["diabetes_probability"],

        hypertension_probability=
            probabilities["hypertension_probability"]
    )

    return {
        "probabilities": probabilities,
        "decision": decision
    }


# ============================================================
# 9. TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("AI HEALTHCARE ASSISTANT - PREDICTION LAYER")
    print("=" * 70)

    print("\nProject root:")
    print(PROJECT_ROOT)

    print("\nModel files:")

    print(
        "[OK]",
        CARDIO_MODEL_PATH
    )

    print(
        "[OK]",
        DIABETES_MODEL_PATH
    )

    print(
        "[OK]",
        HYPERTENSION_MODEL_PATH
    )

    print("\nThreshold files:")

    print(
        "[OK]",
        CARDIO_THRESHOLD_PATH
    )

    print(
        "[OK]",
        DIABETES_THRESHOLD_PATH
    )

    print(
        "[OK]",
        HYPERTENSION_THRESHOLD_PATH
    )

    print("\nLoading models...")

    models = load_models()

    print("[OK] Cardio model loaded")
    print("[OK] Diabetes model loaded")
    print("[OK] Hypertension model loaded")

    print("\nLoading thresholds...")

    thresholds = load_thresholds()

    print(
        f"[OK] Cardio threshold: "
        f"{thresholds['cardio']:.2f}"
    )

    print(
        f"[OK] Diabetes threshold: "
        f"{thresholds['diabetes']:.2f}"
    )

    print(
        f"[OK] Hypertension threshold: "
        f"{thresholds['hypertension']:.2f}"
    )

    print("\n" + "=" * 70)
    print("PREDICTION LAYER TEST COMPLETED")
    print("=" * 70)