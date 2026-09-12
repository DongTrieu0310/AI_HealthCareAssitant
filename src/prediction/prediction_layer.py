"""
AI Healthcare Assistant - Prediction Layer
-------------------------------------------

This module provides the prediction layer between
the application and trained machine learning models.

Responsibilities:
1. Load trained ML models.
2. Load model-specific decision thresholds.
3. Validate model input features.
4. Generate disease probabilities.
5. Return structured prediction results.

This module does NOT:
- Train models.
- Perform fairness analysis.
- Perform explainability analysis.
- Make final medical decisions.
- Replace healthcare professionals.
"""

import joblib
import pandas as pd

from pathlib import Path


# ============================================================
# 1. PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_MODELS = PROJECT_ROOT / "data" / "models"


# ============================================================
# 2. MODEL PATHS
# ============================================================

MODEL_PATHS = {
    "cardio": DATA_MODELS / "cardio_random_forest.pkl",
    "diabetes": DATA_MODELS / "diabetes_random_forest.pkl",
    "hypertension": DATA_MODELS / "hypertension_random_forest.pkl"
}


THRESHOLD_PATHS = {
    "cardio": DATA_MODELS / "cardio_threshold.pkl",
    "diabetes": DATA_MODELS / "diabetes_threshold.pkl",
    "hypertension": DATA_MODELS / "hypertension_threshold.pkl"
}


# ============================================================
# 2.1 PREPROCESSING TRANSFORMERS
# ============================================================

# CARDIO and DIABETES were trained on standardized data, so the same
# scaler must be applied before predicting. HYPERTENSION was trained on
# raw units and needs none.
#
# These files are produced by src/preprocessing/export_transformers.py.

SCALER_PATHS = {
    "cardio": DATA_MODELS / "cardio_scaler.pkl",
    "diabetes": DATA_MODELS / "diabetes_scaler.pkl",
}

IMPUTER_PATHS = {
    "diabetes": DATA_MODELS / "diabetes_imputer.pkl",
}

# Insulin values treated as unreliable during preprocessing.
SUSPICIOUS_INSULIN = [102.5, 169.5]


# ============================================================
# 3. EXPECTED FEATURES
# ============================================================

EXPECTED_FEATURES = {

    "cardio": [
        "age",
        "gender",
        "height",
        "weight",
        "ap_hi",
        "ap_lo",
        "cholesterol",
        "gluc",
        "smoke",
        "alco",
        "active"
    ],

    "diabetes": [
        "Pregnancies",
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
        "DiabetesPedigreeFunction",
        "Age"
    ],

    "hypertension": [
        "male",
        "age",
        "currentSmoker",
        "cigsPerDay",
        "BPMeds",
        "diabetes",
        "totChol",
        "sysBP",
        "diaBP",
        "BMI",
        "heartRate",
        "glucose"
    ]
}


# ============================================================
# 4. LOAD MODELS
# ============================================================

def load_models():
    """
    Load all trained Random Forest models.

    Returns
    -------
    dict
        Dictionary containing the three trained models.
    """

    models = {}

    for disease, path in MODEL_PATHS.items():

        if not path.exists():
            raise FileNotFoundError(
                f"Model file not found: {path}"
            )

        models[disease] = joblib.load(path)

    return models


# ============================================================
# 5. LOAD THRESHOLDS
# ============================================================

def load_thresholds():
    """
    Load decision thresholds for all disease models.

    Returns
    -------
    dict
        Dictionary containing model thresholds.
    """

    thresholds = {}

    for disease, path in THRESHOLD_PATHS.items():

        if not path.exists():
            raise FileNotFoundError(
                f"Threshold file not found: {path}"
            )

        thresholds[disease] = float(
            joblib.load(path)
        )

    return thresholds


# ============================================================
# 5.1 LOAD PREPROCESSING TRANSFORMERS
# ============================================================

def load_transformers():
    """
    Load the scalers and imputers used during training.

    Returns
    -------
    dict
        {disease: {"scaler": ..., "imputer": ...}} for every disease
        that has a transformer file on disk.

    Raises
    ------
    FileNotFoundError
        When a disease known to need a scaler has none saved. Predicting
        without it would silently produce meaningless probabilities.
    """

    transformers = {}

    for disease, path in SCALER_PATHS.items():

        if not path.exists():
            raise FileNotFoundError(
                f"Missing scaler for '{disease}': {path}\n"
                "This model was trained on standardized data. Run "
                "`python src/preprocessing/export_transformers.py` "
                "to regenerate it."
            )

        transformers.setdefault(disease, {})["scaler"] = joblib.load(path)

    for disease, path in IMPUTER_PATHS.items():

        if path.exists():
            transformers.setdefault(disease, {})["imputer"] = (
                joblib.load(path)
            )

    return transformers


def apply_transformers(patient_data, disease, transformers):
    """
    Apply the training-time preprocessing to one patient's features.

    Feeding raw values to a model trained on standardized data makes its
    output nearly constant, so this step is required for correctness —
    not an optimisation.
    """

    steps = (transformers or {}).get(disease)

    if not steps:
        return patient_data

    data = patient_data.copy()

    imputer = steps.get("imputer")

    if imputer is not None:

        if "Insulin" in data.columns:
            data["Insulin"] = data["Insulin"].replace(
                SUSPICIOUS_INSULIN,
                float("nan")
            )

        data = pd.DataFrame(
            imputer.transform(data),
            columns=data.columns,
            index=data.index
        )

    scaler = steps.get("scaler")

    if scaler is not None:
        data = pd.DataFrame(
            scaler.transform(data),
            columns=data.columns,
            index=data.index
        )

    return data


# ============================================================
# 6. VALIDATE INPUT
# ============================================================

def validate_input(
    patient_data,
    disease
):
    """
    Validate that patient input contains
    exactly the features required by a model.

    Parameters
    ----------
    patient_data : pandas.DataFrame
        Patient feature data.

    disease : str
        Disease model name.

    Returns
    -------
    pandas.DataFrame
        Validated input.
    """

    if disease not in EXPECTED_FEATURES:
        raise ValueError(
            f"Unknown disease: {disease}"
        )

    if not isinstance(
        patient_data,
        pd.DataFrame
    ):
        raise TypeError(
            "patient_data must be a pandas DataFrame."
        )

    expected = EXPECTED_FEATURES[disease]

    missing_features = [
        feature
        for feature in expected
        if feature not in patient_data.columns
    ]

    if missing_features:

        raise ValueError(
            f"{disease} input is missing features: "
            f"{missing_features}"
        )

    return patient_data[expected].copy()


# ============================================================
# 7. PREDICT SINGLE DISEASE
# ============================================================

def predict_disease(
    patient_data,
    disease,
    models=None,
    thresholds=None,
    transformers=None
):
    """
    Generate probability and prediction
    for a single disease.

    The patient data must be supplied in ORIGINAL units (mmHg, cm, kg,
    years). Any standardization the model was trained with is applied
    here, so callers never deal with scaled values.

    Returns
    -------
    dict
        Structured prediction result.
    """

    if models is None:
        models = load_models()

    if thresholds is None:
        thresholds = load_thresholds()

    if transformers is None:
        transformers = load_transformers()

    validated_data = validate_input(
        patient_data,
        disease
    )

    validated_data = apply_transformers(
        validated_data,
        disease,
        transformers
    )

    model = models[disease]

    threshold = thresholds[disease]

    probability = float(
        model.predict_proba(
            validated_data
        )[0, 1]
    )

    prediction = int(
        probability >= threshold
    )

    return {
        "disease": disease,
        "probability": probability,
        "prediction": prediction,
        "threshold": threshold
    }


# ============================================================
# 8. PREDICT ALL DISEASES
# ============================================================

def predict_all(
    cardio_data,
    diabetes_data,
    hypertension_data
):
    """
    Generate predictions for all supported diseases.

    Parameters
    ----------
    cardio_data : pandas.DataFrame
        Input features for cardiovascular model.

    diabetes_data : pandas.DataFrame
        Input features for diabetes model.

    hypertension_data : pandas.DataFrame
        Input features for hypertension model.

    Returns
    -------
    dict
        Structured prediction results.
    """

    models = load_models()
    thresholds = load_thresholds()
    transformers = load_transformers()

    cardio_result = predict_disease(
        cardio_data,
        "cardio",
        models,
        thresholds,
        transformers
    )

    diabetes_result = predict_disease(
        diabetes_data,
        "diabetes",
        models,
        thresholds,
        transformers
    )

    hypertension_result = predict_disease(
        hypertension_data,
        "hypertension",
        models,
        thresholds,
        transformers
    )

    return {
        "cardio": cardio_result,
        "diabetes": diabetes_result,
        "hypertension": hypertension_result
    }


# ============================================================
# 9. SIMPLE PROBABILITY OUTPUT
# ============================================================

def predict_probabilities(
    cardio_data,
    diabetes_data,
    hypertension_data
):
    """
    Return only disease probabilities.

    This function is useful for the
    healthcare predictor layer.
    """

    results = predict_all(
        cardio_data,
        diabetes_data,
        hypertension_data
    )

    return {
        "cardiovascular": results[
            "cardio"
        ]["probability"],

        "diabetes": results[
            "diabetes"
        ]["probability"],

        "hypertension": results[
            "hypertension"
        ]["probability"]
    }


# ============================================================
# 10. MODEL INFORMATION
# ============================================================

def get_model_information():
    """
    Return basic information about
    loaded prediction models.
    """

    models = load_models()
    thresholds = load_thresholds()

    information = {}

    for disease in models:

        model = models[disease]

        information[disease] = {
            "model_type": type(model).__name__,
            "number_of_features": (
                model.n_features_in_
                if hasattr(
                    model,
                    "n_features_in_"
                )
                else None
            ),
            "threshold": thresholds[disease],
            "expected_features": (
                EXPECTED_FEATURES[disease]
            )
        }

    return information


# ============================================================
# 11. TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("AI HEALTHCARE ASSISTANT - PREDICTION LAYER")
    print("=" * 70)

    print("\nProject root:")
    print(PROJECT_ROOT)

    # --------------------------------------------------------
    # Check model files
    # --------------------------------------------------------

    print("\nChecking model files...")

    for disease, path in MODEL_PATHS.items():

        if path.exists():
            print(
                f"[OK] {disease.capitalize()} model: "
                f"{path}"
            )
        else:
            print(
                f"[MISSING] {disease.capitalize()} model: "
                f"{path}"
            )

    # --------------------------------------------------------
    # Check threshold files
    # --------------------------------------------------------

    print("\nChecking threshold files...")

    for disease, path in THRESHOLD_PATHS.items():

        if path.exists():
            print(
                f"[OK] {disease.capitalize()} threshold: "
                f"{path}"
            )
        else:
            print(
                f"[MISSING] {disease.capitalize()} threshold: "
                f"{path}"
            )

    # --------------------------------------------------------
    # Load models
    # --------------------------------------------------------

    print("\nLoading models...")

    models = load_models()

    for disease in models:
        print(
            f"[OK] {disease.capitalize()} model loaded"
        )

    # --------------------------------------------------------
    # Load thresholds
    # --------------------------------------------------------

    print("\nLoading thresholds...")

    thresholds = load_thresholds()

    for disease, threshold in thresholds.items():

        print(
            f"[OK] {disease.capitalize()} threshold: "
            f"{threshold:.2f}"
        )

    # --------------------------------------------------------
    # Display model information
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("MODEL INFORMATION")
    print("-" * 70)

    information = get_model_information()

    for disease, info in information.items():

        print(
            f"\n{disease.upper()}"
        )

        print(
            f"Model type: "
            f"{info['model_type']}"
        )

        print(
            f"Features: "
            f"{info['number_of_features']}"
        )

        print(
            f"Threshold: "
            f"{info['threshold']:.2f}"
        )

        print(
            "Expected features:"
        )

        for feature in info[
            "expected_features"
        ]:

            print(
                f"  - {feature}"
            )

    print("\n" + "=" * 70)
    print("PREDICTION LAYER TEST COMPLETED")
    print("=" * 70)