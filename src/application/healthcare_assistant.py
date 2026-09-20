"""
AI Healthcare Assistant - Application Layer
-------------------------------------------

Application layer responsible for:

1. Receiving patient input.
2. Preparing model-specific inputs.
3. Calling Prediction Layer.
4. Receiving disease probabilities.
5. Calling Risk Decision Engine.
6. Formatting the final healthcare risk assessment.

Important:
This application provides risk estimates for decision support.
It does NOT provide an autonomous medical diagnosis.
"""
import pandas as pd
from pathlib import Path
import sys

# ============================================================
# 1. PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from recommendation.recommendation_engine import (
    generate_recommendations
)
# ============================================================
# 2. IMPORT PREDICTION LAYER
# ============================================================

from prediction.prediction_layer import predict_all


# ============================================================
# 3. IMPORT DECISION ENGINE
# ============================================================

from decision_engine.decision_engine import make_decision


# ============================================================
# 4. HEALTHCARE ASSISTANT
# ============================================================

class HealthcareAssistant:

    def __init__(self):

        print("=" * 70)
        print("AI HEALTHCARE ASSISTANT - APPLICATION LAYER")
        print("=" * 70)

        print("\nProject root:")
        print(PROJECT_ROOT)

        print("\n[OK] Application layer initialized.")

    # ========================================================
    # 5. PREDICTION
    # ============================================================

    def predict(self, patient):
        """
        Run the existing Prediction Layer.

        The Application Layer receives one patient dictionary,
        then converts the data into three DataFrames because
        the Prediction Layer requires pandas.DataFrame input.

        Returns
        -------
        dict
            Prediction results returned by Prediction Layer.
        """

        if not isinstance(patient, dict):
            raise TypeError(
                "Patient input must be a dictionary."
            )

        if not patient:
            raise ValueError(
                "Patient input cannot be empty."
            )

        # ========================================================
        # CARDIO DATA
        # ========================================================

        cardio_data = pd.DataFrame([{
            "age": patient["age"],
            "gender": patient["gender"],
            "height": patient["height"],
            "weight": patient["weight"],
            "ap_hi": patient["ap_hi"],
            "ap_lo": patient["ap_lo"],
            "cholesterol": patient["cholesterol"],
            "gluc": patient["gluc"],
            "smoke": patient["smoke"],
            "alco": patient["alco"],
            "active": patient["active"]
        }])

        # ========================================================
        # DIABETES DATA
        # ========================================================

        diabetes_data = pd.DataFrame([{
            "Pregnancies": patient["Pregnancies"],
            "Glucose": patient["Glucose"],
            "BloodPressure": patient["BloodPressure"],
            "SkinThickness": patient["SkinThickness"],
            "Insulin": patient["Insulin"],
            "BMI": patient["BMI"],
            "DiabetesPedigreeFunction":
                patient["DiabetesPedigreeFunction"],
            "Age": patient["Age"]
        }])

        # ========================================================
        # HYPERTENSION DATA
        # ========================================================

        hypertension_data = pd.DataFrame([{
            "male": patient["male"],
            "age": patient["age"],
            "currentSmoker": patient["currentSmoker"],
            "cigsPerDay": patient["cigsPerDay"],
            "BPMeds": patient["BPMeds"],
            "diabetes": patient["diabetes"],
            "totChol": patient["totChol"],
            "sysBP": patient["sysBP"],
            "diaBP": patient["diaBP"],
            "BMI": patient["BMI"],
            "heartRate": patient["heartRate"],
            "glucose": patient["glucose"]
        }])

        # ========================================================
        # CALL EXISTING PREDICTION LAYER
        # ========================================================

        prediction_result = predict_all(
            cardio_data,
            diabetes_data,
            hypertension_data
        )

        # ========================================================
        # VALIDATE RESULT
        # ========================================================

        if not isinstance(prediction_result, dict):
            raise TypeError(
                "Prediction Layer must return a dictionary."
            )

        return prediction_result

    # ========================================================
    # 6. RISK DECISION
    # ============================================================

    def make_risk_decision(
            self,
            prediction_result
    ):
        """
        Convert Prediction Layer results into
        interpretable healthcare risk levels.

        The Prediction Layer returns:

        {
            "cardio": {...},
            "diabetes": {...},
            "hypertension": {...}
        }

        The existing Decision Engine requires
        the three disease probabilities.
        """

        if not isinstance(
                prediction_result,
                dict
        ):
            raise TypeError(
                "Prediction result must be a dictionary."
            )

        # --------------------------------------------------------
        # Validate disease results
        # --------------------------------------------------------

        required_diseases = [
            "cardio",
            "diabetes",
            "hypertension"
        ]

        for disease in required_diseases:

            if disease not in prediction_result:
                raise KeyError(
                    f"Prediction result is missing "
                    f"'{disease}'."
                )

            if not isinstance(
                    prediction_result[disease],
                    dict
            ):
                raise TypeError(
                    f"Prediction result for '{disease}' "
                    f"must be a dictionary."
                )

        # --------------------------------------------------------
        # Extract probabilities
        # --------------------------------------------------------

        cardio_result = prediction_result["cardio"]

        diabetes_result = prediction_result["diabetes"]

        hypertension_result = prediction_result["hypertension"]

        # --------------------------------------------------------
        # Find probability field
        # --------------------------------------------------------

        if "probability" not in cardio_result:
            raise KeyError(
                "Cardio prediction result does not "
                "contain 'probability'."
            )

        if "probability" not in diabetes_result:
            raise KeyError(
                "Diabetes prediction result does not "
                "contain 'probability'."
            )

        if "probability" not in hypertension_result:
            raise KeyError(
                "Hypertension prediction result does not "
                "contain 'probability'."
            )

        cardio_probability = float(
            cardio_result["probability"]
        )

        diabetes_probability = float(
            diabetes_result["probability"]
        )

        hypertension_probability = float(
            hypertension_result["probability"]
        )

        # --------------------------------------------------------
        # Validate probabilities
        # --------------------------------------------------------

        probabilities = {
            "cardio": cardio_probability,
            "diabetes": diabetes_probability,
            "hypertension": hypertension_probability
        }

        for disease, probability in probabilities.items():

            if not 0 <= probability <= 1:
                raise ValueError(
                    f"{disease} probability must be "
                    f"between 0 and 1. "
                    f"Received: {probability}"
                )

        # --------------------------------------------------------
        # Call existing Risk Decision Engine
        # --------------------------------------------------------

        decision = make_decision(
            cardio_probability,
            diabetes_probability,
            hypertension_probability
        )

        # --------------------------------------------------------
        # Validate Decision Engine output
        # --------------------------------------------------------

        if not isinstance(
                decision,
                dict
        ):
            raise TypeError(
                "Risk Decision Engine must return "
                "a dictionary."
            )

        return decision

    # ========================================================
    # 7. FORMAT RESPONSE
    # ========================================================

    def format_response(
        self,
        prediction_result,
        decision
    ):
        """
        Format the final application response.
        """

        response = {
            "prediction": prediction_result,
            "decision": decision,
            "message": (
                "These results are AI-generated "
                "risk estimates and should not be "
                "considered a medical diagnosis."
            )
        }

        return response

    # ========================================================
    # 8. COMPLETE ASSESSMENT
    # ========================================================

    def assess(self, patient):
        """
        Execute the complete application pipeline.

        Patient
            ↓
        Prediction Layer
            ↓
        Risk Decision Engine
            ↓
        Recommendation Engine
            ↓
        Application Response
        """

        # ----------------------------------------------------
        # Step 1: Prediction
        # ----------------------------------------------------

        prediction_result = self.predict(
            patient
        )

        # ----------------------------------------------------
        # Step 2: Risk Decision
        # ----------------------------------------------------

        decision = self.make_risk_decision(
            prediction_result
        )

        # ----------------------------------------------------
        # Step 3: Generate Recommendations
        # ----------------------------------------------------

        recommendations = generate_recommendations(
            decision
        )

        # ----------------------------------------------------
        # Step 4: Format response
        # ----------------------------------------------------

        response = self.format_response(
            prediction_result,
            decision
        )

        # ----------------------------------------------------
        # Step 5: Add recommendations
        # ----------------------------------------------------

        response["recommendations"] = recommendations

        return response

# ============================================================
# 9. DISPLAY RESULT
# ============================================================

# 8. DISPLAY ASSESSMENT
# ============================================================

def display_assessment(assessment):
    """
    Display the final healthcare assessment.

    The assessment contains:
        - prediction results
        - disease risk levels
        - overall risk
        - high-risk count
        - moderate-risk count
        - priority disease
        - recommendations
    """

    if not isinstance(assessment, dict):
        raise TypeError(
            "Assessment must be a dictionary."
        )

    prediction = assessment.get("prediction")
    decision = assessment.get("decision")
    recommendations = assessment.get(
        "recommendations"
    )

    if not isinstance(prediction, dict):
        raise TypeError(
            "Assessment prediction must be a dictionary."
        )

    if not isinstance(decision, dict):
        raise TypeError(
            "Assessment decision must be a dictionary."
        )

    if not isinstance(recommendations, dict):
        raise TypeError(
            "Assessment recommendations must be a dictionary."
        )

    print("\n")
    print("=" * 70)
    print("AI HEALTHCARE ASSISTANT - FINAL ASSESSMENT")
    print("=" * 70)

    # ========================================================
    # DISEASE PROBABILITIES
    # ========================================================

    print("\nDisease probabilities:")

    disease_mapping = {
        "cardio": "Cardiovascular Disease",
        "diabetes": "Diabetes",
        "hypertension": "Hypertension"
    }

    for disease_key, disease_name in disease_mapping.items():

        disease_result = prediction.get(
            disease_key
        )

        if not isinstance(
            disease_result,
            dict
        ):
            print(
                f"{disease_name:24}: N/A"
            )
            continue

        probability = disease_result.get(
            "probability"
        )

        if probability is None:
            print(
                f"{disease_name:24}: N/A"
            )
        else:
            print(
                f"{disease_name:24}: "
                f"{float(probability):.2%}"
            )

    # ========================================================
    # DISEASE RISK LEVELS
    # ========================================================

    print("\n" + "-" * 70)
    print("DISEASE RISK LEVELS")
    print("-" * 70)

    diseases = decision.get(
        "diseases",
        {}
    )

    # Decision Engine uses:
    # cardiovascular
    # diabetes
    # hypertension

    decision_mapping = {
        "cardiovascular":
            "Cardiovascular Disease",
        "diabetes":
            "Diabetes",
        "hypertension":
            "Hypertension"
    }

    for disease_key, disease_name in (
        decision_mapping.items()
    ):

        disease_result = diseases.get(
            disease_key
        )

        if not isinstance(
            disease_result,
            dict
        ):
            print(
                f"{disease_name:24}: N/A"
            )
            continue

        risk_level = disease_result.get(
            "risk_level",
            "N/A"
        )

        risk_score = disease_result.get(
            "risk_score"
        )

        if risk_score is not None:

            print(
                f"{disease_name:24}: "
                f"{risk_level} "
                f"({float(risk_score):.2f}%)"
            )

        else:

            print(
                f"{disease_name:24}: "
                f"{risk_level}"
            )

    # ========================================================
    # OVERALL RISK ASSESSMENT
    # ========================================================

    print("\n" + "-" * 70)
    print("OVERALL RISK ASSESSMENT")
    print("-" * 70)

    print(
        "Overall risk:",
        decision.get(
            "overall_risk",
            "N/A"
        )
    )

    print(
        "High-risk conditions:",
        decision.get(
            "high_risk_count",
            0
        )
    )

    print(
        "Moderate-risk conditions:",
        decision.get(
            "moderate_risk_count",
            0
        )
    )

    print(
        "Priority disease:",
        decision.get(
            "priority_disease",
            "N/A"
        )
    )

    # ========================================================
    # RECOMMENDATIONS
    # ========================================================

    print("\n" + "-" * 70)
    print("HEALTH RECOMMENDATIONS")
    print("-" * 70)

    disease_recommendations = recommendations.get(
        "disease_recommendations",
        {}
    )

    recommendation_mapping = {
        "cardiovascular":
            "Cardiovascular Disease",
        "diabetes":
            "Diabetes",
        "hypertension":
            "Hypertension"
    }

    for disease_key, disease_name in (
        recommendation_mapping.items()
    ):

        disease_recommendation = (
            disease_recommendations.get(
                disease_key
            )
        )

        if not isinstance(
            disease_recommendation,
            dict
        ):
            continue

        risk_level = disease_recommendation.get(
            "risk_level",
            "N/A"
        )

        recommendation_list = (
            disease_recommendation.get(
                "recommendations",
                []
            )
        )

        print(
            f"\n{disease_name}"
        )

        print(
            f"Risk level: {risk_level}"
        )

        for recommendation in (
            recommendation_list
        ):

            print(
                f"  - {recommendation}"
            )

    # ========================================================
    # OVERALL RECOMMENDATIONS
    # ========================================================

    overall_recommendations = (
        recommendations.get(
            "overall_recommendations",
            []
        )
    )

    if overall_recommendations:

        print("\nOverall recommendations:")

        for recommendation in (
            overall_recommendations
        ):

            print(
                f"  - {recommendation}"
            )

    # ========================================================
    # SAFETY / DISCLAIMER
    # ========================================================

    print("\n" + "-" * 70)
    print("IMPORTANT")
    print("-" * 70)

    disclaimer = recommendations.get(
        "disclaimer"
    )

    if disclaimer:

        print(disclaimer)

    else:

        print(
            "These results are AI-generated risk "
            "estimates and are not medical diagnoses."
        )

        print(
            "Final medical decisions should be made "
            "by qualified healthcare professionals."
        )

    print("\n" + "=" * 70)
    print("APPLICATION ASSESSMENT COMPLETED")
    print("=" * 70)


# ============================================================
# 10. INTEGRATION TEST
# ============================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 70)
    print("AI HEALTHCARE ASSISTANT - APPLICATION INTEGRATION TEST")
    print("=" * 70)

    # --------------------------------------------------------
    # Test patient
    # --------------------------------------------------------

    test_patient = {

        # Cardio
        "age": 55,
        "gender": 1,
        "height": 170,
        "weight": 80,
        "ap_hi": 145,
        "ap_lo": 90,
        "cholesterol": 2,
        "gluc": 1,
        "smoke": 0,
        "alco": 0,
        "active": 1,

        # Diabetes
        "Pregnancies": 3,
        "Glucose": 140,
        "BloodPressure": 80,
        "SkinThickness": 35,
        "Insulin": 120,
        "BMI": 32.0,
        "DiabetesPedigreeFunction": 0.5,
        "Age": 55,

        # Hypertension
        "male": 1,
        "currentSmoker": 0,
        "cigsPerDay": 0,
        "BPMeds": 0,
        "diabetes": 0,
        "totChol": 220,
        "sysBP": 145,
        "diaBP": 90,
        "heartRate": 75,
        "glucose": 140
    }

    # --------------------------------------------------------
    # Create assistant
    # --------------------------------------------------------

    assistant = HealthcareAssistant()

    print(
        "\n[OK] Healthcare Assistant created."
    )

    # --------------------------------------------------------
    # Run complete assessment
    # --------------------------------------------------------

    assessment = assistant.assess(
        test_patient
    )

    # --------------------------------------------------------
    # Display result
    # --------------------------------------------------------

    display_assessment(
        assessment
    )