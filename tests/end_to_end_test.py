"""
AI Healthcare Assistant
End-to-End Test
----------------------

This test verifies the complete healthcare assistant
pipeline from patient input to final recommendations.

Pipeline:

Patient Input
    ↓
Application Layer
    ↓
Prediction Layer
    ↓
Risk Decision Engine
    ↓
Recommendation Engine
    ↓
Final Assessment
"""


# ============================================================
# 1. IMPORT
# ============================================================

import sys
from pathlib import Path


# ============================================================
# 2. PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


# ============================================================
# 3. IMPORT APPLICATION LAYER
# ============================================================

from application.healthcare_assistant import (
    HealthcareAssistant
)


# ============================================================
# 4. TEST PATIENT
# ============================================================

test_patient = {

    # --------------------------------------------------------
    # Cardiovascular features
    # --------------------------------------------------------

    "age": 55,
    "gender": 2,
    "height": 170,
    "weight": 75,
    "ap_hi": 140,
    "ap_lo": 90,
    "cholesterol": 2,
    "gluc": 1,
    "smoke": 0,
    "alco": 0,
    "active": 1,

    # --------------------------------------------------------
    # Diabetes features
    # --------------------------------------------------------

    "Pregnancies": 2,
    "Glucose": 148,
    "BloodPressure": 90,
    "SkinThickness": 35,
    "Insulin": 120,
    "BMI": 32.0,
    "DiabetesPedigreeFunction": 0.6,
    "Age": 55,

    # --------------------------------------------------------
    # Hypertension features
    # --------------------------------------------------------

    "male": 1,
    "currentSmoker": 0,
    "cigsPerDay": 0,
    "BPMeds": 0,
    "diabetes": 0,
    "totChol": 230,
    "sysBP": 140,
    "diaBP": 90,
    "heartRate": 75,
    "glucose": 110
}


# ============================================================
# 5. VALIDATION HELPERS
# ============================================================

def check(condition, message):
    """
    Print PASS / FAIL for a test condition.
    """

    if condition:
        print(f"[PASS] {message}")
        return True

    print(f"[FAIL] {message}")
    return False


# ============================================================
# 6. END-TO-END TEST
# ============================================================

def run_end_to_end_test():

    print("=" * 70)
    print("AI HEALTHCARE ASSISTANT - END-TO-END TEST")
    print("=" * 70)

    print("\nProject root:")
    print(PROJECT_ROOT)

    print("\n" + "-" * 70)
    print("STEP 1 - INITIALIZE APPLICATION")
    print("-" * 70)

    assistant = HealthcareAssistant()

    check(
        assistant is not None,
        "Healthcare Assistant initialized"
    )

    print("\n" + "-" * 70)
    print("STEP 2 - RUN COMPLETE PIPELINE")
    print("-" * 70)

    assessment = assistant.assess(
        test_patient
    )

    check(
        isinstance(assessment, dict),
        "Application returned a dictionary"
    )

    # --------------------------------------------------------
    # Prediction validation
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("STEP 3 - VALIDATE PREDICTION")
    print("-" * 70)

    prediction = assessment.get(
        "prediction"
    )

    check(
        isinstance(prediction, dict),
        "Prediction result exists"
    )

    required_diseases = {
        "cardio",
        "diabetes",
        "hypertension"
    }

    check(
        required_diseases.issubset(
            prediction.keys()
        ),
        "All three disease predictions exist"
    )

    for disease in required_diseases:

        result = prediction.get(
            disease
        )

        check(
            isinstance(result, dict),
            f"{disease} prediction is valid"
        )

        if isinstance(result, dict):

            probability = result.get(
                "probability"
            )

            check(
                probability is not None,
                f"{disease} probability exists"
            )

            if probability is not None:

                check(
                    0 <= float(probability) <= 1,
                    f"{disease} probability is between 0 and 1"
                )

    # --------------------------------------------------------
    # Decision validation
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("STEP 4 - VALIDATE RISK DECISION")
    print("-" * 70)

    decision = assessment.get(
        "decision"
    )

    check(
        isinstance(decision, dict),
        "Decision result exists"
    )

    if isinstance(decision, dict):

        diseases = decision.get(
            "diseases"
        )

        check(
            isinstance(diseases, dict),
            "Disease risk results exist"
        )

        required_decisions = {
            "cardiovascular",
            "diabetes",
            "hypertension"
        }

        if isinstance(diseases, dict):

            check(
                required_decisions.issubset(
                    diseases.keys()
                ),
                "All three disease risk assessments exist"
            )

            for disease in required_decisions:

                result = diseases.get(
                    disease
                )

                if isinstance(result, dict):

                    risk_level = result.get(
                        "risk_level"
                    )

                    check(
                        risk_level in {
                            "LOW",
                            "MODERATE",
                            "HIGH"
                        },
                        f"{disease} risk level is valid"
                    )

        overall_risk = decision.get(
            "overall_risk"
        )

        check(
            overall_risk in {
                "LOW",
                "MODERATE",
                "HIGH"
            },
            "Overall risk is valid"
        )

    # --------------------------------------------------------
    # Recommendation validation
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("STEP 5 - VALIDATE RECOMMENDATIONS")
    print("-" * 70)

    recommendations = assessment.get(
        "recommendations"
    )

    check(
        isinstance(recommendations, dict),
        "Recommendation result exists"
    )

    if isinstance(recommendations, dict):

        disease_recommendations = (
            recommendations.get(
                "disease_recommendations"
            )
        )

        check(
            isinstance(
                disease_recommendations,
                dict
            ),
            "Disease recommendations exist"
        )

        if isinstance(
            disease_recommendations,
            dict
        ):

            check(
                required_decisions.issubset(
                    disease_recommendations.keys()
                ),
                "Recommendations exist for all three diseases"
            )

        overall_recommendations = (
            recommendations.get(
                "overall_recommendations"
            )
        )

        check(
            isinstance(
                overall_recommendations,
                list
            ),
            "Overall recommendations exist"
        )

        disclaimer = recommendations.get(
            "disclaimer"
        )

        check(
            isinstance(disclaimer, str)
            and len(disclaimer) > 0,
            "Medical disclaimer exists"
        )

    # --------------------------------------------------------
    # Final assessment validation
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("STEP 6 - FINAL ASSESSMENT")
    print("-" * 70)

    check(
        "prediction" in assessment,
        "Final assessment contains prediction"
    )

    check(
        "decision" in assessment,
        "Final assessment contains decision"
    )

    check(
        "recommendations" in assessment,
        "Final assessment contains recommendations"
    )

    # --------------------------------------------------------
    # Display summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("END-TO-END TEST SUMMARY")
    print("=" * 70)

    if isinstance(prediction, dict):

        print("\nDisease probabilities:")

        for disease, result in prediction.items():

            if isinstance(result, dict):

                probability = result.get(
                    "probability"
                )

                if probability is not None:

                    print(
                        f"  {disease:15}: "
                        f"{float(probability):.2%}"
                    )

    if isinstance(decision, dict):

        print(
            "\nOverall risk:",
            decision.get(
                "overall_risk",
                "N/A"
            )
        )

        print(
            "Priority disease:",
            decision.get(
                "priority_disease",
                "N/A"
            )
        )

    print("\n" + "=" * 70)
    print("END-TO-END TEST COMPLETED")
    print("=" * 70)


# ============================================================
# 7. MAIN
# ============================================================

if __name__ == "__main__":

    run_end_to_end_test()