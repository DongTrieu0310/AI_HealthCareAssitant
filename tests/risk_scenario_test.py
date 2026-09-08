"""
AI Healthcare Assistant
Risk Scenario Test

Test the complete healthcare pipeline
with different patient risk scenarios.
"""

import sys
from pathlib import Path


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


# ============================================================
# IMPORT APPLICATION
# ============================================================

from application.healthcare_assistant import (
    HealthcareAssistant
)


# ============================================================
# TEST PATIENTS
# ============================================================

base_patient = {

    # --------------------------------------------------------
    # Cardiovascular
    # --------------------------------------------------------

    "age": 40,
    "gender": 1,
    "height": 170,
    "weight": 65,
    "ap_hi": 115,
    "ap_lo": 75,
    "cholesterol": 1,
    "gluc": 1,
    "smoke": 0,
    "alco": 0,
    "active": 1,

    # --------------------------------------------------------
    # Diabetes
    # --------------------------------------------------------

    "Pregnancies": 1,
    "Glucose": 90,
    "BloodPressure": 70,
    "SkinThickness": 20,
    "Insulin": 80,
    "BMI": 22,
    "DiabetesPedigreeFunction": 0.3,
    "Age": 40,

    # --------------------------------------------------------
    # Hypertension
    # --------------------------------------------------------

    "male": 0,
    "currentSmoker": 0,
    "cigsPerDay": 0,
    "BPMeds": 0,
    "diabetes": 0,
    "totChol": 180,
    "sysBP": 115,
    "diaBP": 75,
    "BMI": 22,
    "heartRate": 70,
    "glucose": 90
}


# ============================================================
# CREATE SCENARIOS
# ============================================================

low_risk_patient = base_patient.copy()


moderate_risk_patient = base_patient.copy()

moderate_risk_patient.update({

    "age": 55,

    "ap_hi": 140,
    "ap_lo": 90,

    "cholesterol": 2,

    "Glucose": 130,
    "BMI": 30,

    "sysBP": 140,
    "diaBP": 90,

    "totChol": 230
})


high_risk_patient = base_patient.copy()

high_risk_patient.update({

    "age": 70,

    "gender": 2,

    "height": 170,
    "weight": 95,

    "ap_hi": 180,
    "ap_lo": 110,

    "cholesterol": 3,
    "gluc": 3,

    "smoke": 1,
    "alco": 1,
    "active": 0,

    "Pregnancies": 6,
    "Glucose": 200,
    "BloodPressure": 110,
    "SkinThickness": 40,
    "Insulin": 250,
    "BMI": 38,
    "DiabetesPedigreeFunction": 1.2,
    "Age": 70,

    "male": 1,
    "currentSmoker": 1,
    "cigsPerDay": 20,
    "BPMeds": 1,
    "diabetes": 1,
    "totChol": 280,

    "sysBP": 180,
    "diaBP": 110,

    "heartRate": 100,
    "glucose": 200
})


# ============================================================
# TEST FUNCTION
# ============================================================

def run_scenario(
    assistant,
    scenario_name,
    patient
):

    print("\n" + "-" * 70)
    print(
        f"SCENARIO: {scenario_name}"
    )
    print("-" * 70)

    assessment = assistant.assess(
        patient
    )

    if not isinstance(
        assessment,
        dict
    ):
        raise TypeError(
            "Assessment must be a dictionary."
        )

    decision = assessment.get(
        "decision"
    )

    if not isinstance(
        decision,
        dict
    ):
        raise TypeError(
            "Decision must be a dictionary."
        )

    overall_risk = decision.get(
        "overall_risk",
        "N/A"
    )

    priority_disease = decision.get(
        "priority_disease",
        "N/A"
    )

    print(
        f"Overall Risk      : {overall_risk}"
    )

    print(
        f"Priority Disease   : {priority_disease}"
    )
    # --------------------------------------------------------
    # Validate overall risk
    # --------------------------------------------------------

    valid_risk_levels = [
        "LOW",
        "MODERATE",
        "HIGH"
    ]

    if overall_risk not in valid_risk_levels:
        raise AssertionError(
            f"Invalid overall risk: {overall_risk}"
        )

    print(
        "[PASS] Overall risk is valid"
    )
    diseases = decision.get(
        "diseases",
        {}
    )

    for disease in [
        "cardiovascular",
        "diabetes",
        "hypertension"
    ]:

        result = diseases.get(disease)

        if not isinstance(result, dict):
            raise AssertionError(
                f"Missing disease result: {disease}"
            )

        risk_level = result.get("risk_level")

        if risk_level not in [
            "LOW",
            "MODERATE",
            "HIGH"
        ]:
            raise AssertionError(
                f"Invalid risk level for {disease}: "
                f"{risk_level}"
            )

        probability = result.get("probability")

        if not isinstance(probability, (int, float)):
            raise AssertionError(
                f"Invalid probability for {disease}"
            )

        if not 0 <= probability <= 1:
            raise AssertionError(
                f"Probability out of range for {disease}: "
                f"{probability}"
            )

        print(
            f"{disease:18}: "
            f"{risk_level} "
            f"({probability:.2%})"
        )

    print("[PASS] All disease assessments are valid")

    return assessment


# ============================================================
# MAIN TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print(
        "AI HEALTHCARE ASSISTANT - "
        "RISK SCENARIO TEST"
    )
    print("=" * 70)

    print(
        f"\nProject root:\n"
        f"{PROJECT_ROOT}"
    )

    # --------------------------------------------------------
    # Initialize
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("INITIALIZING APPLICATION")
    print("-" * 70)

    assistant = HealthcareAssistant()

    print(
        "[PASS] Healthcare Assistant initialized"
    )

    # --------------------------------------------------------
    # Run scenarios
    # --------------------------------------------------------

    low_result = run_scenario(
        assistant,
        "LOW RISK PATIENT",
        low_risk_patient
    )

    moderate_result = run_scenario(
        assistant,
        "MODERATE RISK PATIENT",
        moderate_risk_patient
    )

    high_result = run_scenario(
        assistant,
        "HIGH RISK PATIENT",
        high_risk_patient
    )

    # --------------------------------------------------------
    # Validate results
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("RISK SCENARIO TEST SUMMARY")
    print("=" * 70)

    for name, result in [
        ("LOW", low_result),
        ("MODERATE", moderate_result),
        ("HIGH", high_result)
    ]:

        decision = result.get(
            "decision",
            {}
        )

        print(
            f"{name:12}: "
            f"{decision.get('overall_risk', 'N/A')}"
        )

    print("\n" + "=" * 70)
    print(
        "RISK SCENARIO TEST COMPLETED"
    )
    print("=" * 70)