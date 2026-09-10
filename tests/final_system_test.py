import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from application.healthcare_assistant import HealthcareAssistant


def print_result(test_name, passed):
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {test_name}")


def check(condition, test_name):
    print_result(test_name, condition)
    return condition


def create_test_patient():
    return {
        "age": 55, "gender": 1, "height": 170, "weight": 75,
        "ap_hi": 140, "ap_lo": 90, "cholesterol": 2, "gluc": 1,
        "smoke": 0, "alco": 0, "active": 1,
        "Pregnancies": 2, "Glucose": 130, "BloodPressure": 85,
        "SkinThickness": 25, "Insulin": 100, "BMI": 28,
        "DiabetesPedigreeFunction": 0.5, "Age": 55,
        "male": 1, "currentSmoker": 0, "cigsPerDay": 0,
        "BPMeds": 0, "diabetes": 0, "totChol": 220,
        "sysBP": 140, "diaBP": 90, "heartRate": 75, "glucose": 130,
    }


def main():
    print("=" * 60)
    print("FINAL SYSTEM TEST")
    print("=" * 60)

    all_passed = True

    try:
        assistant = HealthcareAssistant()
        all_passed &= check(assistant is not None, "HealthcareAssistant initialization")
    except Exception as e:
        print_result("HealthcareAssistant initialization", False)
        print(f"Error: {e}")
        return 1

    patient = create_test_patient()
    all_passed &= check(isinstance(patient, dict), "Patient input creation")
    required_patient_features = [
        "age", "gender", "height", "weight", "ap_hi", "ap_lo",
        "cholesterol", "gluc", "smoke", "alco", "active",
        "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
        "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
        "male", "currentSmoker", "cigsPerDay", "BPMeds", "diabetes",
        "totChol", "sysBP", "diaBP", "heartRate", "glucose"
    ]

    missing_features = [
        feature for feature in required_patient_features
        if feature not in patient
    ]

    all_passed &= check(
        not missing_features,
        "Patient contains all required features"
    )

    if missing_features:
        print(f"Missing features: {missing_features}")

    try:
        assessment = assistant.assess(patient)
        all_passed &= check(isinstance(assessment, dict), "Complete assessment returns dictionary")
    except Exception as e:
        print_result("Complete assessment", False)
        print(f"Error: {e}")
        return 1

    prediction = assessment.get("prediction")
    all_passed &= check(isinstance(prediction, dict), "Prediction result exists")

    prediction_keys = ["cardio", "diabetes", "hypertension"]
    for disease in prediction_keys:
        result = prediction.get(disease)
        all_passed &= check(isinstance(result, dict), f"{disease} prediction exists")
        if isinstance(result, dict):
            probability = result.get("probability")
            all_passed &= check(probability is not None, f"{disease} probability exists")
            try:
                valid = probability is not None and 0 <= float(probability) <= 1
            except (TypeError, ValueError):
                valid = False
            all_passed &= check(valid, f"{disease} probability is between 0 and 1")

    decision = assessment.get("decision")
    all_passed &= check(isinstance(decision, dict), "Decision result exists")

    decision_keys = ["cardiovascular", "diabetes", "hypertension"]
    diseases = decision.get("diseases", {})
    all_passed &= check(isinstance(diseases, dict), "Disease decision structure exists")

    for disease in decision_keys:
        result = diseases.get(disease)
        all_passed &= check(isinstance(result, dict), f"{disease} risk assessment exists")
        if isinstance(result, dict):
            risk_level = result.get("risk_level")
            risk_score = result.get("risk_score")
            all_passed &= check(
                risk_level in ["LOW", "MODERATE", "HIGH"],
                f"{disease} risk level is valid"
            )
            try:
                valid_score = risk_score is not None and 0 <= float(risk_score) <= 100
            except (TypeError, ValueError):
                valid_score = False
            all_passed &= check(valid_score, f"{disease} risk score is valid")

    all_passed &= check(
        decision.get("overall_risk") in ["LOW", "MODERATE", "HIGH"],
        "Overall risk is valid"
    )

    high_count = decision.get("high_risk_count")
    moderate_count = decision.get("moderate_risk_count")
    all_passed &= check(
        isinstance(high_count, int) and 0 <= high_count <= 3,
        "High-risk count is valid"
    )
    all_passed &= check(
        isinstance(moderate_count, int) and 0 <= moderate_count <= 3,
        "Moderate-risk count is valid"
    )
    all_passed &= check(
        decision.get("priority_disease") in decision_keys,
        "Priority disease is valid"
    )

    recommendations = assessment.get("recommendations")
    all_passed &= check(isinstance(recommendations, dict), "Recommendation result exists")

    disease_recommendations = recommendations.get("disease_recommendations", {})
    all_passed &= check(
        isinstance(disease_recommendations, dict),
        "Disease recommendations structure exists"
    )

    for disease in decision_keys:
        rec = disease_recommendations.get(disease)
        all_passed &= check(isinstance(rec, dict), f"{disease} recommendation exists")
        if isinstance(rec, dict):
            risk_level = rec.get("risk_level")
            rec_list = rec.get("recommendations", [])
            all_passed &= check(
                risk_level in ["LOW", "MODERATE", "HIGH"],
                f"{disease} recommendation risk level is valid"
            )
            all_passed &= check(
                isinstance(rec_list, list) and len(rec_list) > 0,
                f"{disease} has recommendations"
            )

    overall_recommendations = recommendations.get("overall_recommendations", [])
    all_passed &= check(
        isinstance(overall_recommendations, list),
        "Overall recommendations exist"
    )
    all_passed &= check(
        isinstance(overall_recommendations, list) and len(overall_recommendations) > 0,
        "Overall recommendations are not empty"
    )

    disclaimer = recommendations.get("disclaimer")
    all_passed &= check(isinstance(disclaimer, str), "Health disclaimer exists")
    all_passed &= check(
        isinstance(disclaimer, str) and len(disclaimer.strip()) > 0,
        "Health disclaimer is not empty"
    )

    all_passed &= check("prediction" in assessment, "Application response contains prediction")
    all_passed &= check("decision" in assessment, "Application response contains decision")
    all_passed &= check("recommendations" in assessment, "Application response contains recommendations")
    all_passed &= check("message" in assessment, "Application response contains safety message")

    mapping = {"cardio": "cardiovascular", "diabetes": "diabetes", "hypertension": "hypertension"}
    for prediction_key, decision_key in mapping.items():
        probability = prediction.get(prediction_key, {}).get("probability")
        risk_score = diseases.get(decision_key, {}).get("risk_score")
        try:
            consistent = abs(float(probability) * 100 - float(risk_score)) < 1e-6
        except (TypeError, ValueError):
            consistent = False
        all_passed &= check(
            consistent,
            f"{decision_key} probability/risk score consistency"
        )

    print()
    print("=" * 60)
    if all_passed:
        print("FINAL SYSTEM STATUS: PASS")
        print("The complete healthcare AI pipeline is working correctly.")
    else:
        print("FINAL SYSTEM STATUS: FAIL")
        print("One or more system checks failed.")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
