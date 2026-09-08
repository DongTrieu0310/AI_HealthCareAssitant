"""
Recommendation Engine
---------------------

This module converts AI risk assessment results
into interpretable healthcare recommendations.

The Recommendation Engine does NOT:
- train machine learning models
- make medical diagnoses
- prescribe medication
- replace healthcare professionals

Its responsibility is to:
1. Receive the output from the Decision Engine.
2. Generate recommendations for each disease.
3. Generate recommendations based on overall risk.
4. Provide general preventive guidance.
5. Clearly communicate that recommendations
   are supportive information, not medical advice.
"""


# ============================================================
# 1. GENERAL RECOMMENDATIONS
# ============================================================

GENERAL_RECOMMENDATIONS = [
    "Maintain a balanced and nutritious diet.",
    "Maintain regular physical activity appropriate for your condition.",
    "Maintain a healthy body weight.",
    "Monitor important health indicators regularly.",
    "Discuss concerning risk factors with a qualified healthcare professional."
]


# ============================================================
# 2. CARDIOVASCULAR RECOMMENDATIONS
# ============================================================

CARDIOVASCULAR_RECOMMENDATIONS = {

    "LOW": [
        "Continue maintaining a heart-healthy lifestyle.",
        "Maintain regular physical activity.",
        "Monitor cardiovascular risk factors periodically."
    ],

    "MODERATE": [
        "Pay closer attention to cardiovascular risk factors.",
        "Maintain a heart-healthy diet.",
        "Maintain regular physical activity.",
        "Monitor blood pressure and other cardiovascular indicators.",
        "Consider discussing cardiovascular risk factors with a healthcare professional."
    ],

    "HIGH": [
        "Seek professional medical evaluation for cardiovascular risk factors.",
        "Monitor blood pressure and other cardiovascular indicators regularly.",
        "Avoid smoking and reduce exposure to tobacco smoke.",
        "Maintain a heart-healthy diet and appropriate physical activity.",
        "Do not use the AI result as a medical diagnosis."
    ]
}


# ============================================================
# 3. DIABETES RECOMMENDATIONS
# ============================================================

DIABETES_RECOMMENDATIONS = {

    "LOW": [
        "Maintain a balanced diet with appropriate carbohydrate intake.",
        "Maintain regular physical activity.",
        "Maintain a healthy body weight.",
        "Continue monitoring diabetes-related risk factors."
    ],

    "MODERATE": [
        "Pay closer attention to diabetes-related risk factors.",
        "Maintain a balanced diet and appropriate portion sizes.",
        "Maintain regular physical activity.",
        "Monitor blood glucose when appropriate.",
        "Consider discussing diabetes risk factors with a healthcare professional."
    ],

    "HIGH": [
        "Seek professional medical evaluation for diabetes-related risk factors.",
        "Consider appropriate blood glucose testing under professional guidance.",
        "Maintain a balanced diet and regular physical activity.",
        "Monitor body weight and other relevant health indicators.",
        "Do not use the AI result as a medical diagnosis."
    ]
}


# ============================================================
# 4. HYPERTENSION RECOMMENDATIONS
# ============================================================

HYPERTENSION_RECOMMENDATIONS = {

    "LOW": [
        "Maintain a healthy lifestyle to support normal blood pressure.",
        "Maintain regular physical activity.",
        "Limit excessive dietary sodium.",
        "Monitor blood pressure periodically."
    ],

    "MODERATE": [
        "Monitor blood pressure more regularly.",
        "Maintain a balanced diet with appropriate sodium intake.",
        "Maintain regular physical activity.",
        "Maintain a healthy body weight.",
        "Consider discussing blood pressure risk factors with a healthcare professional."
    ],

    "HIGH": [
        "Seek professional medical evaluation for blood pressure risk factors.",
        "Monitor blood pressure regularly.",
        "Pay close attention to dietary sodium intake.",
        "Maintain appropriate physical activity as advised by a healthcare professional.",
        "Do not use the AI result as a medical diagnosis."
    ]
}


# ============================================================
# 5. DISEASE RECOMMENDATION FUNCTION
# ============================================================

def get_disease_recommendations(
    disease,
    risk_level
):
    """
    Generate recommendations for a specific disease.

    Parameters
    ----------
    disease : str
        Disease identifier.

    risk_level : str
        LOW, MODERATE or HIGH.

    Returns
    -------
    list
        List of recommendations.
    """

    if not isinstance(disease, str):
        raise TypeError(
            "Disease must be a string."
        )

    if not isinstance(risk_level, str):
        raise TypeError(
            "Risk level must be a string."
        )

    risk_level = risk_level.upper()

    valid_risk_levels = {
        "LOW",
        "MODERATE",
        "HIGH"
    }

    if risk_level not in valid_risk_levels:
        raise ValueError(
            "Risk level must be LOW, MODERATE or HIGH."
        )

    disease = disease.lower()

    if disease in {
        "cardiovascular",
        "cardio",
        "cardiovascular disease"
    }:

        return CARDIOVASCULAR_RECOMMENDATIONS[
            risk_level
        ].copy()

    if disease == "diabetes":

        return DIABETES_RECOMMENDATIONS[
            risk_level
        ].copy()

    if disease == "hypertension":

        return HYPERTENSION_RECOMMENDATIONS[
            risk_level
        ].copy()

    raise ValueError(
        f"Unsupported disease: {disease}"
    )


# ============================================================
# 6. OVERALL RISK RECOMMENDATIONS
# ============================================================

def get_overall_recommendations(
    overall_risk
):
    """
    Generate recommendations based on
    the overall healthcare risk level.
    """

    if not isinstance(overall_risk, str):
        raise TypeError(
            "Overall risk must be a string."
        )

    overall_risk = overall_risk.upper()

    if overall_risk == "LOW":

        return [
            "Continue maintaining a healthy lifestyle.",
            "Continue regular physical activity.",
            "Monitor important health indicators periodically."
        ]

    if overall_risk == "MODERATE":

        return [
            "Pay closer attention to your overall health risk factors.",
            "Maintain a balanced diet and regular physical activity.",
            "Monitor relevant health indicators regularly.",
            "Consider discussing important risk factors with a healthcare professional."
        ]

    if overall_risk == "HIGH":

        return [
            "Consider seeking professional healthcare evaluation.",
            "Monitor relevant health indicators closely.",
            "Follow appropriate healthcare guidance.",
            "Do not rely on the AI result as a standalone medical decision."
        ]

    raise ValueError(
        "Overall risk must be LOW, MODERATE or HIGH."
    )


# ============================================================
# 7. COMPLETE RECOMMENDATION
# ============================================================

def generate_recommendations(
    decision
):
    """
    Generate a complete recommendation package
    from the Decision Engine output.

    Parameters
    ----------
    decision : dict
        Output generated by the Risk Decision Engine.

    Returns
    -------
    dict
        Structured recommendation results.
    """

    if not isinstance(decision, dict):

        raise TypeError(
            "Decision must be a dictionary."
        )

    if "diseases" not in decision:

        raise KeyError(
            "Decision is missing 'diseases'."
        )

    if "overall_risk" not in decision:

        raise KeyError(
            "Decision is missing 'overall_risk'."
        )

    diseases = decision["diseases"]

    if not isinstance(diseases, dict):

        raise TypeError(
            "Decision 'diseases' must be a dictionary."
        )

    recommendations = {}

    # --------------------------------------------------------
    # Disease-specific recommendations
    # --------------------------------------------------------

    for disease, result in diseases.items():

        if not isinstance(result, dict):

            raise TypeError(
                f"Invalid result for disease: {disease}"
            )

        if "risk_level" not in result:

            raise KeyError(
                f"Missing risk_level for {disease}"
            )

        risk_level = result["risk_level"]

        recommendations[disease] = {
            "risk_level": risk_level,
            "recommendations":
                get_disease_recommendations(
                    disease,
                    risk_level
                )
        }

    # --------------------------------------------------------
    # Overall recommendations
    # --------------------------------------------------------

    overall_risk = decision[
        "overall_risk"
    ]

    overall_recommendations = (
        get_overall_recommendations(
            overall_risk
        )
    )

    return {
        "overall_risk": overall_risk,
        "disease_recommendations": recommendations,
        "overall_recommendations":
            overall_recommendations,
        "disclaimer": (
            "These recommendations are general "
            "health-support information. They are "
            "not medical diagnoses or prescriptions. "
            "Final medical decisions should be made "
            "by qualified healthcare professionals."
        )
    }


# ============================================================
# 8. TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("AI HEALTHCARE ASSISTANT - RECOMMENDATION ENGINE")
    print("=" * 70)

    # --------------------------------------------------------
    # Example Decision Engine output
    # --------------------------------------------------------

    test_decision = {

        "diseases": {

            "cardiovascular": {
                "disease":
                    "Cardiovascular Disease",
                "probability": 0.6963,
                "risk_score": 69.63,
                "risk_level": "MODERATE"
            },

            "diabetes": {
                "disease":
                    "Diabetes",
                "probability": 0.6800,
                "risk_score": 68.00,
                "risk_level": "MODERATE"
            },

            "hypertension": {
                "disease":
                    "Hypertension",
                "probability": 0.6609,
                "risk_score": 66.09,
                "risk_level": "MODERATE"
            }
        },

        "overall_risk": "MODERATE",

        "high_risk_count": 0,

        "moderate_risk_count": 3,

        "priority_disease":
            "cardiovascular"
    }

    # --------------------------------------------------------
    # Generate recommendations
    # --------------------------------------------------------

    recommendation_result = (
        generate_recommendations(
            test_decision
        )
    )

    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("DISEASE RECOMMENDATIONS")
    print("-" * 70)

    for (
        disease,
        result
    ) in recommendation_result[
        "disease_recommendations"
    ].items():

        print(
            f"\n{disease.upper()}"
        )

        print(
            f"Risk level: "
            f"{result['risk_level']}"
        )

        for recommendation in result[
            "recommendations"
        ]:

            print(
                f"  - {recommendation}"
            )

    print("\n" + "-" * 70)
    print("OVERALL RECOMMENDATIONS")
    print("-" * 70)

    print(
        f"Overall risk: "
        f"{recommendation_result['overall_risk']}"
    )

    for recommendation in (
        recommendation_result[
            "overall_recommendations"
        ]
    ):

        print(
            f"  - {recommendation}"
        )

    print("\n" + "-" * 70)
    print("DISCLAIMER")
    print("-" * 70)

    print(
        recommendation_result[
            "disclaimer"
        ]
    )

    print("\n" + "=" * 70)
    print("RECOMMENDATION ENGINE TEST COMPLETED")
    print("=" * 70)