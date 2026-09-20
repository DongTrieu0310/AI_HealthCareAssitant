"""
Risk Decision Engine
--------------------

This module converts ML model probabilities into
interpretable healthcare risk levels.

The engine does NOT train machine learning models.

Its responsibility is to:
1. Receive model probabilities.
2. Classify risk levels.
3. Aggregate multiple disease risks.
4. Generate an overall risk assessment.
"""


# ============================================================
# 1. RISK THRESHOLDS
# ============================================================

LOW_RISK_THRESHOLD = 0.30
HIGH_RISK_THRESHOLD = 0.70


# ============================================================
# 2. CLASSIFY SINGLE DISEASE RISK
# ============================================================

def classify_risk(probability):
    """
    Convert a probability into a risk category.

    Parameters
    ----------
    probability : float
        Probability between 0 and 1.

    Returns
    -------
    str
        LOW, MODERATE or HIGH
    """

    if not 0 <= probability <= 1:
        raise ValueError(
            "Probability must be between 0 and 1."
        )

    if probability < LOW_RISK_THRESHOLD:
        return "LOW"

    if probability < HIGH_RISK_THRESHOLD:
        return "MODERATE"

    return "HIGH"


# ============================================================
# 3. RISK SCORE
# ============================================================

def risk_score(probability):
    """
    Convert probability into a percentage score.

    Example:
        0.82 -> 82.0
    """

    if not 0 <= probability <= 1:
        raise ValueError(
            "Probability must be between 0 and 1."
        )

    return probability * 100


# ============================================================
# 4. SINGLE DISEASE ASSESSMENT
# ============================================================

def assess_disease(disease_name, probability):
    """
    Create a structured risk assessment for one disease.
    """

    return {
        "disease": disease_name,
        "probability": probability,
        "risk_score": risk_score(probability),
        "risk_level": classify_risk(probability)
    }


# ============================================================
# 5. MULTI-DISEASE ASSESSMENT
# ============================================================

def assess_all_diseases(
    cardio_probability,
    diabetes_probability,
    hypertension_probability
):
    """
    Assess risks for all supported diseases.

    Returns
    -------
    dict
        Structured multi-disease assessment.
    """

    results = {
        "cardiovascular": assess_disease(
            "Cardiovascular Disease",
            cardio_probability
        ),

        "diabetes": assess_disease(
            "Diabetes",
            diabetes_probability
        ),

        "hypertension": assess_disease(
            "Hypertension",
            hypertension_probability
        )
    }

    return results


# ============================================================
# 6. OVERALL RISK
# ============================================================

def calculate_overall_risk(results):
    """
    Calculate an overall healthcare risk level.

    Decision logic:

    - HIGH if any disease has HIGH risk.
    - MODERATE if no HIGH risk but at least one MODERATE risk.
    - LOW otherwise.
    """

    risk_levels = [
        result["risk_level"]
        for result in results.values()
    ]

    if "HIGH" in risk_levels:
        return "HIGH"

    if "MODERATE" in risk_levels:
        return "MODERATE"

    return "LOW"


# ============================================================
# 7. COUNT HIGH-RISK CONDITIONS
# ============================================================

def count_high_risk(results):
    """
    Count the number of diseases classified as HIGH risk.
    """

    return sum(
        1
        for result in results.values()
        if result["risk_level"] == "HIGH"
    )


# ============================================================
# 8. COUNT MODERATE-RISK CONDITIONS
# ============================================================

def count_moderate_risk(results):
    """
    Count the number of diseases classified as MODERATE risk.
    """

    return sum(
        1
        for result in results.values()
        if result["risk_level"] == "MODERATE"
    )


# ============================================================
# 9. PRIORITY DISEASE
# ============================================================

def determine_priority_disease(results):
    """
    Determine which disease has the highest predicted probability.
    """

    if not results:
        raise ValueError("Results cannot be empty.")

    priority = max(
        results.items(),
        key=lambda item: item[1]["probability"]
    )

    return priority[0]


# ============================================================
# 10. COMPLETE DECISION
# ============================================================

def make_decision(
    cardio_probability,
    diabetes_probability,
    hypertension_probability
):
    """
    Generate a complete healthcare risk decision.

    This function combines:
        - Individual disease risks
        - Overall risk
        - High-risk count
        - Moderate-risk count
        - Priority disease
    """

    results = assess_all_diseases(
        cardio_probability,
        diabetes_probability,
        hypertension_probability
    )

    overall_risk = calculate_overall_risk(results)

    high_risk_count = count_high_risk(results)

    moderate_risk_count = count_moderate_risk(results)

    priority_disease = determine_priority_disease(results)

    return {
        "diseases": results,
        "overall_risk": overall_risk,
        "high_risk_count": high_risk_count,
        "moderate_risk_count": moderate_risk_count,
        "priority_disease": priority_disease
    }


# ============================================================
# 11. TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("AI HEALTHCARE ASSISTANT - DECISION ENGINE")
    print("=" * 60)

    decision = make_decision(
        cardio_probability=0.82,
        diabetes_probability=0.35,
        hypertension_probability=0.74
    )

    print("\n--- DISEASE RISKS ---")

    for disease, result in decision["diseases"].items():

        print(f"\n{result['disease']}")

        print(
            f"Probability: "
            f"{result['probability']:.2%}"
        )

        print(
            f"Risk score: "
            f"{result['risk_score']:.2f}%"
        )

        print(
            f"Risk level: "
            f"{result['risk_level']}"
        )

    print("\n--- OVERALL DECISION ---")

    print(
        "Overall risk:",
        decision["overall_risk"]
    )

    print(
        "High-risk conditions:",
        decision["high_risk_count"]
    )

    print(
        "Moderate-risk conditions:",
        decision["moderate_risk_count"]
    )

    print(
        "Priority disease:",
        decision["priority_disease"]
    )

    print("\n" + "=" * 60)
    print("DECISION ENGINE TEST COMPLETED")
    print("=" * 60)