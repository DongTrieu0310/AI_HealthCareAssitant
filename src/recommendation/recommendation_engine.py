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
    "Duy trì chế độ ăn cân đối và đủ dinh dưỡng.",
    "Vận động thể chất đều đặn, phù hợp với thể trạng.",
    "Giữ cân nặng ở mức hợp lý.",
    "Theo dõi các chỉ số sức khỏe quan trọng thường xuyên.",
    "Trao đổi với nhân viên y tế về những yếu tố nguy cơ đáng lo ngại."
]


# ============================================================
# 2. CARDIOVASCULAR RECOMMENDATIONS
# ============================================================

CARDIOVASCULAR_RECOMMENDATIONS = {

    "LOW": [
        "Tiếp tục duy trì lối sống tốt cho tim mạch.",
        "Vận động thể chất đều đặn.",
        "Theo dõi định kỳ các yếu tố nguy cơ tim mạch."
    ],

    "MODERATE": [
        "Chú ý hơn tới các yếu tố nguy cơ tim mạch.",
        "Duy trì chế độ ăn tốt cho tim mạch.",
        "Vận động thể chất đều đặn.",
        "Theo dõi huyết áp và các chỉ số tim mạch khác.",
        "Cân nhắc trao đổi với nhân viên y tế về nguy cơ tim mạch."
    ],

    "HIGH": [
        "Đi khám để được đánh giá chuyên môn về nguy cơ tim mạch.",
        "Theo dõi huyết áp và các chỉ số tim mạch thường xuyên.",
        "Không hút thuốc và hạn chế tiếp xúc với khói thuốc.",
        "Duy trì chế độ ăn tốt cho tim mạch và vận động phù hợp.",
        "Không xem kết quả AI như một chẩn đoán y khoa."
    ]
}


# ============================================================
# 3. DIABETES RECOMMENDATIONS
# ============================================================

DIABETES_RECOMMENDATIONS = {

    "LOW": [
        "Duy trì chế độ ăn cân đối, lượng tinh bột hợp lý.",
        "Vận động thể chất đều đặn.",
        "Giữ cân nặng ở mức hợp lý.",
        "Tiếp tục theo dõi các yếu tố nguy cơ đái tháo đường."
    ],

    "MODERATE": [
        "Chú ý hơn tới các yếu tố nguy cơ đái tháo đường.",
        "Ăn cân đối và kiểm soát khẩu phần.",
        "Vận động thể chất đều đặn.",
        "Theo dõi đường huyết khi cần thiết.",
        "Cân nhắc trao đổi với nhân viên y tế về nguy cơ đái tháo đường."
    ],

    "HIGH": [
        "Đi khám để được đánh giá chuyên môn về nguy cơ đái tháo đường.",
        "Cân nhắc xét nghiệm đường huyết theo hướng dẫn của nhân viên y tế.",
        "Duy trì chế độ ăn cân đối và vận động đều đặn.",
        "Theo dõi cân nặng và các chỉ số sức khỏe liên quan.",
        "Không xem kết quả AI như một chẩn đoán y khoa."
    ]
}


# ============================================================
# 4. HYPERTENSION RECOMMENDATIONS
# ============================================================

HYPERTENSION_RECOMMENDATIONS = {

    "LOW": [
        "Duy trì lối sống lành mạnh để giữ huyết áp ổn định.",
        "Vận động thể chất đều đặn.",
        "Hạn chế ăn mặn.",
        "Đo huyết áp định kỳ."
    ],

    "MODERATE": [
        "Đo huyết áp thường xuyên hơn.",
        "Ăn cân đối, giảm lượng muối trong khẩu phần.",
        "Vận động thể chất đều đặn.",
        "Giữ cân nặng ở mức hợp lý.",
        "Cân nhắc trao đổi với nhân viên y tế về nguy cơ huyết áp."
    ],

    "HIGH": [
        "Đi khám để được đánh giá chuyên môn về huyết áp.",
        "Theo dõi huyết áp thường xuyên.",
        "Đặc biệt chú ý lượng muối trong khẩu phần ăn.",
        "Vận động ở mức phù hợp theo tư vấn của nhân viên y tế.",
        "Không xem kết quả AI như một chẩn đoán y khoa."
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
            "Tiếp tục duy trì lối sống lành mạnh.",
            "Tiếp tục vận động thể chất đều đặn.",
            "Theo dõi định kỳ các chỉ số sức khỏe quan trọng."
        ]

    if overall_risk == "MODERATE":

        return [
            "Chú ý hơn tới các yếu tố nguy cơ sức khỏe tổng thể.",
            "Ăn cân đối và vận động thể chất đều đặn.",
            "Theo dõi thường xuyên các chỉ số sức khỏe liên quan.",
            "Cân nhắc trao đổi với nhân viên y tế về các yếu tố nguy cơ quan trọng."
        ]

    if overall_risk == "HIGH":

        return [
            "Nên đi khám để được đánh giá chuyên môn.",
            "Theo dõi sát các chỉ số sức khỏe liên quan.",
            "Tuân thủ hướng dẫn của nhân viên y tế.",
            "Không dùng kết quả AI làm căn cứ y khoa duy nhất."
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
            "Đây là thông tin hỗ trợ sức khỏe mang tính "
            "chung, không phải chẩn đoán y khoa hay đơn "
            "thuốc. Quyết định y khoa cuối cùng phải do "
            "nhân viên y tế có chuyên môn đưa ra."
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