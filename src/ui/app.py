"""
AI Healthcare Assistant
User Interface - Streamlit

This UI connects to the existing Application Layer.

Pipeline:

User Input
    ↓
Healthcare Assistant
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
# 1. IMPORTS
# ============================================================

import sys
from pathlib import Path

import streamlit as st


# ============================================================
# 2. PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
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
# 4. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Healthcare Assistant",
    page_icon="🩺",
    layout="wide"
)


# ============================================================
# 5. HEADER
# ============================================================

st.title("🩺 AI Healthcare Assistant")

st.markdown(
    """
    ### AI-powered health risk assessment

    This system estimates the risk of:

    - Cardiovascular Disease
    - Diabetes
    - Hypertension

    **Important:** These results are AI-generated risk estimates
    and are not medical diagnoses.
    """
)

st.divider()


# ============================================================
# 6. PATIENT INFORMATION
# ============================================================

st.header("👤 Patient Information")

col1, col2, col3 = st.columns(3)

with col1:

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=55
    )

    gender = st.selectbox(
        "Gender",
        options=[1, 2],
        format_func=lambda x:
            "Female" if x == 1 else "Male"
    )

    height = st.number_input(
        "Height (cm)",
        min_value=50,
        max_value=250,
        value=170
    )

    weight = st.number_input(
        "Weight (kg)",
        min_value=20.0,
        max_value=300.0,
        value=75.0
    )


with col2:

    ap_hi = st.number_input(
        "Systolic Blood Pressure",
        min_value=50,
        max_value=250,
        value=140
    )

    ap_lo = st.number_input(
        "Diastolic Blood Pressure",
        min_value=30,
        max_value=150,
        value=90
    )

    cholesterol = st.selectbox(
        "Cholesterol",
        options=[1, 2, 3],
        format_func=lambda x:
            {
                1: "Normal",
                2: "Above Normal",
                3: "Well Above Normal"
            }[x]
    )

    gluc = st.selectbox(
        "Glucose Level",
        options=[1, 2, 3],
        format_func=lambda x:
            {
                1: "Normal",
                2: "Above Normal",
                3: "Well Above Normal"
            }[x]
    )


with col3:

    smoke = st.selectbox(
        "Smoking",
        options=[0, 1],
        format_func=lambda x:
            "No" if x == 0 else "Yes"
    )

    alco = st.selectbox(
        "Alcohol Consumption",
        options=[0, 1],
        format_func=lambda x:
            "No" if x == 0 else "Yes"
    )

    active = st.selectbox(
        "Physical Activity",
        options=[0, 1],
        format_func=lambda x:
            "No" if x == 0 else "Yes"
    )


# ============================================================
# 7. DIABETES INFORMATION
# ============================================================

st.divider()

st.header("🩸 Diabetes Information")

col1, col2, col3 = st.columns(3)

with col1:

    pregnancies = st.number_input(
        "Pregnancies",
        min_value=0,
        max_value=20,
        value=2
    )

    glucose = st.number_input(
        "Glucose",
        min_value=0.0,
        max_value=300.0,
        value=148.0
    )

    blood_pressure = st.number_input(
        "Blood Pressure",
        min_value=0.0,
        max_value=200.0,
        value=90.0
    )


with col2:

    skin_thickness = st.number_input(
        "Skin Thickness",
        min_value=0.0,
        max_value=100.0,
        value=35.0
    )

    insulin = st.number_input(
        "Insulin",
        min_value=0.0,
        max_value=1000.0,
        value=120.0
    )


with col3:

    bmi = st.number_input(
        "BMI",
        min_value=0.0,
        max_value=80.0,
        value=32.0
    )

    diabetes_pedigree = st.number_input(
        "Diabetes Pedigree Function",
        min_value=0.0,
        max_value=3.0,
        value=0.6
    )

    diabetes_age = st.number_input(
        "Diabetes Age",
        min_value=1,
        max_value=120,
        value=55
    )


# ============================================================
# 8. HYPERTENSION INFORMATION
# ============================================================

st.divider()

st.header("❤️ Hypertension Information")

col1, col2, col3 = st.columns(3)

with col1:

    male = st.selectbox(
        "Sex",
        options=[0, 1],
        format_func=lambda x:
            "Female" if x == 0 else "Male"
    )

    current_smoker = st.selectbox(
        "Current Smoker",
        options=[0, 1],
        format_func=lambda x:
            "No" if x == 0 else "Yes"
    )

    cigs_per_day = st.number_input(
        "Cigarettes per Day",
        min_value=0.0,
        max_value=100.0,
        value=0.0
    )

    bp_meds = st.selectbox(
        "Blood Pressure Medication",
        options=[0, 1],
        format_func=lambda x:
            "No" if x == 0 else "Yes"
    )


with col2:

    hypertension_diabetes = st.selectbox(
        "Diabetes History",
        options=[0, 1],
        format_func=lambda x:
            "No" if x == 0 else "Yes"
    )

    total_chol = st.number_input(
        "Total Cholesterol",
        min_value=50.0,
        max_value=500.0,
        value=230.0
    )

    systolic_bp = st.number_input(
        "Systolic BP",
        min_value=50.0,
        max_value=300.0,
        value=140.0
    )

    diastolic_bp = st.number_input(
        "Diastolic BP",
        min_value=30.0,
        max_value=200.0,
        value=90.0
    )


with col3:

    hypertension_bmi = st.number_input(
        "BMI",
        min_value=10.0,
        max_value=80.0,
        value=32.0
    )

    heart_rate = st.number_input(
        "Heart Rate",
        min_value=30.0,
        max_value=220.0,
        value=75.0
    )

    hypertension_glucose = st.number_input(
        "Glucose",
        min_value=30.0,
        max_value=500.0,
        value=110.0
    )


# ============================================================
# 9. ASSESSMENT BUTTON
# ============================================================

st.divider()

st.header("🔍 Health Risk Assessment")

assess_button = st.button(
    "🩺 ASSESS HEALTH RISK",
    type="primary",
    use_container_width=True
)

# ============================================================
# 9.5 INPUT VALIDATION
# ============================================================

def validate_patient_input(patient):
    """
    Validate patient input before sending it
    to the Healthcare Assistant.
    """

    errors = []

    # --------------------------------------------------------
    # Basic values
    # --------------------------------------------------------

    if not 1 <= patient["age"] <= 120:
        errors.append(
            "Age must be between 1 and 120."
        )

    if not 50 <= patient["height"] <= 250:
        errors.append(
            "Height must be between 50 and 250 cm."
        )

    if not 20 <= patient["weight"] <= 300:
        errors.append(
            "Weight must be between 20 and 300 kg."
        )

    # --------------------------------------------------------
    # Cardiovascular blood pressure
    # --------------------------------------------------------

    if not 50 <= patient["ap_hi"] <= 250:
        errors.append(
            "Systolic blood pressure must be between "
            "50 and 250 mmHg."
        )

    if not 30 <= patient["ap_lo"] <= 150:
        errors.append(
            "Diastolic blood pressure must be between "
            "30 and 150 mmHg."
        )

    if patient["ap_hi"] <= patient["ap_lo"]:
        errors.append(
            "Systolic blood pressure must be higher "
            "than diastolic blood pressure."
        )

    # --------------------------------------------------------
    # Diabetes
    # --------------------------------------------------------

    if not 0 <= patient["Pregnancies"] <= 20:
        errors.append(
            "Pregnancies must be between 0 and 20."
        )

    if not 0 <= patient["Glucose"] <= 300:
        errors.append(
            "Diabetes glucose must be between 0 and 300."
        )

    if not 0 <= patient["BMI"] <= 80:
        errors.append(
            "BMI must be between 0 and 80."
        )

    if not 0 <= patient[
        "DiabetesPedigreeFunction"
    ] <= 3:
        errors.append(
            "Diabetes Pedigree Function must be "
            "between 0 and 3."
        )

    # --------------------------------------------------------
    # Hypertension
    # --------------------------------------------------------

    if not 0 <= patient["cigsPerDay"] <= 100:
        errors.append(
            "Cigarettes per day must be between 0 and 100."
        )

    if not 50 <= patient["sysBP"] <= 300:
        errors.append(
            "Hypertension systolic BP must be "
            "between 50 and 300 mmHg."
        )

    if not 30 <= patient["diaBP"] <= 200:
        errors.append(
            "Hypertension diastolic BP must be "
            "between 30 and 200 mmHg."
        )

    if patient["sysBP"] <= patient["diaBP"]:
        errors.append(
            "Hypertension systolic BP must be higher "
            "than diastolic BP."
        )

    if not 30 <= patient["heartRate"] <= 220:
        errors.append(
            "Heart rate must be between 30 and 220 bpm."
        )

    if not 30 <= patient["glucose"] <= 500:
        errors.append(
            "Hypertension glucose must be "
            "between 30 and 500."
        )

    return errors
# ============================================================
# 10. RUN ASSESSMENT
# ============================================================

if assess_button:

    try:

        # ----------------------------------------------------
        # Build patient dictionary
        # ----------------------------------------------------

        patient = {

            # Cardiovascular
            "age": age,
            "gender": gender,
            "height": height,
            "weight": weight,
            "ap_hi": ap_hi,
            "ap_lo": ap_lo,
            "cholesterol": cholesterol,
            "gluc": gluc,
            "smoke": smoke,
            "alco": alco,
            "active": active,

            # Diabetes
            "Pregnancies": pregnancies,
            "Glucose": glucose,
            "BloodPressure": blood_pressure,
            "SkinThickness": skin_thickness,
            "Insulin": insulin,
            "BMI": bmi,
            "DiabetesPedigreeFunction":
                diabetes_pedigree,
            "Age": diabetes_age,

            # Hypertension
            "male": male,
            "currentSmoker": current_smoker,
            "cigsPerDay": cigs_per_day,
            "BPMeds": bp_meds,
            "diabetes": hypertension_diabetes,
            "totChol": total_chol,
            "sysBP": systolic_bp,
            "diaBP": diastolic_bp,
            "heartRate": heart_rate,
            "glucose": hypertension_glucose
        }
        # ----------------------------------------------------
        # Validate patient input
        # ----------------------------------------------------

        validation_errors = validate_patient_input(
            patient
        )

        if validation_errors:

            st.error(
                "Please correct the following input errors:"
            )

            for error in validation_errors:

                st.write(
                    f"• {error}"
                )

            st.stop()
        # ----------------------------------------------------
        # Initialize Healthcare Assistant
        # ----------------------------------------------------

        assistant = HealthcareAssistant()

        # ----------------------------------------------------
        # Run complete application pipeline
        # ----------------------------------------------------

        assessment = assistant.assess(
            patient
        )

        # Store result
        st.session_state["assessment"] = assessment

        st.success(
            "Health assessment completed successfully."
        )

    except Exception as e:

        st.error(
            f"Assessment failed: {e}"
        )


# ============================================================
# 11. DISPLAY RESULTS
# ============================================================

if "assessment" in st.session_state:

    assessment = st.session_state[
        "assessment"
    ]

    prediction = assessment.get(
        "prediction",
        {}
    )

    decision = assessment.get(
        "decision",
        {}
    )

    recommendations = assessment.get(
        "recommendations",
        {}
    )

    # ========================================================
    # RESULTS
    # ========================================================

    st.divider()

    st.header("📊 Assessment Results")

    # --------------------------------------------------------
    # Disease probabilities
    # --------------------------------------------------------

    st.subheader(
        "Disease Risk Probabilities"
    )

    probability_columns = st.columns(3)

    disease_mapping = {
        "cardio":
            "Cardiovascular Disease",
        "diabetes":
            "Diabetes",
        "hypertension":
            "Hypertension"
    }

    for column, (
        disease_key,
        disease_name
    ) in zip(
        probability_columns,
        disease_mapping.items()
    ):

        disease_result = prediction.get(
            disease_key,
            {}
        )

        probability = disease_result.get(
            "probability",
            0
        )

        with column:

            st.metric(
                disease_name,
                f"{float(probability):.2%}"
            )

            st.progress(
                min(
                    max(
                        float(probability),
                        0.0
                    ),
                    1.0
                )
            )

    # --------------------------------------------------------
    # Risk levels
    # --------------------------------------------------------

    st.subheader(
        "Risk Levels"
    )

    diseases = decision.get(
        "diseases",
        {}
    )

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
            disease_key,
            {}
        )

        risk_level = disease_result.get(
            "risk_level",
            "N/A"
        )

        risk_score = disease_result.get(
            "risk_score"
        )

        if risk_score is not None:

            st.write(
                f"**{disease_name}:** "
                f"{risk_level} "
                f"({float(risk_score):.2f}%)"
            )

        else:

            st.write(
                f"**{disease_name}:** "
                f"{risk_level}"
            )

    # --------------------------------------------------------
    # Overall risk
    # --------------------------------------------------------

    st.subheader(
        "Overall Risk Assessment"
    )

    overall_risk = decision.get(
        "overall_risk",
        "N/A"
    )

    priority_disease = decision.get(
        "priority_disease",
        "N/A"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Overall Risk",
            overall_risk
        )

    with col2:

        st.metric(
            "Priority Disease",
            priority_disease
        )

    # --------------------------------------------------------
    # Risk counts
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        st.write(
            "**High-risk conditions:**",
            decision.get(
                "high_risk_count",
                0
            )
        )

    with col2:

        st.write(
            "**Moderate-risk conditions:**",
            decision.get(
                "moderate_risk_count",
                0
            )
        )

    # ========================================================
    # RECOMMENDATIONS
    # ========================================================

    st.divider()

    st.header(
        "💡 Health Recommendations"
    )

    disease_recommendations = (
        recommendations.get(
            "disease_recommendations",
            {}
        )
    )

    for disease_key, disease_name in (
        decision_mapping.items()
    ):

        disease_recommendation = (
            disease_recommendations.get(
                disease_key,
                {}
            )
        )

        if not isinstance(
            disease_recommendation,
            dict
        ):
            continue

        risk_level = (
            disease_recommendation.get(
                "risk_level",
                "N/A"
            )
        )

        recommendation_list = (
            disease_recommendation.get(
                "recommendations",
                []
            )
        )

        st.subheader(
            disease_name
        )

        st.write(
            f"Risk level: **{risk_level}**"
        )

        for recommendation in (
            recommendation_list
        ):

            st.write(
                f"• {recommendation}"
            )

    # --------------------------------------------------------
    # Overall recommendations
    # --------------------------------------------------------

    overall_recommendations = (
        recommendations.get(
            "overall_recommendations",
            []
        )
    )

    if overall_recommendations:

        st.subheader(
            "Overall Recommendations"
        )

        for recommendation in (
            overall_recommendations
        ):

            st.write(
                f"• {recommendation}"
            )

    # ========================================================
    # DISCLAIMER
    # ========================================================

    st.divider()

    st.warning(
        recommendations.get(
            "disclaimer",
            (
                "These results are AI-generated "
                "risk estimates and are not medical "
                "diagnoses."
            )
        )
    )


# ============================================================
# 12. FOOTER
# ============================================================

st.divider()

st.caption(
    "AI Healthcare Assistant | "
    "Prediction + Decision + Recommendation Pipeline"
)