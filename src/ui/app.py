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

# Keep only the latest completed assessment.
if "assessment" not in st.session_state:
    st.session_state["assessment"] = None


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
# 9. CURRENT INPUT SIGNATURE
# ============================================================

# Streamlit reruns the whole script whenever an input changes.
# Keep the assessment tied to the exact input values that created it.
current_input_signature = (
    age, gender, height, weight,
    ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active,
    pregnancies, glucose, blood_pressure, skin_thickness, insulin,
    bmi, diabetes_pedigree, diabetes_age,
    male, current_smoker, cigs_per_day, bp_meds,
    hypertension_diabetes, total_chol, systolic_bp,
    diastolic_bp, hypertension_bmi, heart_rate, hypertension_glucose
)

# If the user changes any input after a previous assessment,
# remove the old result so old/new predictions can never be mixed.
if (
    "assessment_input_signature" in st.session_state
    and st.session_state["assessment_input_signature"]
    != current_input_signature
):
    st.session_state.pop("assessment", None)
    st.session_state.pop("ask_ai_auto_submit", None)


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

    # Clear the previous result before calculating a new assessment.
    # This prevents an old result from remaining visible if the new run fails.
    st.session_state["assessment"] = None

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

        # Store result together with the exact input signature.
        # This prevents a previous assessment from being displayed
        # after the patient inputs are changed.
        st.session_state["assessment"] = assessment
        st.session_state["assessment_input_signature"] = (
            current_input_signature
        )

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

if st.session_state.get("assessment") is not None:

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

    # Mapping used by the Decision Engine and Recommendation Engine.
    # Keep it aligned with the disease keys returned by the application layer.
    decision_mapping = {
        "cardiovascular":
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

    # The Decision Engine supplies the risk level. The percentage shown here
    # is taken from the SAME prediction object displayed above.
    display_mapping = {
        "cardio": ("cardiovascular", "Cardiovascular Disease"),
        "diabetes": ("diabetes", "Diabetes"),
        "hypertension": ("hypertension", "Hypertension")
    }

    for prediction_key, (decision_key, disease_name) in (
        display_mapping.items()
    ):

        prediction_result = prediction.get(
            prediction_key,
            {}
        )

        decision_result = diseases.get(
            decision_key,
            {}
        )

        probability = prediction_result.get(
            "probability"
        )

        risk_level = decision_result.get(
            "risk_level",
            "N/A"
        )

        if probability is None:
            st.write(
                f"**{disease_name}:** {risk_level}"
            )
        else:
            # Never use decision_result["risk_score"] for the displayed %.
            st.write(
                f"**{disease_name}:** "
                f"{risk_level} "
                f"({float(probability):.2%})"
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
    # TRUSTWORTHY AI DASHBOARD
    # ========================================================

    st.divider()
    st.header("🤖 Trustworthy AI")

    st.markdown(
        """
        This section presents the Trustworthy AI evaluation of the
        healthcare risk-assessment system. The evaluation covers
        fairness, robustness, explainability, bias, privacy,
        accountability, social impact, and mathematical foundations.
        """
    )

    ta_tabs = st.tabs([
        "📋 Overview",
        "🔍 Explainability & SHAP",
        "⚖️ Fairness & Bias",
        "🛡️ Robustness",
        "🔐 Privacy",
        "📋 Accountability",
        "🌍 Social Impact",
        "🧮 Math AI"
    ])

    # ------------------------------------------------------------
    # Overview
    # ------------------------------------------------------------
    with ta_tabs[0]:

        st.subheader("Trustworthy AI Evaluation Status")

        overview_items = [
            ("⚖️ Fairness", "Evaluated"),
            ("🛡️ Robustness", "Evaluated"),
            ("🔍 Explainability", "Evaluated"),
            ("📊 Bias Analysis", "Evaluated"),
            ("🔐 Privacy", "Evaluated"),
            ("📋 Accountability", "Evaluated"),
            ("🌍 Social Impact", "Evaluated"),
            ("🧮 Math AI", "Evaluated"),
        ]

        overview_cols = st.columns(4)

        for i, (name, status) in enumerate(overview_items):
            with overview_cols[i % 4]:
                st.metric(name, status)

        st.info(
            "Trustworthy AI evaluation is intended to support "
            "responsible use of the system. It does not guarantee "
            "that every prediction is correct or free from bias."
        )

        st.subheader("Known Evaluation Limitations")

        limitations = [
            "Performance can differ between demographic groups.",
            "Model performance can decrease when input noise is introduced.",
            "False-positive and false-negative predictions remain possible.",
            "Healthcare data contains sensitive information.",
            "The CARDIO dataset contains a direct identifier.",
            "Human oversight is required for medical decisions.",
            "Production monitoring and audit logging are not yet implemented."
        ]

        for item in limitations:
            st.write(f"• {item}")

    # ------------------------------------------------------------
    # Explainability & SHAP
    # ------------------------------------------------------------
    with ta_tabs[1]:

        st.subheader("🔍 Model Explainability")

        explainability = {
            "Cardiovascular Disease": ("ap_hi", 0.145809),
            "Diabetes": ("SkinThickness", 0.134053),
            "Hypertension": ("sysBP", 0.200924),
        }

        for disease, (feature, importance) in explainability.items():
            with st.expander(disease, expanded=True):
                st.write(
                    f"**Most influential feature:** `{feature}`"
                )
                st.write(
                    f"**Mean absolute SHAP value:** {importance:.6f}"
                )

        st.markdown("---")
        st.subheader("SHAP Visualizations")

        xai_dir = PROJECT_ROOT / "data" / "models" / "xai"

        image_groups = {
            "Cardiovascular Disease": ["cardio"],
            "Diabetes": ["diabetes"],
            "Hypertension": ["hypertension"],
        }

        if xai_dir.exists():
            for disease, keywords in image_groups.items():
                matching_images = []

                for image_path in sorted(xai_dir.glob("*.png")):
                    name_lower = image_path.name.lower()

                    if any(keyword in name_lower for keyword in keywords):
                        matching_images.append(image_path)

                if matching_images:
                    st.write(f"**{disease}**")

                    for image_path in matching_images:
                        st.image(
                            str(image_path),
                            caption=image_path.name,
                            use_container_width=True
                        )
        else:
            st.info(
                "SHAP visualization directory was not found. "
                "The numerical explainability results are still shown above."
            )

        st.caption(
            "SHAP values describe how input features contribute to "
            "model predictions. They should be interpreted as model "
            "explanations, not medical causation."
        )

    # ------------------------------------------------------------
    # Fairness & Bias
    # ------------------------------------------------------------
    with ta_tabs[2]:

        st.subheader("⚖️ Fairness")

        st.write(
            "Fairness analysis compares model performance across "
            "different demographic groups using Accuracy, Precision, "
            "Recall, and F1-score."
        )

        fairness_metrics = [
            "Accuracy",
            "Precision",
            "Recall",
            "F1-score"
        ]

        for metric in fairness_metrics:
            st.write(f"• {metric}: evaluated across groups")

        st.warning(
            "Observed performance differences between groups should "
            "be monitored. A performance disparity alone should not "
            "automatically be interpreted as proof of discriminatory bias."
        )

        st.subheader("📊 Bias Analysis")

        st.write(
            "Bias analysis is used to identify performance disparities "
            "and potential sources of systematic differences."
        )

        st.success("Bias analysis completed.")

        st.caption(
            "The current prototype focuses on evaluation and monitoring. "
            "It does not claim that the models are completely bias-free."
        )

    # ------------------------------------------------------------
    # Robustness
    # ------------------------------------------------------------
    with ta_tabs[3]:

        st.subheader("🛡️ Robustness Evaluation")

        st.write(
            "Robustness testing evaluates whether model performance "
            "remains stable when perturbations or noise are introduced "
            "into the input data."
        )

        st.code(
            """Original input
      ↓
Add perturbation / noise
      ↓
Run model
      ↓
Compare predictions
      ↓
Measure performance degradation""",
            language="text"
        )

        st.warning(
            "The evaluation shows that model performance can decrease "
            "when input noise is introduced. Therefore, predictions "
            "should be interpreted with appropriate caution."
        )

        st.success("Robustness analysis completed.")

    # ------------------------------------------------------------
    # Privacy
    # ------------------------------------------------------------
    with ta_tabs[4]:

        st.subheader("🔐 Privacy")

        st.write(
            "The privacy analysis checks the healthcare datasets for "
            "direct identifiers and highlights the presence of "
            "sensitive health information."
        )

        privacy_items = [
            ("Healthcare data", "Sensitive health information present"),
            ("CARDIO dataset", "Direct identifier detected in source data"),
            ("System purpose", "Risk estimation / decision support"),
        ]

        for item, result in privacy_items:
            st.write(f"**{item}:** {result}")

        st.warning(
            "The CARDIO source dataset contains a direct identifier. "
            "Such identifiers should not be exposed in a production "
            "deployment and should be removed or protected."
        )

        st.success("Privacy analysis completed.")

    # ------------------------------------------------------------
    # Accountability
    # ------------------------------------------------------------
    with ta_tabs[5]:

        st.subheader("📋 Accountability")

        accountability_items = [
            "The system identifies the three disease-specific models.",
            "Prediction results are passed through a Decision Engine.",
            "Risk levels are explicitly classified as LOW, MODERATE, or HIGH.",
            "Recommendations are separated from medical diagnosis.",
            "The system includes a medical-use disclaimer.",
            "Human healthcare professionals remain responsible for final decisions."
        ]

        for item in accountability_items:
            st.write(f"✅ {item}")

        st.info(
            "The AI system is designed as a decision-support prototype "
            "and should not replace qualified healthcare professionals."
        )

    # ------------------------------------------------------------
    # Social Impact
    # ------------------------------------------------------------
    with ta_tabs[6]:

        st.subheader("🌍 Social Impact")

        col_positive, col_negative = st.columns(2)

        with col_positive:
            st.markdown("### Potential Positive Impacts")

            positive = [
                "Support early risk screening.",
                "Help users monitor important health indicators.",
                "Provide accessible AI-assisted risk estimation.",
                "Support healthcare professionals with additional information."
            ]

            for item in positive:
                st.write(f"🟢 {item}")

        with col_negative:
            st.markdown("### Potential Negative Impacts")

            negative = [
                "False-positive predictions may cause unnecessary concern.",
                "False-negative predictions may create false reassurance.",
                "Users may rely too heavily on AI-generated results.",
                "Bias or performance differences may affect some groups."
            ]

            for item in negative:
                st.write(f"🔴 {item}")

        st.info(
            "Human oversight, explainability, fairness evaluation, "
            "and clear disclaimers are important mitigation measures."
        )

    # ------------------------------------------------------------
    # Math AI
    # ------------------------------------------------------------
    with ta_tabs[7]:

        st.subheader("🧮 Mathematical Foundations")

        math_items = [
            (
                "Calculus",
                "Supports understanding of change, optimization, "
                "and model-related mathematical concepts."
            ),
            (
                "Linear Algebra",
                "Provides the mathematical foundation for vectors, "
                "matrices, and feature representations."
            ),
            (
                "Advanced Linear Algebra",
                "Supports matrix-based analysis and transformations."
            ),
            (
                "Probability",
                "Provides the foundation for probabilistic risk estimates."
            ),
            (
                "Statistics",
                "Supports descriptive statistics, variability, "
                "correlation, and model evaluation."
            ),
        ]

        for topic, description in math_items:
            with st.expander(topic):
                st.write(description)

        st.success("Math AI evaluation completed.")

    # ========================================================
    # DEFENSE / Q&A
    # ========================================================
    st.divider()
    st.header("❓ Defense Questions & Answers")
    st.markdown(
        "Use these questions to explain the safety, limitations, and responsible-use design of the system during a project demonstration or defense."
    )

    defense_questions = [
        (
            "Vì sao không để AI/LLM tự chẩn đoán cho bệnh nhân?",
            "Hệ thống được thiết kế để đánh giá rủi ro và hỗ trợ quyết định, không thay thế chẩn đoán lâm sàng. Mô hình ML cung cấp risk estimate, Decision Engine phân loại mức rủi ro, còn quyết định y khoa cuối cùng thuộc về nhân viên y tế."
        ),
        (
            "Probability 80% có nghĩa là bệnh nhân chắc chắn mắc bệnh 80% không?",
            "Không. Đây là xác suất dự đoán của mô hình trên dữ liệu đầu vào, không phải xác suất chẩn đoán lâm sàng. Kết quả cần được hiểu như một risk estimate và phải kết hợp với đánh giá của chuyên gia y tế."
        ),
        (
            "Làm sao tránh người dùng tự điều trị dựa trên kết quả AI?",
            "Ứng dụng hiển thị disclaimer, chỉ đưa ra khuyến nghị sức khỏe chung, không kê đơn và không đưa ra chẩn đoán. Hệ thống nhấn mạnh human oversight và khuyến nghị người dùng trao đổi với healthcare professional."
        ),
        (
            "Dữ liệu có bias không?",
            "Hệ thống có thực hiện Bias Analysis và Fairness Evaluation để phát hiện sự khác biệt về hiệu năng giữa các nhóm. Nếu có disparity, điều đó cần được theo dõi và phân tích thêm; không nên tự động kết luận rằng disparity đồng nghĩa với discriminatory bias."
        ),
        (
            "Nếu hệ thống AI dự đoán sai thì ai chịu trách nhiệm?",
            "Hệ thống được định vị là decision-support prototype. AI không thay thế con người; quyết định y khoa cuối cùng phải do healthcare professional đưa ra. Đây là nguyên tắc accountability và human-in-the-loop của hệ thống."
        ),
        (
            "Người lớn tuổi khó sử dụng máy tính thì sao?",
            "Giao diện được thiết kế đơn giản, chia nhóm thông tin, dùng nhãn dễ hiểu và nút đánh giá rõ ràng. Đây là một phần của Social Impact và Accessibility. Trong tương lai có thể bổ sung voice interaction hoặc hỗ trợ người thân/nhân viên y tế nhập dữ liệu."
        ),
        (
            "App có kê đơn thuốc hoặc đưa ra phác đồ điều trị không?",
            "Không. Ứng dụng chỉ cung cấp risk estimation và các khuyến nghị sức khỏe chung như theo dõi chỉ số, duy trì chế độ ăn và vận động phù hợp. Không có chức năng kê đơn hay thay thế chỉ định của bác sĩ."
        ),
    ]

    for question, answer in defense_questions:
        with st.expander(f"❓ {question}"):
            st.write(answer)

    st.info(
        "Defense principle: AI supports screening and decision support; it does not replace clinical diagnosis or medical decision-making."
    )

    # ========================================================
    # ASK AI - HEALTH INFORMATION ASSISTANT
    # ========================================================
    st.divider()
    st.header("🤖 Ask AI")
    st.markdown(
        "Ask general health-information questions or ask the assistant "
        "to explain the current risk assessment. This assistant is for "
        "education and decision support only."
    )

    def ask_ai_local(question, assessment_data):
        """Controlled health-information assistant for the UI prototype.

        The function intentionally avoids diagnosis, prescriptions,
        medication changes, and emergency decision-making.
        """
        q = question.strip().lower()

        if not q:
            return "Vui lòng nhập câu hỏi."

        # Safety guardrails for high-risk requests.
        unsafe_terms = [
            "prescribe", "prescription", "kê đơn", "thuốc gì", "nên uống thuốc",
            "liều thuốc", "dosage", "dose", "đổi thuốc", "ngừng thuốc",
            "tự điều trị", "self-treat", "diagnose me", "chẩn đoán tôi"
        ]

        if any(term in q for term in unsafe_terms):
            return (
                "I can't diagnose a condition, prescribe medication, recommend "
                "a medication dose, or tell you to start/stop/change treatment. "
                "For treatment decisions, please consult a qualified healthcare "
                "professional."
            )

        # Explain the current assessment when available.
        if assessment_data:
            decision_data = assessment_data.get("decision", {})
            prediction_data = assessment_data.get("prediction", {})

            if any(word in q for word in [
                "current result", "my result", "kết quả", "risk", "rủi ro",
                "overall", "tổng thể", "priority", "ưu tiên"
            ]):
                overall = decision_data.get("overall_risk", "N/A")
                priority = decision_data.get("priority_disease", "N/A")

                lines = [
                    f"Mức nguy cơ tổng thể: {overall}",
                    f"Bệnh có mức ưu tiên cao nhất: {priority}",
                    ""
                ]

                mapping = {
                    "cardio": "Cardiovascular Disease",
                    "diabetes": "Diabetes",
                    "hypertension": "Hypertension"
                }

                for key, name in mapping.items():
                    result = prediction_data.get(key, {})
                    probability = result.get("probability")
                    disease_decision = decision_data.get("diseases", {}).get(
                        "cardiovascular" if key == "cardio" else key,
                        {}
                    )
                    if probability is not None:
                        level = disease_decision.get("risk_level", "N/A")
                        lines.append(
                            f"{name}: {float(probability):.2%} "
                            f"({level})"
                        )

                lines.append(
                    "Các giá trị này là ước tính nguy cơ do mô hình tạo ra, không phải là chẩn đoán y khoa."
                )
                return "\n".join(lines)

            if any(word in q for word in [
                "why high", "why moderate", "why low", "tại sao cao",
                "tại sao trung bình", "tại sao thấp", "why my risk",
                "vì sao rủi ro"
            ]):
                return (
                    "The risk level is determined by the Decision Engine from the "
                    "probabilities produced by the three disease models. A higher "
                    "model probability leads to a higher risk category. This is a "
                    "modeling result, not a statement that a particular feature "
                    "caused the disease."
                )

        # General educational answers.
        if any(word in q for word in ["blood pressure", "huyết áp", "bp"]):
            return (
                "Blood pressure has two main values: systolic pressure and "
                "diastolic pressure. Repeatedly elevated readings can be an "
                "important cardiovascular risk factor. The app uses blood-pressure "
                "features as model inputs, but the model output should not be treated "
                "as a clinical diagnosis."
            )

        if any(word in q for word in ["bmi", "cân nặng", "weight"]):
            return (
                "BMI is a simple ratio of weight to height used as one population-level "
                "indicator related to body size. It does not by itself diagnose a disease "
                "and does not directly describe body composition."
            )

        if any(word in q for word in ["diabetes", "tiểu đường", "glucose", "đường huyết"]):
            return (
                "Glucose is an important variable in diabetes risk assessment. Other "
                "factors can also contribute to risk. This application estimates risk "
                "from a machine-learning model and is not a diagnostic test."
            )

        if any(word in q for word in ["hypertension", "tăng huyết áp"]):
            return (
                "Hypertension is commonly associated with persistently elevated blood "
                "pressure. The application estimates hypertension risk from several "
                "features, including blood pressure and other health indicators."
            )

        if any(word in q for word in ["cardiovascular", "tim mạch", "heart disease"]):
            return (
                "Cardiovascular risk can be associated with multiple factors, including "
                "age, blood pressure, weight, smoking, cholesterol, and activity. In this "
                "project, the cardiovascular model combines its input features to produce "
                "a risk estimate."
            )

        if any(word in q for word in ["ai", "model", "machine learning","ml", "mô hình"]):
            return (
                "This application uses separate machine-learning models for cardiovascular "
                "disease, diabetes, and hypertension. Their outputs are passed to the "
                "Decision Engine, which assigns risk levels. The system is designed for "
                "screening and decision support rather than diagnosis."
            )

        return (
            "I can help explain general health concepts, the meaning of the risk "
            "assessment, and how this AI system works. I cannot provide a diagnosis, "
            "prescription, medication dosage, or individualized treatment plan."
        )

    st.markdown("### 💡 Quick Questions")
    quick_questions = [
        "What does my current result mean?",
        "Why is my overall risk HIGH?",
        "Which disease has the highest risk?",
        "Why is my cardiovascular risk high?",
        "Why is my hypertension risk high?",
        "Why is my diabetes risk high?",
        "Does 70% risk mean I definitely have the disease?",
        "What factors affect the AI prediction?",
        "How does the AI make predictions?",
        "What is SHAP?",
        "What is high blood pressure?",
        "What is BMI?",
    ]

    # Keep the selected Quick Question across Streamlit reruns.
    if "ask_ai_question" not in st.session_state:
        st.session_state["ask_ai_question"] = ""

    if "ask_ai_auto_submit" not in st.session_state:
        st.session_state["ask_ai_auto_submit"] = False

    cols = st.columns(2)
    for idx, quick_question in enumerate(quick_questions):
        if cols[idx % 2].button(
            quick_question,
            key=f"quick_q_{idx}",
            use_container_width=True,
        ):
            # Quick Question: save it and automatically answer it.
            st.session_state["ask_ai_question"] = quick_question
            st.session_state["ask_ai_auto_submit"] = True
            st.rerun()

    # Manual question input.
    # A manually typed question is NOT auto-submitted.
    question = st.text_input(
        "Your question",
        key="ask_ai_question",
        placeholder="Example: Why is my cardiovascular risk high?"
    )

    # Manual questions require clicking ASK AI.
    ask_button = st.button("🤖 ASK AI", type="secondary")

    # Only Quick Questions use auto-submit.
    should_answer = (
        ask_button
        or st.session_state.get("ask_ai_auto_submit", False)
    )

    if should_answer:
        current_assessment = st.session_state.get("assessment")
        answer = ask_ai_local(question, current_assessment)

        st.markdown("### 💬 AI Response")
        st.info(answer)

        # Reset auto-submit so typing a new question requires ASK AI.
        st.session_state["ask_ai_auto_submit"] = False

    st.caption(
        "AI Health Information Assistant — educational prototype. "
        "It does not diagnose, prescribe, or replace healthcare professionals."
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
