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

import pandas as pd
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

from storage import patient_records


# ============================================================
# 4. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Trợ lý Sức khỏe AI",
    page_icon="🩺",
    layout="wide"
)

# Keep only the latest completed assessment.
if "assessment" not in st.session_state:
    st.session_state["assessment"] = None


# ============================================================
# 4.1 SECTION VISIBILITY FLAGS
# ============================================================

# Set to True to show these sections again in the UI.
SHOW_TRUSTWORTHY_AI = False
SHOW_DEFENSE_QA = False


# ============================================================
# 4.1.1 VIETNAMESE DISPLAY LABELS
# ============================================================

# The engines keep their English keys (LOW / MODERATE / HIGH and the
# disease keys). Only the text shown on screen is translated.

RISK_LEVEL_VI = {
    "LOW": "THẤP",
    "MODERATE": "TRUNG BÌNH",
    "HIGH": "CAO",
}

DISEASE_NAME_VI = {
    "cardio": "Bệnh tim mạch",
    "cardiovascular": "Bệnh tim mạch",
    "diabetes": "Đái tháo đường",
    "hypertension": "Tăng huyết áp",
}


def risk_level_vi(level):
    """Hiển thị mức nguy cơ bằng tiếng Việt."""

    return RISK_LEVEL_VI.get(
        str(level).upper(),
        level
    )


def disease_name_vi(key):
    """Hiển thị tên bệnh bằng tiếng Việt."""

    return DISEASE_NAME_VI.get(
        str(key).lower(),
        key
    )


# ============================================================
# 4.2 PATIENT RECORDS (SIDEBAR)
# ============================================================

patient_records.init_db()

if "selected_patient_id" not in st.session_state:
    st.session_state["selected_patient_id"] = None

saved_patients = patient_records.list_patients()

with st.sidebar:

    st.header("👥 Bệnh nhân")

    if saved_patients:

        patient_ids = [
            patient["id"] for patient in saved_patients
        ]

        patient_labels = {
            patient["id"]: (
                f"{patient['name']} "
                f"({patient['measurement_count']} lần đo)"
            )
            for patient in saved_patients
        }

        current_id = st.session_state["selected_patient_id"]

        default_index = (
            patient_ids.index(current_id)
            if current_id in patient_ids
            else 0
        )

        st.session_state["selected_patient_id"] = st.selectbox(
            "Bệnh nhân đang chọn",
            options=patient_ids,
            index=default_index,
            format_func=lambda pid: patient_labels[pid]
        )

    else:

        st.session_state["selected_patient_id"] = None

        st.info(
            "Chưa có bệnh nhân nào. Thêm bệnh nhân để bắt đầu "
            "theo dõi các lần đo theo thời gian."
        )

    with st.expander("➕ Thêm bệnh nhân mới"):

        new_patient_name = st.text_input(
            "Họ và tên",
            key="new_patient_name"
        )

        new_patient_gender = st.selectbox(
            "Giới tính",
            options=["", "Nữ", "Nam"],
            key="new_patient_gender"
        )

        new_patient_birth_year = st.number_input(
            "Năm sinh",
            min_value=1900,
            max_value=2100,
            value=1970,
            key="new_patient_birth_year"
        )

        new_patient_note = st.text_input(
            "Ghi chú (không bắt buộc)",
            key="new_patient_note"
        )

        if st.button("Lưu bệnh nhân", use_container_width=True):

            try:

                new_id = patient_records.add_patient(
                    name=new_patient_name,
                    gender=new_patient_gender or None,
                    birth_year=int(new_patient_birth_year),
                    note=new_patient_note or None
                )

                st.session_state["selected_patient_id"] = new_id

                st.success(
                    f"Đã lưu bệnh nhân '{new_patient_name}'."
                )

                st.rerun()

            except ValueError as error:

                st.error(str(error))

    selected_patient_id = st.session_state["selected_patient_id"]

    if selected_patient_id is not None:

        with st.expander("🗑️ Xoá bệnh nhân đang chọn"):

            st.warning(
                "Xoá bệnh nhân sẽ xoá luôn toàn bộ số liệu đo "
                "đã lưu của người đó."
            )

            if st.button(
                "Xoá bệnh nhân",
                use_container_width=True
            ):

                patient_records.delete_patient(
                    selected_patient_id
                )

                st.session_state["selected_patient_id"] = None

                st.rerun()


# ============================================================
# 5. HEADER
# ============================================================

st.title("🩺 Trợ lý Sức khỏe AI")

st.markdown(
    """
    ### Đánh giá nguy cơ sức khỏe bằng AI

    Hệ thống ước tính nguy cơ mắc:

    - Bệnh tim mạch
    - Đái tháo đường
    - Tăng huyết áp

    **Lưu ý:** Đây là ước tính nguy cơ do AI đưa ra,
    không phải chẩn đoán y khoa.
    """
)

st.divider()


# ============================================================
# 6. PATIENT INFORMATION
# ============================================================

st.header("👤 Thông tin bệnh nhân")

col1, col2, col3 = st.columns(3)

with col1:

    age = st.number_input(
        "Tuổi",
        min_value=1,
        max_value=120,
        value=55
    )

    gender = st.selectbox(
        "Giới tính",
        options=[1, 2],
        format_func=lambda x:
            "Nữ" if x == 1 else "Nam",
        key="cardio_gender"
    )

    height = st.number_input(
        "Chiều cao (cm)",
        min_value=50,
        max_value=250,
        value=170
    )

    weight = st.number_input(
        "Cân nặng (kg)",
        min_value=20.0,
        max_value=300.0,
        value=75.0
    )


with col2:

    ap_hi = st.number_input(
        "Huyết áp tâm thu (mmHg)",
        min_value=50,
        max_value=250,
        value=140
    )

    ap_lo = st.number_input(
        "Huyết áp tâm trương (mmHg)",
        min_value=30,
        max_value=150,
        value=90
    )

    cholesterol = st.selectbox(
        "Cholesterol",
        options=[1, 2, 3],
        format_func=lambda x:
            {
                1: "Bình thường",
                2: "Trên mức bình thường",
                3: "Cao hơn nhiều mức bình thường"
            }[x]
    )

    gluc = st.selectbox(
        "Mức đường huyết",
        options=[1, 2, 3],
        format_func=lambda x:
            {
                1: "Bình thường",
                2: "Trên mức bình thường",
                3: "Cao hơn nhiều mức bình thường"
            }[x]
    )


with col3:

    smoke = st.selectbox(
        "Hút thuốc",
        options=[0, 1],
        format_func=lambda x:
            "Không" if x == 0 else "Có"
    )

    alco = st.selectbox(
        "Uống rượu bia",
        options=[0, 1],
        format_func=lambda x:
            "Không" if x == 0 else "Có"
    )

    active = st.selectbox(
        "Vận động thể chất",
        options=[0, 1],
        format_func=lambda x:
            "Không" if x == 0 else "Có"
    )


# ============================================================
# 7. DIABETES INFORMATION
# ============================================================

st.divider()

st.header("🩸 Thông tin đái tháo đường")

col1, col2, col3 = st.columns(3)

with col1:

    pregnancies = st.number_input(
        "Số lần mang thai",
        min_value=0,
        max_value=20,
        value=2
    )

    glucose = st.number_input(
        "Đường huyết (Glucose)",
        min_value=0.0,
        max_value=300.0,
        value=148.0
    )

    blood_pressure = st.number_input(
        "Huyết áp",
        min_value=0.0,
        max_value=200.0,
        value=90.0
    )


with col2:

    skin_thickness = st.number_input(
        "Độ dày nếp da (mm)",
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
        "BMI (chỉ số khối cơ thể)",
        min_value=0.0,
        max_value=80.0,
        value=32.0
    )

    diabetes_pedigree = st.number_input(
        "Hệ số tiền sử gia đình (DPF)",
        min_value=0.0,
        max_value=3.0,
        value=0.6
    )

    diabetes_age = st.number_input(
        "Tuổi (mô hình đái tháo đường)",
        min_value=1,
        max_value=120,
        value=55
    )


# ============================================================
# 8. HYPERTENSION INFORMATION
# ============================================================

st.divider()

st.header("❤️ Thông tin tăng huyết áp")

col1, col2, col3 = st.columns(3)

with col1:

    male = st.selectbox(
        "Giới tính",
        options=[0, 1],
        format_func=lambda x:
            "Nữ" if x == 0 else "Nam",
        key="hypertension_sex"
    )

    current_smoker = st.selectbox(
        "Hiện đang hút thuốc",
        options=[0, 1],
        format_func=lambda x:
            "Không" if x == 0 else "Có"
    )

    cigs_per_day = st.number_input(
        "Số điếu thuốc mỗi ngày",
        min_value=0.0,
        max_value=100.0,
        value=0.0
    )

    bp_meds = st.selectbox(
        "Đang dùng thuốc huyết áp",
        options=[0, 1],
        format_func=lambda x:
            "Không" if x == 0 else "Có"
    )


with col2:

    hypertension_diabetes = st.selectbox(
        "Tiền sử đái tháo đường",
        options=[0, 1],
        format_func=lambda x:
            "Không" if x == 0 else "Có"
    )

    total_chol = st.number_input(
        "Cholesterol toàn phần",
        min_value=50.0,
        max_value=500.0,
        value=230.0
    )

    systolic_bp = st.number_input(
        "Huyết áp tâm thu",
        min_value=50.0,
        max_value=300.0,
        value=140.0
    )

    diastolic_bp = st.number_input(
        "Huyết áp tâm trương",
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
        "Nhịp tim (lần/phút)",
        min_value=30.0,
        max_value=220.0,
        value=75.0
    )

    hypertension_glucose = st.number_input(
        "Đường huyết",
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

st.header("🔍 Đánh giá nguy cơ sức khỏe")

assess_button = st.button(
    "🩺 ĐÁNH GIÁ NGUY CƠ",
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
            "Tuổi phải nằm trong khoảng 1 đến 120."
        )

    if not 50 <= patient["height"] <= 250:
        errors.append(
            "Chiều cao phải nằm trong khoảng 50 đến 250 cm."
        )

    if not 20 <= patient["weight"] <= 300:
        errors.append(
            "Cân nặng phải nằm trong khoảng 20 đến 300 kg."
        )

    # --------------------------------------------------------
    # Cardiovascular blood pressure
    # --------------------------------------------------------

    if not 50 <= patient["ap_hi"] <= 250:
        errors.append(
            "Huyết áp tâm thu phải nằm trong khoảng "
            "50 đến 250 mmHg."
        )

    if not 30 <= patient["ap_lo"] <= 150:
        errors.append(
            "Huyết áp tâm trương phải nằm trong khoảng "
            "30 đến 150 mmHg."
        )

    if patient["ap_hi"] <= patient["ap_lo"]:
        errors.append(
            "Huyết áp tâm thu phải cao hơn "
            "huyết áp tâm trương."
        )

    # --------------------------------------------------------
    # Diabetes
    # --------------------------------------------------------

    if not 0 <= patient["Pregnancies"] <= 20:
        errors.append(
            "Số lần mang thai phải nằm trong khoảng 0 đến 20."
        )

    if not 0 <= patient["Glucose"] <= 300:
        errors.append(
            "Đường huyết (đái tháo đường) phải nằm trong khoảng 0 đến 300."
        )

    if not 0 <= patient["BMI"] <= 80:
        errors.append(
            "BMI phải nằm trong khoảng 0 đến 80."
        )

    if not 0 <= patient[
        "DiabetesPedigreeFunction"
    ] <= 3:
        errors.append(
            "Hệ số tiền sử gia đình (DPF) phải nằm "
            "trong khoảng 0 đến 3."
        )

    # --------------------------------------------------------
    # Hypertension
    # --------------------------------------------------------

    if not 0 <= patient["cigsPerDay"] <= 100:
        errors.append(
            "Số điếu thuốc mỗi ngày phải nằm trong khoảng 0 đến 100."
        )

    if not 50 <= patient["sysBP"] <= 300:
        errors.append(
            "Huyết áp tâm thu (tăng huyết áp) phải nằm "
            "trong khoảng 50 đến 300 mmHg."
        )

    if not 30 <= patient["diaBP"] <= 200:
        errors.append(
            "Huyết áp tâm trương (tăng huyết áp) phải nằm "
            "trong khoảng 30 đến 200 mmHg."
        )

    if patient["sysBP"] <= patient["diaBP"]:
        errors.append(
            "Huyết áp tâm thu phải cao hơn "
            "huyết áp tâm trương."
        )

    if not 30 <= patient["heartRate"] <= 220:
        errors.append(
            "Nhịp tim phải nằm trong khoảng 30 đến 220 lần/phút."
        )

    if not 30 <= patient["glucose"] <= 500:
        errors.append(
            "Đường huyết (tăng huyết áp) phải nằm "
            "trong khoảng 30 đến 500."
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
                "Vui lòng sửa các lỗi nhập liệu sau:"
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
            "Đã hoàn tất đánh giá sức khỏe."
        )

    except Exception as e:

        st.error(
            f"Đánh giá thất bại: {e}"
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

    st.header("📊 Kết quả đánh giá")

    # --------------------------------------------------------
    # Disease probabilities
    # --------------------------------------------------------

    st.subheader(
        "Xác suất nguy cơ theo từng bệnh"
    )

    probability_columns = st.columns(3)

    disease_mapping = {
        "cardio":
            "Bệnh tim mạch",
        "diabetes":
            "Đái tháo đường",
        "hypertension":
            "Tăng huyết áp"
    }

    # Mapping used by the Decision Engine and Recommendation Engine.
    # Keep it aligned with the disease keys returned by the application layer.
    decision_mapping = {
        "cardiovascular":
            "Bệnh tim mạch",
        "diabetes":
            "Đái tháo đường",
        "hypertension":
            "Tăng huyết áp"
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
        "Mức nguy cơ"
    )

    diseases = decision.get(
        "diseases",
        {}
    )

    # The Decision Engine supplies the risk level. The percentage shown here
    # is taken from the SAME prediction object displayed above.
    display_mapping = {
        "cardio": ("cardiovascular", "Bệnh tim mạch"),
        "diabetes": ("diabetes", "Đái tháo đường"),
        "hypertension": ("hypertension", "Tăng huyết áp")
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
                f"**{disease_name}:** {risk_level_vi(risk_level)}"
            )
        else:
            # Never use decision_result["risk_score"] for the displayed %.
            st.write(
                f"**{disease_name}:** "
                f"{risk_level_vi(risk_level)} "
                f"({float(probability):.2%})"
            )

    # --------------------------------------------------------
    # Overall risk
    # --------------------------------------------------------

    st.subheader(
        "Đánh giá nguy cơ tổng thể"
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
            "Nguy cơ tổng thể",
            risk_level_vi(overall_risk)
        )

    with col2:

        st.metric(
            "Bệnh cần ưu tiên",
            disease_name_vi(priority_disease)
        )

    # --------------------------------------------------------
    # Risk counts
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        st.write(
            "**Số bệnh nguy cơ cao:**",
            decision.get(
                "high_risk_count",
                0
            )
        )

    with col2:

        st.write(
            "**Số bệnh nguy cơ trung bình:**",
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
        "💡 Khuyến nghị sức khỏe"
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
            f"Mức nguy cơ: **{risk_level_vi(risk_level)}**"
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
            "Khuyến nghị chung"
        )

        for recommendation in (
            overall_recommendations
        ):

            st.write(
                f"• {recommendation}"
            )

    # ========================================================
    # SAVE MEASUREMENT
    # ========================================================

    st.divider()

    st.subheader("💾 Lưu lần đo này")

    selected_patient_id = st.session_state.get(
        "selected_patient_id"
    )

    if selected_patient_id is None:

        st.info(
            "Chọn hoặc thêm bệnh nhân ở thanh bên để lưu kết quả "
            "đánh giá này và theo dõi bệnh nhân theo thời gian."
        )

    else:

        selected_patient = patient_records.get_patient(
            selected_patient_id
        )

        measurement_note = st.text_input(
            "Ghi chú cho lần đo này (không bắt buộc)",
            key="measurement_note"
        )

        if st.button(
            f"💾 Lưu lần đo cho {selected_patient['name']}",
            use_container_width=True
        ):

            measurement_values = {
                "age": int(age),
                "height": float(height),
                "weight": float(weight),
                "bmi": float(bmi),
                "systolic_bp": float(ap_hi),
                "diastolic_bp": float(ap_lo),
                "heart_rate": float(heart_rate),
                "glucose": float(glucose),
                "total_cholesterol": float(total_chol),
                "cardio_probability": prediction.get(
                    "cardio", {}
                ).get("probability"),
                "diabetes_probability": prediction.get(
                    "diabetes", {}
                ).get("probability"),
                "hypertension_probability": prediction.get(
                    "hypertension", {}
                ).get("probability"),
                "overall_risk": decision.get("overall_risk"),
                "priority_disease": decision.get(
                    "priority_disease"
                ),
                "note": measurement_note or None,
            }

            patient_records.add_measurement(
                selected_patient_id,
                measurement_values
            )

            st.success(
                "Đã lưu lần đo cho "
                f"{selected_patient['name']}."
            )

            st.rerun()

    # ========================================================
    # TRUSTWORTHY AI DASHBOARD
    # ========================================================

    if SHOW_TRUSTWORTHY_AI:
        st.divider()
        st.header("🤖 Trí tuệ nhân tạo đáng tin cậy")

        st.markdown(
            """
            Phần này trình bày đánh giá Trustworthy AI của hệ thống
            đánh giá nguy cơ sức khỏe, bao gồm: tính công bằng, độ bền
            vững, khả năng giải thích, thiên lệch, quyền riêng tư, trách
            nhiệm giải trình, tác động xã hội và nền tảng toán học.
            """
        )

        ta_tabs = st.tabs([
            "📋 Tổng quan",
            "🔍 Khả năng giải thích & SHAP",
            "⚖️ Công bằng & Thiên lệch",
            "🛡️ Độ bền vững",
            "🔐 Quyền riêng tư",
            "📋 Trách nhiệm giải trình",
            "🌍 Tác động xã hội",
            "🧮 Nền tảng toán học"
        ])

        # ------------------------------------------------------------
        # Overview
        # ------------------------------------------------------------
        with ta_tabs[0]:

            st.subheader("Tình trạng đánh giá Trustworthy AI")

            overview_items = [
                ("⚖️ Công bằng", "Đã đánh giá"),
                ("🛡️ Độ bền vững", "Đã đánh giá"),
                ("🔍 Khả năng giải thích", "Đã đánh giá"),
                ("📊 Phân tích thiên lệch", "Đã đánh giá"),
                ("🔐 Quyền riêng tư", "Đã đánh giá"),
                ("📋 Trách nhiệm giải trình", "Đã đánh giá"),
                ("🌍 Tác động xã hội", "Đã đánh giá"),
                ("🧮 Nền tảng toán học", "Đã đánh giá"),
            ]

            overview_cols = st.columns(4)

            for i, (name, status) in enumerate(overview_items):
                with overview_cols[i % 4]:
                    st.metric(name, status)

            st.info(
                "Đánh giá Trustworthy AI nhằm hỗ trợ sử dụng hệ thống một "
                "cách có trách nhiệm. Nó không bảo đảm mọi dự đoán đều đúng "
                "hoặc hoàn toàn không có thiên lệch."
            )

            st.subheader("Các hạn chế đã biết")

            limitations = [
                "Hiệu năng có thể khác nhau giữa các nhóm nhân khẩu học.",
                "Hiệu năng mô hình có thể giảm khi dữ liệu đầu vào bị nhiễu.",
                "Vẫn có khả năng dự đoán dương tính giả và âm tính giả.",
                "Dữ liệu y tế chứa thông tin nhạy cảm.",
                "Bộ dữ liệu CARDIO chứa định danh trực tiếp.",
                "Quyết định y khoa vẫn cần sự giám sát của con người.",
                "Chưa triển khai giám sát vận hành và nhật ký kiểm toán."
            ]

            for item in limitations:
                st.write(f"• {item}")

        # ------------------------------------------------------------
        # Explainability & SHAP
        # ------------------------------------------------------------
        with ta_tabs[1]:

            st.subheader("🔍 Khả năng giải thích của mô hình")

            explainability = {
                "Bệnh tim mạch": ("ap_hi", 0.145809),
                "Đái tháo đường": ("SkinThickness", 0.134053),
                "Tăng huyết áp": ("sysBP", 0.200924),
            }

            for disease, (feature, importance) in explainability.items():
                with st.expander(disease, expanded=True):
                    st.write(
                        f"**Đặc trưng ảnh hưởng nhiều nhất:** `{feature}`"
                    )
                    st.write(
                        f"**Giá trị SHAP tuyệt đối trung bình:** {importance:.6f}"
                    )

            st.markdown("---")
            st.subheader("Biểu đồ SHAP")

            xai_dir = PROJECT_ROOT / "data" / "models" / "xai"

            image_groups = {
                "Bệnh tim mạch": ["cardio"],
                "Đái tháo đường": ["diabetes"],
                "Tăng huyết áp": ["hypertension"],
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
                    "Không tìm thấy thư mục biểu đồ SHAP. "
                    "Kết quả giải thích dạng số vẫn được hiển thị ở trên."
                )

            st.caption(
                "Giá trị SHAP cho biết các đặc trưng đầu vào đóng góp thế nào "
                "vào dự đoán của mô hình. Đây là lời giải thích về mô hình, "
                "không phải quan hệ nhân quả y khoa."
            )

        # ------------------------------------------------------------
        # Fairness & Bias
        # ------------------------------------------------------------
        with ta_tabs[2]:

            st.subheader("⚖️ Tính công bằng")

            st.write(
                "Phân tích công bằng so sánh hiệu năng mô hình giữa các nhóm "
                "nhân khẩu học khác nhau qua Accuracy, Precision, Recall và "
                "F1-score."
            )

            fairness_metrics = [
                "Accuracy",
                "Precision",
                "Recall",
                "F1-score"
            ]

            for metric in fairness_metrics:
                st.write(f"• {metric}: đã đánh giá theo từng nhóm")

            st.warning(
                "Chênh lệch hiệu năng giữa các nhóm cần được theo dõi. Chỉ "
                "riêng chênh lệch hiệu năng thì chưa đủ để kết luận là có "
                "thiên lệch mang tính phân biệt đối xử."
            )

            st.subheader("📊 Phân tích thiên lệch")

            st.write(
                "Phân tích thiên lệch giúp phát hiện chênh lệch hiệu năng và "
                "các nguồn gây khác biệt mang tính hệ thống."
            )

            st.success("Đã hoàn tất phân tích thiên lệch.")

            st.caption(
                "Bản thử nghiệm hiện tại tập trung vào đánh giá và theo dõi, "
                "không khẳng định mô hình hoàn toàn không có thiên lệch."
            )

        # ------------------------------------------------------------
        # Robustness
        # ------------------------------------------------------------
        with ta_tabs[3]:

            st.subheader("🛡️ Đánh giá độ bền vững")

            st.write(
                "Kiểm thử độ bền vững xem xét hiệu năng mô hình có ổn định "
                "hay không khi dữ liệu đầu vào bị nhiễu hoặc bị thay đổi nhẹ."
            )

            st.code(
                """Dữ liệu đầu vào gốc
          ↓
    Thêm nhiễu / thay đổi nhẹ
          ↓
    Chạy mô hình
          ↓
    So sánh dự đoán
          ↓
    Đo mức suy giảm hiệu năng""",
                language="text"
            )

            st.warning(
                "Kết quả cho thấy hiệu năng mô hình có thể giảm khi dữ liệu "
                "đầu vào bị nhiễu. Vì vậy cần diễn giải dự đoán một cách "
                "thận trọng."
            )

            st.success("Đã hoàn tất phân tích độ bền vững.")

        # ------------------------------------------------------------
        # Privacy
        # ------------------------------------------------------------
        with ta_tabs[4]:

            st.subheader("🔐 Quyền riêng tư")

            st.write(
                "Phân tích quyền riêng tư kiểm tra xem bộ dữ liệu y tế có "
                "chứa định danh trực tiếp hay không và chỉ ra sự hiện diện "
                "của thông tin sức khỏe nhạy cảm."
            )

            privacy_items = [
                ("Dữ liệu y tế", "Có chứa thông tin sức khỏe nhạy cảm"),
                ("Bộ dữ liệu CARDIO", "Phát hiện định danh trực tiếp trong dữ liệu gốc"),
                ("Mục đích hệ thống", "Ước tính nguy cơ / hỗ trợ quyết định"),
            ]

            for item, result in privacy_items:
                st.write(f"**{item}:** {result}")

            st.warning(
                "Bộ dữ liệu gốc CARDIO chứa định danh trực tiếp. Khi triển "
                "khai thực tế, những định danh này phải được loại bỏ hoặc "
                "bảo vệ, không được để lộ."
            )

            st.success("Đã hoàn tất phân tích quyền riêng tư.")

        # ------------------------------------------------------------
        # Accountability
        # ------------------------------------------------------------
        with ta_tabs[5]:

            st.subheader("📋 Trách nhiệm giải trình")

            accountability_items = [
                "Hệ thống nêu rõ ba mô hình tương ứng với ba bệnh.",
                "Kết quả dự đoán được đưa qua Decision Engine.",
                "Mức nguy cơ được phân loại rõ ràng: THẤP, TRUNG BÌNH hoặc CAO.",
                "Khuyến nghị được tách bạch với chẩn đoán y khoa.",
                "Hệ thống có cảnh báo về giới hạn sử dụng trong y tế.",
                "Nhân viên y tế vẫn chịu trách nhiệm về quyết định cuối cùng."
            ]

            for item in accountability_items:
                st.write(f"✅ {item}")

            st.info(
                "Hệ thống AI này là bản thử nghiệm hỗ trợ quyết định, không "
                "thay thế nhân viên y tế có chuyên môn."
            )

        # ------------------------------------------------------------
        # Social Impact
        # ------------------------------------------------------------
        with ta_tabs[6]:

            st.subheader("🌍 Tác động xã hội")

            col_positive, col_negative = st.columns(2)

            with col_positive:
                st.markdown("### Tác động tích cực tiềm năng")

                positive = [
                    "Hỗ trợ sàng lọc nguy cơ sớm.",
                    "Giúp người dùng theo dõi các chỉ số sức khỏe quan trọng.",
                    "Cung cấp ước tính nguy cơ có AI hỗ trợ, dễ tiếp cận.",
                    "Cung cấp thêm thông tin tham khảo cho nhân viên y tế."
                ]

                for item in positive:
                    st.write(f"🟢 {item}")

            with col_negative:
                st.markdown("### Tác động tiêu cực tiềm năng")

                negative = [
                    "Dự đoán dương tính giả có thể gây lo lắng không cần thiết.",
                    "Dự đoán âm tính giả có thể tạo cảm giác an tâm sai lầm.",
                    "Người dùng có thể phụ thuộc quá mức vào kết quả của AI.",
                    "Thiên lệch hoặc chênh lệch hiệu năng có thể ảnh hưởng tới một số nhóm."
                ]

                for item in negative:
                    st.write(f"🔴 {item}")

            st.info(
                "Giám sát của con người, khả năng giải thích, đánh giá công "
                "bằng và cảnh báo rõ ràng là những biện pháp giảm thiểu quan "
                "trọng."
            )

        # ------------------------------------------------------------
        # Math AI
        # ------------------------------------------------------------
        with ta_tabs[7]:

            st.subheader("🧮 Nền tảng toán học")

            math_items = [
                (
                    "Giải tích",
                    "Nền tảng để hiểu về sự biến thiên, tối ưu hóa và các "
                    "khái niệm toán học liên quan tới mô hình."
                ),
                (
                    "Đại số tuyến tính",
                    "Nền tảng toán học cho vector, ma trận và cách biểu diễn "
                    "đặc trưng."
                ),
                (
                    "Đại số tuyến tính nâng cao",
                    "Hỗ trợ phân tích và biến đổi dựa trên ma trận."
                ),
                (
                    "Xác suất",
                    "Nền tảng cho các ước tính nguy cơ mang tính xác suất."
                ),
                (
                    "Thống kê",
                    "Hỗ trợ thống kê mô tả, độ biến thiên, tương quan và "
                    "đánh giá mô hình."
                ),
            ]

            for topic, description in math_items:
                with st.expander(topic):
                    st.write(description)

            st.success("Đã hoàn tất phần nền tảng toán học.")

    # ========================================================
    # DEFENSE / Q&A
    # ========================================================
    if SHOW_DEFENSE_QA:
        st.divider()
        st.header("❓ Câu hỏi và câu trả lời bảo vệ")
        st.markdown(
            "Dùng các câu hỏi này để trình bày về độ an toàn, giới hạn và thiết kế sử dụng có trách nhiệm của hệ thống khi demo hoặc bảo vệ đồ án."
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
            "Nguyên tắc bảo vệ: AI hỗ trợ sàng lọc và hỗ trợ quyết định; không thay thế chẩn đoán lâm sàng hay quyết định y khoa."
        )

    # ========================================================
    # ASK AI - HEALTH INFORMATION ASSISTANT
    # ========================================================
    st.divider()
    st.header("🤖 Hỏi AI")
    st.markdown(
        "Đặt câu hỏi chung về sức khỏe hoặc nhờ trợ lý giải thích kết quả "
        "đánh giá hiện tại. Trợ lý chỉ nhằm mục đích tham khảo và hỗ trợ "
        "quyết định."
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
                "Tôi không thể chẩn đoán bệnh, kê đơn, đề xuất liều thuốc, "
                "hay yêu cầu bạn bắt đầu/ngừng/thay đổi điều trị. "
                "Với các quyết định điều trị, vui lòng trao đổi với nhân viên "
                "y tế có chuyên môn."
            )

        # Explain the current assessment when available.
        if assessment_data:
            decision_data = assessment_data.get("decision", {})
            prediction_data = assessment_data.get("prediction", {})

            if any(word in q for word in [
                "current result", "my result", "kết quả", "risk", "rủi ro",
                "overall", "tổng thể", "priority", "ưu tiên", "nguy cơ",
                "bệnh nào"
            ]):
                overall = decision_data.get("overall_risk", "N/A")
                priority = decision_data.get("priority_disease", "N/A")

                lines = [
                    f"Mức nguy cơ tổng thể: {risk_level_vi(overall)}",
                    f"Bệnh có mức ưu tiên cao nhất: {disease_name_vi(priority)}",
                    ""
                ]

                mapping = {
                    "cardio": "Bệnh tim mạch",
                    "diabetes": "Đái tháo đường",
                    "hypertension": "Tăng huyết áp"
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
                            f"({risk_level_vi(level)})"
                        )

                lines.append(
                    "Các giá trị này là ước tính nguy cơ do mô hình tạo ra, không phải là chẩn đoán y khoa."
                )
                return "\n".join(lines)

            if any(word in q for word in [
                "why high", "why moderate", "why low", "tại sao cao",
                "tại sao trung bình", "tại sao thấp", "why my risk",
                "vì sao rủi ro", "vì sao nguy cơ", "tại sao nguy cơ"
            ]):
                return (
                    "Mức nguy cơ do Decision Engine xác định dựa trên xác suất "
                    "của ba mô hình bệnh. Xác suất mô hình càng cao thì mức "
                    "nguy cơ càng cao. Đây là kết quả của mô hình, không khẳng "
                    "định rằng một yếu tố cụ thể nào đó gây ra bệnh."
                )

        # General educational answers.
        if any(word in q for word in ["blood pressure", "huyết áp", "bp"]):
            return (
                "Huyết áp gồm hai trị số: huyết áp tâm thu và huyết áp tâm "
                "trương. Chỉ số cao lặp lại nhiều lần có thể là yếu tố nguy cơ "
                "tim mạch quan trọng. Ứng dụng dùng các chỉ số huyết áp làm đầu "
                "vào cho mô hình, nhưng kết quả mô hình không phải là chẩn đoán "
                "lâm sàng."
            )

        if any(word in q for word in ["bmi", "cân nặng", "weight"]):
            return (
                "BMI là tỉ số đơn giản giữa cân nặng và chiều cao, dùng như một "
                "chỉ báo ở mức quần thể liên quan tới thể trạng. Bản thân BMI không "
                "chẩn đoán bệnh và không mô tả trực tiếp thành phần cơ thể."
            )

        if any(word in q for word in ["diabetes", "tiểu đường", "glucose", "đường huyết"]):
            return (
                "Đường huyết là biến quan trọng khi đánh giá nguy cơ đái tháo đường. "
                "Nhiều yếu tố khác cũng góp phần vào nguy cơ. Ứng dụng ước tính nguy "
                "cơ bằng mô hình học máy, không phải là xét nghiệm chẩn đoán."
            )

        if any(word in q for word in ["hypertension", "tăng huyết áp"]):
            return (
                "Tăng huyết áp thường gắn với tình trạng huyết áp cao kéo dài. "
                "Ứng dụng ước tính nguy cơ tăng huyết áp từ nhiều chỉ số, trong đó "
                "có huyết áp và các chỉ số sức khỏe khác."
            )

        if any(word in q for word in ["cardiovascular", "tim mạch", "heart disease"]):
            return (
                "Nguy cơ tim mạch liên quan tới nhiều yếu tố: tuổi, huyết áp, cân "
                "nặng, hút thuốc, cholesterol và mức vận động. Trong dự án này, mô "
                "hình tim mạch kết hợp các đặc trưng đầu vào để đưa ra ước tính "
                "nguy cơ."
            )

        if any(word in q for word in ["ai", "model", "machine learning","ml", "mô hình"]):
            return (
                "Ứng dụng dùng ba mô hình học máy riêng cho bệnh tim mạch, đái tháo "
                "đường và tăng huyết áp. Kết quả của chúng được đưa sang Decision "
                "Engine để xếp mức nguy cơ. Hệ thống phục vụ sàng lọc và hỗ trợ "
                "quyết định, không phải để chẩn đoán."
            )

        return (
            "Tôi có thể giải thích các khái niệm sức khỏe cơ bản, ý nghĩa của kết "
            "quả đánh giá nguy cơ và cách hệ thống AI này hoạt động. Tôi không "
            "chẩn đoán, không kê đơn, không đưa ra liều thuốc hay phác đồ điều "
            "trị cá nhân."
        )

    st.markdown("### 💡 Câu hỏi nhanh")
    quick_questions = [
        "Kết quả hiện tại của tôi có ý nghĩa gì?",
        "Vì sao nguy cơ tổng thể của tôi ở mức CAO?",
        "Bệnh nào đang có nguy cơ cao nhất?",
        "Vì sao nguy cơ tim mạch của tôi cao?",
        "Vì sao nguy cơ tăng huyết áp của tôi cao?",
        "Vì sao nguy cơ đái tháo đường của tôi cao?",
        "Nguy cơ 70% có nghĩa là chắc chắn mắc bệnh không?",
        "Những yếu tố nào ảnh hưởng tới dự đoán của AI?",
        "AI đưa ra dự đoán bằng cách nào?",
        "SHAP là gì?",
        "Huyết áp cao là gì?",
        "BMI là gì?",
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
        "Câu hỏi của bạn",
        key="ask_ai_question",
        placeholder="Ví dụ: Vì sao nguy cơ tim mạch của tôi cao?"
    )

    # Manual questions require clicking ASK AI.
    ask_button = st.button("🤖 HỎI AI", type="secondary")

    # Only Quick Questions use auto-submit.
    should_answer = (
        ask_button
        or st.session_state.get("ask_ai_auto_submit", False)
    )

    if should_answer:
        current_assessment = st.session_state.get("assessment")
        answer = ask_ai_local(question, current_assessment)

        st.markdown("### 💬 Trả lời của AI")
        st.info(answer)

        # Reset auto-submit so typing a new question requires ASK AI.
        st.session_state["ask_ai_auto_submit"] = False

    st.caption(
        "Trợ lý thông tin sức khỏe AI — bản thử nghiệm phục vụ học tập. "
        "Không chẩn đoán, không kê đơn và không thay thế nhân viên y tế."
    )

    # ========================================================
    # DISCLAIMER
    # ========================================================

    st.divider()

    st.warning(
        recommendations.get(
            "disclaimer",
            (
                "Đây là ước tính nguy cơ do AI đưa ra, "
                "không phải chẩn đoán y khoa."
            )
        )
    )


# ============================================================
# 11.5 PATIENT LIST & MEASUREMENT HISTORY
# ============================================================

st.divider()

st.header("📈 Hồ sơ bệnh nhân")

record_tabs = st.tabs([
    "👥 Danh sách bệnh nhân",
    "📊 Lịch sử đo"
])

with record_tabs[0]:

    all_patients = patient_records.list_patients()

    if not all_patients:

        st.info(
            "Chưa lưu bệnh nhân nào. Dùng thanh bên để "
            "thêm bệnh nhân đầu tiên."
        )

    else:

        st.caption(
            f"Đã lưu {len(all_patients)} bệnh nhân."
        )

        patient_table = pd.DataFrame([
            {
                "Họ tên": patient["name"],
                "Giới tính": patient["gender"] or "-",
                "Năm sinh": patient["birth_year"] or "-",
                "Số lần đo": patient["measurement_count"],
                "Lần đo gần nhất": (
                    patient["last_measured_at"] or "-"
                ),
                "Ghi chú": patient["note"] or "",
            }
            for patient in all_patients
        ])

        st.dataframe(
            patient_table,
            use_container_width=True,
            hide_index=True
        )

with record_tabs[1]:

    selected_patient_id = st.session_state.get(
        "selected_patient_id"
    )

    if selected_patient_id is None:

        st.info(
            "Chọn một bệnh nhân ở thanh bên để xem "
            "lịch sử đo của họ."
        )

    else:

        selected_patient = patient_records.get_patient(
            selected_patient_id
        )

        measurements = patient_records.list_measurements(
            selected_patient_id
        )

        st.subheader(
            f"Lịch sử đo — {selected_patient['name']}"
        )

        if not measurements:

            st.info(
                "Bệnh nhân này chưa có lần đo nào. Hãy chạy "
                "đánh giá rồi bấm 'Lưu lần đo này'."
            )

        else:

            history = pd.DataFrame(measurements)

            history["measured_at"] = pd.to_datetime(
                history["measured_at"]
            )

            # ------------------------------------------------
            # Latest values and change since previous visit
            # ------------------------------------------------

            latest = history.iloc[-1]

            previous = (
                history.iloc[-2]
                if len(history) > 1
                else None
            )

            def delta_of(column):
                """Change since the previous measurement."""

                if previous is None:
                    return None

                if pd.isna(latest[column]) or pd.isna(
                    previous[column]
                ):
                    return None

                return round(
                    float(latest[column])
                    - float(previous[column]),
                    1
                )

            metric_columns = st.columns(4)

            metric_columns[0].metric(
                "Huyết áp",
                f"{latest['systolic_bp']:.0f}/"
                f"{latest['diastolic_bp']:.0f}",
                delta=delta_of("systolic_bp")
            )

            metric_columns[1].metric(
                "Đường huyết",
                f"{latest['glucose']:.0f}",
                delta=delta_of("glucose")
            )

            metric_columns[2].metric(
                "Cân nặng (kg)",
                f"{latest['weight']:.1f}",
                delta=delta_of("weight")
            )

            metric_columns[3].metric(
                "Nguy cơ tổng thể",
                risk_level_vi(latest["overall_risk"] or "N/A")
            )

            st.caption(
                f"{len(history)} lần đo, từ "
                f"{history['measured_at'].min():%d/%m/%Y %H:%M} "
                f"đến "
                f"{history['measured_at'].max():%d/%m/%Y %H:%M}."
            )

            # ------------------------------------------------
            # Trends
            # ------------------------------------------------

            chart_data = history.set_index("measured_at")

            st.markdown("#### Xu hướng huyết áp")
            st.line_chart(
                chart_data[["systolic_bp", "diastolic_bp"]]
            )

            st.markdown("#### Đường huyết, cân nặng và nhịp tim")
            st.line_chart(
                chart_data[["glucose", "weight", "heart_rate"]]
            )

            risk_columns = [
                "cardio_probability",
                "diabetes_probability",
                "hypertension_probability",
            ]

            if chart_data[risk_columns].notna().any().any():

                st.markdown("#### Nguy cơ dự đoán theo thời gian")
                st.line_chart(chart_data[risk_columns])

            # ------------------------------------------------
            # Full table
            # ------------------------------------------------

            st.markdown("#### Toàn bộ các lần đo")

            st.dataframe(
                history.drop(columns=["id"]),
                use_container_width=True,
                hide_index=True
            )

            st.download_button(
                "⬇️ Tải lịch sử (CSV)",
                data=history.to_csv(index=False).encode("utf-8"),
                file_name=(
                    f"lich_su_do_{selected_patient['name']}.csv"
                ),
                mime="text/csv"
            )

            # ------------------------------------------------
            # Delete a single measurement
            # ------------------------------------------------

            with st.expander("🗑️ Xoá một lần đo"):

                measurement_choice = st.selectbox(
                    "Lần đo",
                    options=[
                        item["id"] for item in measurements
                    ],
                    format_func=lambda mid: next(
                        item["measured_at"]
                        for item in measurements
                        if item["id"] == mid
                    )
                )

                if st.button("Xoá lần đo"):

                    patient_records.delete_measurement(
                        measurement_choice
                    )

                    st.rerun()


# ============================================================
# 12. FOOTER
# ============================================================

st.divider()

st.caption(
    "Trợ lý Sức khỏe AI | "
    "Quy trình: Dự đoán → Quyết định → Khuyến nghị"
)
