"""
Trợ lý Sức khỏe AI — giao diện Streamlit.

Bố cục:
    Thanh bên   : chọn / thêm / xoá bệnh nhân
    Tab 1       : nhập chỉ số và xem kết quả đánh giá
    Tab 2       : hồ sơ bệnh nhân và lịch sử đo
    Tab 3       : hỏi đáp thông tin sức khỏe

Luồng xử lý giữ nguyên như cũ:
    Người dùng → HealthcareAssistant → Prediction → Decision → Recommendation
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st


# ============================================================
# 1. ĐƯỜNG DẪN DỰ ÁN
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


from application.healthcare_assistant import HealthcareAssistant  # noqa: E402
from storage import patient_records  # noqa: E402
from ui import sections, theme  # noqa: E402
from ui.labels import disease_name_vi, risk_level_vi  # noqa: E402


# ============================================================
# 2. CẤU HÌNH TRANG
# ============================================================

st.set_page_config(
    page_title="Trợ lý Sức khỏe AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

theme.inject_css()


# Đặt True để hiện lại hai phần dành cho báo cáo đồ án.
SHOW_TRUSTWORTHY_AI = False
SHOW_DEFENSE_QA = False


# Ba bệnh hệ thống đánh giá: khoá bên Prediction ↔ khoá bên Decision.
DISEASE_KEYS = [
    ("cardio", "cardiovascular"),
    ("diabetes", "diabetes"),
    ("hypertension", "hypertension"),
]


# ============================================================
# 3. TRẠNG THÁI PHIÊN
# ============================================================

patient_records.init_db()

st.session_state.setdefault("assessment", None)
st.session_state.setdefault("assessment_inputs", None)
st.session_state.setdefault("selected_patient_id", None)
st.session_state.setdefault("ask_ai_question", "")
st.session_state.setdefault("ask_ai_auto_submit", False)


# ============================================================
# 4. THANH BÊN — BỆNH NHÂN
# ============================================================

def render_sidebar():
    """Chọn, thêm và xoá bệnh nhân. Trả về bệnh nhân đang chọn."""

    saved_patients = patient_records.list_patients()

    with st.sidebar:

        st.markdown("### 👥 Bệnh nhân")

        if saved_patients:

            ids = [p["id"] for p in saved_patients]
            labels = {
                p["id"]: f"{p['name']} · {p['measurement_count']} lần đo"
                for p in saved_patients
            }

            current = st.session_state["selected_patient_id"]
            index = ids.index(current) if current in ids else 0

            st.session_state["selected_patient_id"] = st.selectbox(
                "Đang chọn",
                options=ids,
                index=index,
                format_func=lambda pid: labels[pid],
                label_visibility="collapsed"
            )

        else:

            st.session_state["selected_patient_id"] = None
            st.caption(
                "Chưa có bệnh nhân nào. Thêm bệnh nhân để lưu lại "
                "và theo dõi nhiều lần đo."
            )

        with st.expander("➕ Thêm bệnh nhân"):

            name = st.text_input("Họ và tên", key="new_patient_name")

            col_a, col_b = st.columns(2)

            gender = col_a.selectbox(
                "Giới tính",
                options=["", "Nữ", "Nam"],
                key="new_patient_gender"
            )

            birth_year = col_b.number_input(
                "Năm sinh",
                min_value=1900,
                max_value=2100,
                value=1970,
                key="new_patient_birth_year"
            )

            note = st.text_input(
                "Ghi chú",
                key="new_patient_note",
                placeholder="Bệnh nền, lưu ý…"
            )

            if st.button("Lưu bệnh nhân", use_container_width=True):

                try:
                    new_id = patient_records.add_patient(
                        name=name,
                        gender=gender or None,
                        birth_year=int(birth_year),
                        note=note or None
                    )
                    st.session_state["selected_patient_id"] = new_id
                    st.rerun()

                except ValueError as error:
                    st.error(str(error))

        selected_id = st.session_state["selected_patient_id"]
        selected = (
            patient_records.get_patient(selected_id)
            if selected_id is not None
            else None
        )

        if selected is not None:

            with st.expander("🗑️ Xoá bệnh nhân đang chọn"):
                st.caption(
                    "Xoá bệnh nhân sẽ xoá luôn toàn bộ số liệu đo của họ."
                )
                if st.button("Xoá vĩnh viễn", use_container_width=True):
                    patient_records.delete_patient(selected_id)
                    st.session_state["selected_patient_id"] = None
                    st.rerun()

        st.divider()
        st.caption(
            "⚠️ Công cụ hỗ trợ sàng lọc. Không chẩn đoán, không kê đơn, "
            "không thay thế nhân viên y tế."
        )

    return selected


# ============================================================
# 5. FORM NHẬP CHỈ SỐ
# ============================================================

def collect_inputs():
    """Vẽ form nhập liệu. Trả về (đã bấm gửi, dict chỉ số người dùng nhập)."""

    with st.form("assessment_form"):

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**👤 Thông tin cơ bản**")

            age = st.number_input(
                "Tuổi", min_value=1, max_value=120, value=55
            )
            gender_label = st.selectbox("Giới tính", options=["Nữ", "Nam"])
            height = st.number_input(
                "Chiều cao (cm)", min_value=50, max_value=250, value=170
            )
            weight = st.number_input(
                "Cân nặng (kg)",
                min_value=20.0, max_value=300.0, value=75.0, step=0.5
            )

        with col2:
            st.markdown("**🩺 Chỉ số đo được**")

            ap_hi = st.number_input(
                "Huyết áp tâm thu (mmHg)",
                min_value=50, max_value=250, value=140,
                help="Số lớn khi đo huyết áp."
            )
            ap_lo = st.number_input(
                "Huyết áp tâm trương (mmHg)",
                min_value=30, max_value=150, value=90,
                help="Số nhỏ khi đo huyết áp."
            )
            heart_rate = st.number_input(
                "Nhịp tim (lần/phút)",
                min_value=30.0, max_value=220.0, value=75.0
            )
            glucose = st.number_input(
                "Đường huyết (mg/dL)",
                min_value=30.0, max_value=500.0, value=110.0,
                help="Kết quả xét nghiệm đường huyết gần nhất."
            )
            total_chol = st.number_input(
                "Cholesterol toàn phần (mg/dL)",
                min_value=50.0, max_value=500.0, value=230.0
            )

        with col3:
            st.markdown("**🚭 Lối sống và tiền sử**")

            smoke = st.selectbox(
                "Hút thuốc", options=[0, 1],
                format_func=lambda x: "Không" if x == 0 else "Có"
            )
            cigs_per_day = st.number_input(
                "Số điếu mỗi ngày",
                min_value=0.0, max_value=100.0, value=0.0,
                help="Để 0 nếu không hút thuốc."
            )
            alco = st.selectbox(
                "Uống rượu bia", options=[0, 1],
                format_func=lambda x: "Không" if x == 0 else "Có"
            )
            active = st.selectbox(
                "Vận động thể chất đều đặn", options=[0, 1],
                format_func=lambda x: "Không" if x == 0 else "Có",
                index=1
            )
            bp_meds = st.selectbox(
                "Đang dùng thuốc huyết áp", options=[0, 1],
                format_func=lambda x: "Không" if x == 0 else "Có"
            )
            history_diabetes = st.selectbox(
                "Tiền sử đái tháo đường", options=[0, 1],
                format_func=lambda x: "Không" if x == 0 else "Có"
            )

        with st.expander(
            "⚙️ Chỉ số chuyên sâu — để mặc định nếu không có kết quả xét nghiệm"
        ):

            adv1, adv2, adv3 = st.columns(3)

            with adv1:
                cholesterol = st.selectbox(
                    "Mức cholesterol",
                    options=[1, 2, 3],
                    format_func=lambda x: {
                        1: "Bình thường",
                        2: "Trên bình thường",
                        3: "Cao hơn nhiều"
                    }[x]
                )
                gluc = st.selectbox(
                    "Mức đường huyết",
                    options=[1, 2, 3],
                    format_func=lambda x: {
                        1: "Bình thường",
                        2: "Trên bình thường",
                        3: "Cao hơn nhiều"
                    }[x]
                )

            with adv2:
                pregnancies = st.number_input(
                    "Số lần mang thai", min_value=0, max_value=20, value=2
                )
                skin_thickness = st.number_input(
                    "Độ dày nếp da (mm)",
                    min_value=0.0, max_value=100.0, value=35.0
                )

            with adv3:
                insulin = st.number_input(
                    "Insulin (µU/mL)",
                    min_value=0.0, max_value=1000.0, value=120.0
                )
                diabetes_pedigree = st.number_input(
                    "Hệ số tiền sử gia đình (DPF)",
                    min_value=0.0, max_value=3.0, value=0.6, step=0.1
                )

            bmi_override = st.number_input(
                "BMI (để 0 để hệ thống tự tính từ chiều cao và cân nặng)",
                min_value=0.0, max_value=80.0, value=0.0, step=0.1
            )

        submitted = st.form_submit_button(
            "🩺 ĐÁNH GIÁ NGUY CƠ",
            type="primary",
            use_container_width=True
        )

    return submitted, {
        "age": age,
        "gender_label": gender_label,
        "height": height,
        "weight": weight,
        "ap_hi": ap_hi,
        "ap_lo": ap_lo,
        "heart_rate": heart_rate,
        "glucose": glucose,
        "total_chol": total_chol,
        "smoke": smoke,
        "cigs_per_day": cigs_per_day,
        "alco": alco,
        "active": active,
        "bp_meds": bp_meds,
        "history_diabetes": history_diabetes,
        "cholesterol": cholesterol,
        "gluc": gluc,
        "pregnancies": pregnancies,
        "skin_thickness": skin_thickness,
        "insulin": insulin,
        "diabetes_pedigree": diabetes_pedigree,
        "bmi_override": bmi_override,
    }


def build_patient(form):
    """Suy ra bộ 30 trường mà tầng ứng dụng cần từ các chỉ số đã nhập.

    Nhiều trường trước đây phải nhập trùng nhau (tuổi, giới tính, BMI,
    huyết áp, đường huyết) nay được suy ra một lần từ cùng một nguồn.
    """

    is_male = form["gender_label"] == "Nam"

    height_m = form["height"] / 100
    bmi = (
        form["bmi_override"]
        if form["bmi_override"] > 0
        else round(form["weight"] / (height_m ** 2), 1)
    )

    return {
        # Tim mạch
        "age": form["age"],
        "gender": 2 if is_male else 1,
        "height": form["height"],
        "weight": form["weight"],
        "ap_hi": form["ap_hi"],
        "ap_lo": form["ap_lo"],
        "cholesterol": form["cholesterol"],
        "gluc": form["gluc"],
        "smoke": form["smoke"],
        "alco": form["alco"],
        "active": form["active"],

        # Đái tháo đường
        "Pregnancies": form["pregnancies"],
        "Glucose": form["glucose"],
        "BloodPressure": float(form["ap_lo"]),
        "SkinThickness": form["skin_thickness"],
        "Insulin": form["insulin"],
        "BMI": bmi,
        "DiabetesPedigreeFunction": form["diabetes_pedigree"],
        "Age": form["age"],

        # Tăng huyết áp
        "male": 1 if is_male else 0,
        "currentSmoker": form["smoke"],
        "cigsPerDay": form["cigs_per_day"],
        "BPMeds": form["bp_meds"],
        "diabetes": form["history_diabetes"],
        "totChol": form["total_chol"],
        "sysBP": float(form["ap_hi"]),
        "diaBP": float(form["ap_lo"]),
        "heartRate": form["heart_rate"],
        "glucose": form["glucose"],
    }


def validate(patient):
    """Kiểm tra các ràng buộc mà ô nhập liệu không tự bảo đảm được."""

    errors = []

    if patient["ap_hi"] <= patient["ap_lo"]:
        errors.append(
            "Huyết áp tâm thu phải cao hơn huyết áp tâm trương."
        )

    if not 10 <= patient["BMI"] <= 80:
        errors.append(
            f"BMI tính ra {patient['BMI']} nằm ngoài khoảng hợp lý "
            "(10–80). Kiểm tra lại chiều cao và cân nặng."
        )

    if patient["cigsPerDay"] > 0 and patient["currentSmoker"] == 0:
        errors.append(
            "Đã khai không hút thuốc nhưng số điếu mỗi ngày lớn hơn 0."
        )

    return errors


# ============================================================
# 6. HIỂN THỊ KẾT QUẢ
# ============================================================

def render_results(assessment):
    """Vẽ phần kết quả đánh giá."""

    prediction = assessment.get("prediction", {})
    decision = assessment.get("decision", {})
    recommendations = assessment.get("recommendations", {})
    diseases = decision.get("diseases", {})

    theme.summary_bar(
        decision.get("overall_risk", "N/A"),
        disease_name_vi(decision.get("priority_disease", "N/A")),
        decision.get("high_risk_count", 0),
        decision.get("moderate_risk_count", 0)
    )

    columns = st.columns(3)

    for column, (prediction_key, decision_key) in zip(columns, DISEASE_KEYS):

        probability = prediction.get(prediction_key, {}).get("probability", 0)
        level = diseases.get(decision_key, {}).get("risk_level", "N/A")

        with column:
            theme.risk_card(
                disease_name_vi(prediction_key),
                probability,
                level
            )

    st.write("")

    disease_recommendations = recommendations.get(
        "disease_recommendations", {}
    )

    tab_rec, tab_input = st.tabs([
        "💡 Khuyến nghị", "🔢 Số liệu đã dùng"
    ])

    with tab_rec:

        for _, decision_key in DISEASE_KEYS:

            item = disease_recommendations.get(decision_key, {})

            if not isinstance(item, dict) or not item:
                continue

            level = item.get("risk_level", "N/A")

            st.markdown(
                f"**{disease_name_vi(decision_key)}** — "
                f"{theme.risk_badge(level)}",
                unsafe_allow_html=True
            )

            for line in item.get("recommendations", []):
                st.markdown(f"- {line}")

            st.write("")

        overall = recommendations.get("overall_recommendations", [])

        if overall:
            st.markdown("**Khuyến nghị chung**")
            for line in overall:
                st.markdown(f"- {line}")

    with tab_input:

        used = st.session_state.get("assessment_inputs") or {}

        st.caption(
            "Các giá trị dưới đây được suy ra từ những gì bạn đã nhập và "
            "được đưa thẳng vào mô hình."
        )

        st.dataframe(
            pd.DataFrame(
                sorted(used.items()),
                columns=["Trường", "Giá trị"]
            ),
            use_container_width=True,
            hide_index=True,
            height=280
        )

    st.warning(
        recommendations.get(
            "disclaimer",
            "Đây là ước tính nguy cơ do AI đưa ra, không phải chẩn đoán y khoa."
        )
    )


def render_save_block(selected_patient):
    """Khối lưu lần đo vào hồ sơ bệnh nhân."""

    assessment = st.session_state["assessment"]
    used = st.session_state.get("assessment_inputs") or {}

    prediction = assessment.get("prediction", {})
    decision = assessment.get("decision", {})

    st.markdown("#### 💾 Lưu lần đo này")

    if selected_patient is None:
        st.info(
            "Chọn hoặc thêm bệnh nhân ở thanh bên để lưu kết quả và theo "
            "dõi thay đổi qua nhiều lần đo."
        )
        return

    col_note, col_button = st.columns([3, 1])

    note = col_note.text_input(
        "Ghi chú cho lần đo",
        key="measurement_note",
        placeholder="Ví dụ: đo buổi sáng, sau khi nghỉ 10 phút",
        label_visibility="collapsed"
    )

    if col_button.button(
        f"Lưu cho {selected_patient['name']}",
        use_container_width=True,
        type="primary"
    ):

        patient_records.add_measurement(
            selected_patient["id"],
            {
                "age": int(used.get("age", 0)),
                "height": float(used.get("height", 0)),
                "weight": float(used.get("weight", 0)),
                "bmi": float(used.get("BMI", 0)),
                "systolic_bp": float(used.get("ap_hi", 0)),
                "diastolic_bp": float(used.get("ap_lo", 0)),
                "heart_rate": float(used.get("heartRate", 0)),
                "glucose": float(used.get("Glucose", 0)),
                "total_cholesterol": float(used.get("totChol", 0)),
                "cardio_probability":
                    prediction.get("cardio", {}).get("probability"),
                "diabetes_probability":
                    prediction.get("diabetes", {}).get("probability"),
                "hypertension_probability":
                    prediction.get("hypertension", {}).get("probability"),
                "overall_risk": decision.get("overall_risk"),
                "priority_disease": decision.get("priority_disease"),
                "note": note or None,
            }
        )

        st.success(f"Đã lưu lần đo cho {selected_patient['name']}.")
        st.rerun()


# ============================================================
# 7. TAB ĐÁNH GIÁ
# ============================================================

def render_assessment_tab(selected_patient):
    """Tab nhập chỉ số và xem kết quả."""

    if selected_patient is not None:
        theme.note(
            f"Đang đánh giá cho: <b>{selected_patient['name']}</b>"
        )

    submitted, form = collect_inputs()

    if submitted:

        patient = build_patient(form)
        errors = validate(patient)

        if errors:
            for error in errors:
                st.error(error)

        else:
            try:
                assistant = HealthcareAssistant()
                st.session_state["assessment"] = assistant.assess(patient)
                st.session_state["assessment_inputs"] = patient

            except Exception as error:  # noqa: BLE001
                st.session_state["assessment"] = None
                st.error(f"Đánh giá thất bại: {error}")

    st.divider()

    if st.session_state["assessment"] is None:
        theme.empty_state(
            "🩺",
            "Chưa có kết quả",
            "Điền các chỉ số phía trên rồi bấm ĐÁNH GIÁ NGUY CƠ. "
            "Các ô để mặc định vẫn chạy được."
        )
        return

    render_results(st.session_state["assessment"])
    st.divider()
    render_save_block(selected_patient)


# ============================================================
# 8. TAB HỒ SƠ VÀ LỊCH SỬ
# ============================================================

def render_records_tab(selected_patient):
    """Tab danh sách bệnh nhân và lịch sử đo."""

    all_patients = patient_records.list_patients()

    st.markdown("#### 👥 Danh sách bệnh nhân")

    if not all_patients:
        theme.empty_state(
            "👤",
            "Chưa có bệnh nhân nào",
            "Dùng thanh bên để thêm bệnh nhân đầu tiên."
        )
        return

    st.dataframe(
        pd.DataFrame([
            {
                "Họ tên": p["name"],
                "Giới tính": p["gender"] or "—",
                "Năm sinh": p["birth_year"] or "—",
                "Số lần đo": p["measurement_count"],
                "Lần đo gần nhất": p["last_measured_at"] or "—",
                "Ghi chú": p["note"] or "",
            }
            for p in all_patients
        ]),
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    if selected_patient is None:
        st.info("Chọn một bệnh nhân ở thanh bên để xem lịch sử đo.")
        return

    st.markdown(f"#### 📊 Lịch sử đo — {selected_patient['name']}")

    measurements = patient_records.list_measurements(selected_patient["id"])

    if not measurements:
        theme.empty_state(
            "📈",
            "Bệnh nhân này chưa có lần đo nào",
            "Chạy một lần đánh giá rồi bấm “Lưu lần đo này”."
        )
        return

    history = pd.DataFrame(measurements)
    history["measured_at"] = pd.to_datetime(history["measured_at"])

    latest = history.iloc[-1]
    previous = history.iloc[-2] if len(history) > 1 else None

    def delta_of(column):
        """Mức thay đổi so với lần đo trước."""

        if previous is None:
            return None

        if pd.isna(latest[column]) or pd.isna(previous[column]):
            return None

        return round(float(latest[column]) - float(previous[column]), 1)

    metrics = st.columns(4)

    metrics[0].metric(
        "Huyết áp",
        f"{latest['systolic_bp']:.0f}/{latest['diastolic_bp']:.0f}",
        delta=delta_of("systolic_bp"),
        delta_color="inverse"
    )
    metrics[1].metric(
        "Đường huyết",
        f"{latest['glucose']:.0f}",
        delta=delta_of("glucose"),
        delta_color="inverse"
    )
    metrics[2].metric(
        "Cân nặng (kg)",
        f"{latest['weight']:.1f}",
        delta=delta_of("weight"),
        delta_color="off"
    )
    metrics[3].metric(
        "Nguy cơ tổng thể",
        risk_level_vi(latest["overall_risk"] or "N/A")
    )

    st.caption(
        f"{len(history)} lần đo, từ "
        f"{history['measured_at'].min():%d/%m/%Y %H:%M} đến "
        f"{history['measured_at'].max():%d/%m/%Y %H:%M}."
    )

    chart_data = history.set_index("measured_at")

    chart_tabs = st.tabs([
        "Huyết áp", "Đường huyết · cân nặng · nhịp tim", "Nguy cơ dự đoán"
    ])

    with chart_tabs[0]:
        st.line_chart(chart_data[["systolic_bp", "diastolic_bp"]])

    with chart_tabs[1]:
        st.line_chart(chart_data[["glucose", "weight", "heart_rate"]])

    with chart_tabs[2]:
        risk_columns = [
            "cardio_probability",
            "diabetes_probability",
            "hypertension_probability",
        ]
        if chart_data[risk_columns].notna().any().any():
            st.line_chart(chart_data[risk_columns])
        else:
            st.caption("Chưa có dữ liệu nguy cơ để vẽ.")

    with st.expander("📋 Bảng đầy đủ và xuất dữ liệu"):

        st.dataframe(
            history.drop(columns=["id"]),
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "⬇️ Tải lịch sử (CSV)",
            data=history.to_csv(index=False).encode("utf-8"),
            file_name=f"lich_su_do_{selected_patient['name']}.csv",
            mime="text/csv"
        )

    with st.expander("🗑️ Xoá một lần đo"):

        choice = st.selectbox(
            "Chọn lần đo",
            options=[m["id"] for m in measurements],
            format_func=lambda mid: next(
                m["measured_at"] for m in measurements if m["id"] == mid
            )
        )

        if st.button("Xoá lần đo"):
            patient_records.delete_measurement(choice)
            st.rerun()


# ============================================================
# 9. TAB HỎI ĐÁP
# ============================================================

def render_assistant_tab():
    """Tab hỏi đáp thông tin sức khỏe."""

    st.markdown("#### 💬 Hỏi đáp thông tin sức khỏe")

    theme.note(
        "Trợ lý này trả lời dựa trên một tập câu trả lời soạn sẵn, "
        "không phải mô hình ngôn ngữ. Nó giải thích khái niệm và kết quả "
        "đánh giá hiện tại, không chẩn đoán và không kê đơn."
    )

    st.write("")

    quick_questions = [
        "Kết quả hiện tại của tôi có ý nghĩa gì?",
        "Vì sao nguy cơ tổng thể của tôi ở mức CAO?",
        "Bệnh nào đang có nguy cơ cao nhất?",
        "Nguy cơ 70% có nghĩa là chắc chắn mắc bệnh không?",
        "Những yếu tố nào ảnh hưởng tới dự đoán?",
        "Huyết áp cao là gì?",
        "BMI là gì?",
        "Hệ thống đưa ra dự đoán bằng cách nào?",
    ]

    st.markdown("**Câu hỏi nhanh**")

    columns = st.columns(2)

    for index, quick in enumerate(quick_questions):
        if columns[index % 2].button(
            quick, key=f"quick_{index}", use_container_width=True
        ):
            st.session_state["ask_ai_question"] = quick
            st.session_state["ask_ai_auto_submit"] = True
            st.rerun()

    st.write("")

    question = st.text_input(
        "Câu hỏi của bạn",
        key="ask_ai_question",
        placeholder="Ví dụ: Vì sao nguy cơ tim mạch của tôi cao?"
    )

    asked = st.button("Gửi câu hỏi", type="primary")

    if asked or st.session_state.get("ask_ai_auto_submit", False):

        answer = sections.ask_ai_local(
            question,
            st.session_state.get("assessment")
        )

        st.markdown("**Trả lời**")
        st.info(answer)

        st.session_state["ask_ai_auto_submit"] = False


# ============================================================
# 10. TRANG CHÍNH
# ============================================================

theme.hero(
    "🩺 Trợ lý Sức khỏe AI",
    "Sàng lọc nguy cơ tim mạch, đái tháo đường và tăng huyết áp — "
    "hỗ trợ quyết định, không thay thế chẩn đoán."
)

selected_patient = render_sidebar()

tab_names = ["🩺 Đánh giá nguy cơ", "📈 Hồ sơ & lịch sử", "💬 Hỏi đáp"]

if SHOW_TRUSTWORTHY_AI:
    tab_names.append("🤖 Trustworthy AI")

if SHOW_DEFENSE_QA:
    tab_names.append("❓ Câu hỏi bảo vệ")

tabs = st.tabs(tab_names)

with tabs[0]:
    render_assessment_tab(selected_patient)

with tabs[1]:
    render_records_tab(selected_patient)

with tabs[2]:
    render_assistant_tab()

next_tab = 3

if SHOW_TRUSTWORTHY_AI:
    with tabs[next_tab]:
        sections.render_trustworthy_ai(PROJECT_ROOT)
    next_tab += 1

if SHOW_DEFENSE_QA:
    with tabs[next_tab]:
        sections.render_defense_qa()

st.divider()

st.caption(
    "Trợ lý Sức khỏe AI · Dự đoán → Phân mức nguy cơ → Khuyến nghị"
)
