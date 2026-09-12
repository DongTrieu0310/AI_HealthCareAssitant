"""
Các phần giao diện tách riêng khỏi màn hình chính.

- render_trustworthy_ai(): bảng đánh giá Trustworthy AI
- render_defense_qa(): câu hỏi và câu trả lời bảo vệ đồ án
- ask_ai_local(): trợ lý hỏi đáp dựa trên từ khoá (KHÔNG dùng mô hình ngôn ngữ)
"""

import streamlit as st

from ui.labels import risk_level_vi, disease_name_vi


# ============================================================
# TRUSTWORTHY AI
# ============================================================

def render_trustworthy_ai(project_root):
    """Hiển thị bảng đánh giá Trustworthy AI."""

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

        xai_dir = project_root / "data" / "models" / "xai"

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


# ============================================================
# CÂU HỎI BẢO VỆ
# ============================================================

def render_defense_qa():
    """Hiển thị danh sách câu hỏi phản biện và gợi ý trả lời."""

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


# ============================================================
# TRỢ LÝ HỎI ĐÁP (DÒ TỪ KHOÁ)
# ============================================================

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
