# AI Healthcare Assistant

## 1. Project Overview

AI Healthcare Assistant is an AI-based healthcare risk assessment system designed to support users in evaluating potential health risks related to three common chronic diseases:

- Cardiovascular Disease
- Diabetes
- Hypertension

The system uses separate Machine Learning models for each disease and combines their predictions through a Decision Engine to generate an overall risk assessment.

The system is designed as a healthcare support tool and does not replace professional medical diagnosis or treatment.

---

## 2. Project Objectives

The main objectives of this project are:

- Build Machine Learning models for cardiovascular disease, diabetes, and hypertension risk assessment.
- Preprocess and analyze healthcare datasets.
- Evaluate and optimize the performance of the Machine Learning models.
- Optimize prediction thresholds for risk classification.
- Integrate multiple disease models into a unified healthcare system.
- Develop a Decision Engine for multi-disease risk assessment.
- Provide health-support recommendations based on the detected risk levels.
- Apply Trustworthy AI principles to the system.
- Provide an interactive Streamlit user interface.

---

## 3. System Architecture

The system follows a multi-model architecture combined with a Decision Engine.

```text
                         User
                          |
                          v
                  +---------------+
                  | Streamlit UI  |
                  +-------+-------+
                          |
                          v
                  +---------------+
                  | Application   |
                  | Layer         |
                  +-------+-------+
                          |
                          v
                  +---------------+
                  | Prediction    |
                  | Layer         |
                  +-------+-------+
                          |
          +---------------+---------------+
          |               |               |
          v               v               v
   +-------------+ +-------------+ +-------------+
   | Cardiovascular| |  Diabetes  | | Hypertension|
   | Random Forest| |Random Forest| |Random Forest|
   +-------------+ +-------------+ +-------------+
          |               |               |
          +---------------+---------------+
                          |
                          v
                  +---------------+
                  | Decision      |
                  | Engine        |
                  +-------+-------+
                          |
                          v
                  +---------------+
                  | Recommendation|
                  | Engine        |
                  +-------+-------+
                          |
                          v
                  +---------------+
                  | Final Health  |
                  | Assessment    |
                  +---------------+

    # ========================================================
    # TRUSK WORTHY
    # ========================================================
    from trustworthy_ai.trustworthy_ai_dashboard import render_trustworthy_ai_dashboard
    render_trustworthy_ai_dashboard(PROJECT_ROOT)



## Patient records & measurement tracking

The Streamlit UI stores patients and their repeated measurements in a local
SQLite database (`data/patient_records.db`, created automatically).

- **Thanh bên → 👥 Bệnh nhân**: add a patient, switch between patients, delete a
  patient (removes their measurements as well).
- **📊 Kết quả đánh giá → 💾 Lưu lần đo này**: stores the current
  vitals (BP, glucose, weight, BMI, heart rate, cholesterol) together with the
  predicted probabilities and the overall risk level for the selected patient.
- **📈 Hồ sơ bệnh nhân**:
  - *👥 Danh sách bệnh nhân*: every saved patient with the number of measurements and
    the date of the last one.
  - *📊 Lịch sử đo*: latest values with the change since the previous
    visit, trend charts (blood pressure, glucose/weight/heart rate, predicted
    risk), the full measurement table, a CSV export, and per-measurement delete.

Storage API: `src/storage/patient_records.py`.

### Section visibility

`src/ui/app.py` defines two flags near the top:

```python
SHOW_TRUSTWORTHY_AI = False
SHOW_DEFENSE_QA = False
```

They hide the "🤖 Trí tuệ nhân tạo đáng tin cậy" dashboard and the
"❓ Câu hỏi và câu trả lời bảo vệ" section. Set either to `True` to show that section again.

## Cấu trúc giao diện

```
src/ui/app.py        màn hình chính: thanh bên + 3 tab (Đánh giá / Hồ sơ / Hỏi đáp)
src/ui/theme.py      CSS dùng chung và các thành phần hiển thị (thẻ nguy cơ, huy hiệu…)
src/ui/labels.py     nhãn tiếng Việt và màu theo mức nguy cơ
src/ui/sections.py   phần Trustworthy AI, câu hỏi bảo vệ, trợ lý hỏi đáp
.streamlit/config.toml  màu chủ đạo của ứng dụng
```

Form nhập liệu dùng `st.form`: ứng dụng chỉ chạy lại khi bấm nút đánh giá, nên
không bị giật và không mất kết quả khi đang chỉnh số liệu.

Các trường trùng nhau giữa ba mô hình (tuổi, giới tính, BMI, huyết áp, đường
huyết) chỉ phải nhập một lần và được suy ra trong `build_patient()`. BMI mặc
định tính từ chiều cao và cân nặng; muốn nhập tay thì mở mục "Chỉ số chuyên
sâu".

## Ngôn ngữ giao diện

Toàn bộ phần hiển thị của ứng dụng Streamlit (nhãn form, thông báo lỗi, kết quả
đánh giá, khuyến nghị, trợ lý Hỏi AI, hồ sơ bệnh nhân) đã được chuyển sang
tiếng Việt. Các khoá dùng trong logic (`LOW` / `MODERATE` / `HIGH`, khoá bệnh
`cardio` / `diabetes` / `hypertension`, tên cột trong CSDL) vẫn giữ nguyên tiếng
Anh; `src/ui/app.py` chỉ dịch ở lớp hiển thị qua `risk_level_vi()` và
`disease_name_vi()`.
