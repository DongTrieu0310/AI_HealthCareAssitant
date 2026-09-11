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

- **Sidebar → 👥 Patients**: add a patient, switch between patients, delete a
  patient (removes their measurements as well).
- **📊 Assessment Results → 💾 Save this measurement**: stores the current
  vitals (BP, glucose, weight, BMI, heart rate, cholesterol) together with the
  predicted probabilities and the overall risk level for the selected patient.
- **📈 Patient Records**:
  - *👥 Patient list*: every saved patient with the number of measurements and
    the date of the last one.
  - *📊 Measurement history*: latest values with the change since the previous
    visit, trend charts (blood pressure, glucose/weight/heart rate, predicted
    risk), the full measurement table, a CSV export, and per-measurement delete.

Storage API: `src/storage/patient_records.py`.

### Section visibility

`src/ui/app.py` defines two flags near the top:

```python
SHOW_TRUSTWORTHY_AI = False
SHOW_DEFENSE_QA = False
```

They hide the "🤖 Trustworthy AI" dashboard and the "❓ Defense Questions &
Answers" section. Set either to `True` to show that section again.
