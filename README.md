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