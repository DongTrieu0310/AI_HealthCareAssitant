"""
Kiểm thử: dữ liệu đưa vào mô hình phải được tiền xử lý đúng như lúc huấn luyện.

Đây là bài kiểm tra chống tái diễn (regression test) cho một lỗi đã từng xảy
ra: mô hình tim mạch và đái tháo đường được huấn luyện trên dữ liệu đã chuẩn
hoá, nhưng ứng dụng lại đưa vào giá trị thô. Khi đó xác suất gần như không đổi
giữa người khoẻ mạnh và người nguy cơ cao.

Chạy:
    python tests/test_prediction_scaling.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from prediction.prediction_layer import (  # noqa: E402
    apply_transformers,
    load_models,
    load_transformers,
)

PROCESSED = PROJECT_ROOT / "data" / "processed"
RAW = PROJECT_ROOT / "data" / "raw"

failures = []


def report(name, passed, detail=""):
    """In kết quả một phép kiểm tra."""

    mark = "PASS" if passed else "FAIL"
    print(f"[{mark}] {name}" + (f" — {detail}" if detail else ""))

    if not passed:
        failures.append(name)


# ============================================================
# 1. Transformer phải tồn tại
# ============================================================

try:
    transformers = load_transformers()
    report(
        "Nạp được scaler cho cardio và diabetes",
        {"cardio", "diabetes"} <= set(transformers)
    )
except FileNotFoundError as error:
    report("Nạp được scaler cho cardio và diabetes", False, str(error))
    transformers = {}


# ============================================================
# 2. Biến đổi lại từ dữ liệu thô phải khớp dữ liệu huấn luyện
# ============================================================

def check_matches_training(disease, raw_file, target, features, age_in_days):
    """So sánh dữ liệu thô đã biến đổi với tệp đã lưu lúc huấn luyện."""

    from sklearn.model_selection import train_test_split

    raw = pd.read_csv(RAW / raw_file)

    if raw.duplicated().sum() > 0:
        raw = raw.drop_duplicates()

    if age_in_days:
        raw["age"] = raw["age"] / 365.25

    X = raw[features].copy()
    y = raw[target].copy()

    _, X_test, _, _ = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    produced = apply_transformers(
        X_test.reset_index(drop=True), disease, transformers
    )

    expected = pd.read_csv(PROCESSED / f"{disease}_X_test.csv")

    difference = np.abs(
        produced.to_numpy() - expected.to_numpy()
    ).max()

    report(
        f"{disease}: dữ liệu thô sau biến đổi khớp tập kiểm thử",
        difference < 1e-8,
        f"sai lệch lớn nhất {difference:.2e}"
    )


if transformers:

    check_matches_training(
        "cardio",
        "cardio.csv",
        "cardio",
        ["age", "gender", "height", "weight", "ap_hi", "ap_lo",
         "cholesterol", "gluc", "smoke", "alco", "active"],
        age_in_days=True
    )

    check_matches_training(
        "diabetes",
        "diabetes.csv",
        "Outcome",
        ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
         "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"],
        age_in_days=False
    )


# ============================================================
# 3. Mô hình phải phân biệt được người khoẻ và người nguy cơ cao
# ============================================================

HEALTHY = dict(
    age=25, gender=1, height=170, weight=60, ap_hi=110, ap_lo=70,
    cholesterol=1, gluc=1, smoke=0, alco=0, active=1,
    Pregnancies=0, Glucose=85, BloodPressure=70, SkinThickness=20,
    Insulin=80, BMI=20.8, DiabetesPedigreeFunction=0.2, Age=25,
    male=0, currentSmoker=0, cigsPerDay=0, BPMeds=0, diabetes=0,
    totChol=170, sysBP=110, diaBP=70, heartRate=65, glucose=85,
)

AT_RISK = dict(
    age=70, gender=2, height=165, weight=95, ap_hi=185, ap_lo=115,
    cholesterol=3, gluc=3, smoke=1, alco=1, active=0,
    Pregnancies=8, Glucose=195, BloodPressure=115, SkinThickness=45,
    Insulin=300, BMI=34.9, DiabetesPedigreeFunction=1.5, Age=70,
    male=1, currentSmoker=1, cigsPerDay=30, BPMeds=1, diabetes=1,
    totChol=320, sysBP=185, diaBP=115, heartRate=95, glucose=195,
)

# Khoảng cách tối thiểu giữa hai ca — đủ để phát hiện mô hình "đứng yên".
MINIMUM_GAP = 0.20

if transformers:

    from application.healthcare_assistant import HealthcareAssistant

    assistant = HealthcareAssistant()

    healthy_result = assistant.predict(HEALTHY)
    at_risk_result = assistant.predict(AT_RISK)

    for disease in ["cardio", "diabetes", "hypertension"]:

        low = healthy_result[disease]["probability"]
        high = at_risk_result[disease]["probability"]

        report(
            f"{disease}: ca nguy cơ cao phải cao hơn ca khoẻ mạnh "
            f"ít nhất {MINIMUM_GAP:.0%}",
            (high - low) >= MINIMUM_GAP,
            f"khoẻ {low:.1%} → nguy cơ cao {high:.1%}"
        )

        report(
            f"{disease}: ca khoẻ mạnh không bị báo nguy cơ cao",
            low < 0.50,
            f"{low:.1%}"
        )


# ============================================================
# KẾT QUẢ
# ============================================================

print()

if failures:
    print(f"{len(failures)} phép kiểm tra THẤT BẠI:")
    for name in failures:
        print(" -", name)
    sys.exit(1)

print("Tất cả phép kiểm tra đều đạt.")
