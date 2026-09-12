"""
Xuất lại các bộ biến đổi (transformer) đã dùng khi huấn luyện.

Vì sao cần script này
---------------------
`cardio_preprocessing.py` và `diabetes_preprocessing.py` có chuẩn hoá dữ liệu
bằng StandardScaler nhưng **không lưu scaler lại**. Hệ quả: khi chạy ứng dụng,
giá trị thô (huyết áp 140, chiều cao 170…) được đưa thẳng vào mô hình vốn được
huấn luyện trên dữ liệu đã chuẩn hoá (trung bình 0, độ lệch chuẩn 1) — dự đoán
gần như không thay đổi theo bệnh nhân.

Script này lặp lại đúng các bước tiền xử lý (cùng random_state, cùng thứ tự)
để dựng lại scaler và imputer, rồi lưu ra `data/models/`. Sau khi lưu, script
tự kiểm chứng: biến đổi lại dữ liệu thô và so với tệp `data/processed/*_X_test.csv`
đã sinh ra lúc huấn luyện. Nếu hai bên không khớp, script báo lỗi và không ghi đè.

Chạy:
    python src/preprocessing/export_transformers.py
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW = PROJECT_ROOT / "data" / "raw"
PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS = PROJECT_ROOT / "data" / "models"

TOLERANCE = 1e-8


CARDIO_FEATURES = [
    "age", "gender", "height", "weight", "ap_hi", "ap_lo",
    "cholesterol", "gluc", "smoke", "alco", "active",
]

DIABETES_FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
]

# Hai giá trị Insulin bị coi là bất thường trong bước tiền xử lý gốc.
SUSPICIOUS_INSULIN = [102.5, 169.5]


def check(name, produced, expected_path):
    """So sánh dữ liệu dựng lại với tệp đã sinh lúc huấn luyện."""

    expected = pd.read_csv(expected_path)

    if list(produced.columns) != list(expected.columns):
        raise SystemExit(
            f"[{name}] Thứ tự cột không khớp:\n"
            f"  dựng lại: {list(produced.columns)}\n"
            f"  mong đợi: {list(expected.columns)}"
        )

    if produced.shape != expected.shape:
        raise SystemExit(
            f"[{name}] Kích thước không khớp: "
            f"{produced.shape} vs {expected.shape}"
        )

    difference = np.abs(
        produced.to_numpy() - expected.to_numpy()
    ).max()

    if difference > TOLERANCE:
        raise SystemExit(
            f"[{name}] Dữ liệu dựng lại KHÔNG khớp với tệp đã lưu "
            f"(sai lệch lớn nhất {difference:.2e}). Không ghi transformer."
        )

    print(f"[{name}] Khớp với dữ liệu huấn luyện (sai lệch {difference:.2e}).")


# ============================================================
# CARDIO
# ============================================================

def export_cardio():
    """Dựng lại và lưu scaler của mô hình tim mạch."""

    cardio = pd.read_csv(RAW / "cardio.csv")

    if cardio.duplicated().sum() > 0:
        cardio = cardio.drop_duplicates()

    # Bộ dữ liệu gốc lưu tuổi theo ngày.
    cardio["age"] = cardio["age"] / 365.25

    X = cardio[CARDIO_FEATURES].copy()
    y = cardio["cardio"].copy()

    X_train, X_test, _, _ = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    scaler.fit(X_train)

    produced = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns
    )

    check("cardio", produced, PROCESSED / "cardio_X_test.csv")

    joblib.dump(scaler, MODELS / "cardio_scaler.pkl")
    print("[cardio] Đã lưu cardio_scaler.pkl")


# ============================================================
# DIABETES
# ============================================================

def export_diabetes():
    """Dựng lại và lưu imputer + scaler của mô hình đái tháo đường."""

    diabetes = pd.read_csv(RAW / "diabetes.csv")

    X = diabetes[DIABETES_FEATURES].copy()
    y = diabetes["Outcome"].copy()

    X_train, X_test, _, _ = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    for frame in (X_train, X_test):
        frame["Insulin"] = frame["Insulin"].replace(
            SUSPICIOUS_INSULIN, float("nan")
        )

    imputer = SimpleImputer(strategy="median")
    imputer.fit(X_train)

    X_train_imputed = pd.DataFrame(
        imputer.transform(X_train), columns=X_train.columns
    )
    X_test_imputed = pd.DataFrame(
        imputer.transform(X_test), columns=X_test.columns
    )

    scaler = StandardScaler()
    scaler.fit(X_train_imputed)

    produced = pd.DataFrame(
        scaler.transform(X_test_imputed),
        columns=X_test_imputed.columns
    )

    check("diabetes", produced, PROCESSED / "diabetes_X_test.csv")

    joblib.dump(imputer, MODELS / "diabetes_imputer.pkl")
    joblib.dump(scaler, MODELS / "diabetes_scaler.pkl")
    print("[diabetes] Đã lưu diabetes_imputer.pkl và diabetes_scaler.pkl")


# ============================================================
# HYPERTENSION
# ============================================================

def check_hypertension():
    """Mô hình tăng huyết áp dùng đơn vị gốc — xác nhận là không cần scaler."""

    X_test = pd.read_csv(PROCESSED / "hypertension_X_test.csv")

    if X_test["sysBP"].max() < 10:
        raise SystemExit(
            "[hypertension] Dữ liệu trông như đã chuẩn hoá — cần thêm scaler."
        )

    print(
        "[hypertension] Dữ liệu giữ nguyên đơn vị gốc "
        f"(sysBP {X_test['sysBP'].min():.0f}–{X_test['sysBP'].max():.0f}), "
        "không cần scaler."
    )


if __name__ == "__main__":

    print("=" * 60)
    print("XUẤT LẠI TRANSFORMER TỪ DỮ LIỆU HUẤN LUYỆN")
    print("=" * 60)

    export_cardio()
    export_diabetes()
    check_hypertension()

    print("\nHoàn tất.")
