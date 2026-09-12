"""
Phát hiện thiên lệch trong tập dữ liệu — bảy phép kiểm tra chạy TRƯỚC khi huấn luyện.

Bảy phương pháp, xếp theo thứ tự nên chạy:

    1. Đại diện nhóm        — nhóm nào bị thiếu trong dữ liệu?
    2. Tỉ lệ nhãn theo nhóm — nhóm nào có tỉ lệ mắc bệnh khác hẳn? (chi-square)
    3. Giá trị lặp bất thường — dấu hiệu dữ liệu đã bị điền sẵn hoặc làm tròn
    4. Thăm dò rò rỉ nhãn   — một biến đơn lẻ có dự đoán nhãn quá tốt không?
    5. Adversarial validation — tập huấn luyện và kiểm thử có cùng phân bố không?
    6. Biến thay thế (proxy) — bỏ cột nhạy cảm có thật sự làm mô hình "mù" không?
    7. Giá trị ngoài khoảng sinh lý — dữ liệu bẩn giả dạng dữ liệu thật

Chạy:
    python src/data/bias_detection.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW = PROJECT_ROOT / "data" / "raw"

# Ngưỡng cảnh báo — chọn theo kinh nghiệm, không phải hằng số khoa học.
DOMINANT_VALUE_ALERT = 0.10   # một giá trị chiếm >10% của biến liên tục
SINGLE_FEATURE_AUC_ALERT = 0.75   # một biến đơn lẻ dự đoán nhãn quá tốt
ADVERSARIAL_AUC_ALERT = 0.55   # phân biệt được train/test
PROXY_AUC_ALERT = 0.70   # đoán được nhóm nhạy cảm từ các biến khác


def title(text):
    """In tiêu đề một mục."""

    print("\n" + "=" * 68)
    print(text)
    print("=" * 68)


def forest_auc(X, y, folds=3):
    """AUC trung bình của Random Forest qua cross-validation."""

    return cross_val_score(
        RandomForestClassifier(n_estimators=60, random_state=0),
        X, y, cv=folds, scoring="roc_auc"
    ).mean()


# ============================================================
# 1. ĐẠI DIỆN NHÓM
# ============================================================

def check_representation(df, name, group_columns, age_column=None):
    """Nhóm nào chiếm bao nhiêu phần trăm dữ liệu?

    Phát hiện: nhóm bị thiếu hoặc chiếm quá ít để mô hình học được.
    """

    print(f"\n{name}")

    for column in group_columns:
        share = (df[column].value_counts(normalize=True) * 100).round(1)
        print(f"  {column}: " + ", ".join(
            f"{index} → {value}%" for index, value in share.items()
        ))

        if share.min() < 20:
            print(f"    ⚠ nhóm nhỏ nhất chỉ {share.min()}% — mô hình học ít về nhóm này")

    if age_column is not None:
        ages = df[age_column]
        print(
            f"  {age_column}: {ages.min():.0f}–{ages.max():.0f} tuổi, "
            f"trung bình {ages.mean():.1f}"
        )
        print(
            "    ⚠ ngoài khoảng này mô hình chưa từng thấy dữ liệu — "
            "dự đoán là ngoại suy"
        )


# ============================================================
# 2. TỈ LỆ NHÃN THEO NHÓM
# ============================================================

def check_label_rate(df, name, group_column, label_column):
    """Tỉ lệ mắc bệnh có khác nhau giữa các nhóm không? (kiểm định chi-square)

    Phát hiện: nhãn phụ thuộc vào nhóm — mô hình sẽ học theo và khuếch đại.
    Lưu ý: với n lớn, chênh lệch rất nhỏ cũng cho p < 0,05, nên phải nhìn
    cả độ lớn chênh lệch chứ không chỉ nhìn p.
    """

    table = pd.crosstab(df[group_column], df[label_column])
    chi2, p_value, _, _ = chi2_contingency(table)
    rate = df.groupby(group_column, observed=True)[label_column].mean() * 100

    print(f"\n{name} — tỉ lệ mắc theo {group_column}")

    for index, value in rate.round(1).items():
        print(f"  {index}: {value}%")

    gap = rate.max() - rate.min()

    print(
        f"  chi2 = {chi2:.1f}, p = {p_value:.2e}, "
        f"chênh lệch lớn nhất = {gap:.1f} điểm phần trăm"
    )

    if p_value < 0.05 and gap >= 5:
        print("    ⚠ khác biệt vừa có ý nghĩa thống kê vừa đủ lớn để đáng lo")
    elif p_value < 0.05:
        print("    → có ý nghĩa thống kê nhưng chênh lệch nhỏ (do cỡ mẫu lớn)")


# ============================================================
# 3. GIÁ TRỊ LẶP BẤT THƯỜNG
# ============================================================

def check_dominant_values(df, name, columns):
    """Một giá trị cụ thể có chiếm tỉ lệ bất thường trong biến liên tục không?

    Phát hiện: giá trị thiếu đã bị điền bằng hằng số, hoặc thói quen làm tròn
    khi đo (ví dụ huyết áp luôn ghi 120/80).
    """

    print(f"\n{name}")
    found = False

    for column in columns:

        if df[column].nunique() <= 5:
            continue

        counts = df[column].value_counts(normalize=True)

        if counts.iloc[0] > DOMINANT_VALUE_ALERT:
            found = True
            print(
                f"  ⚠ {column}: giá trị {counts.index[0]} chiếm "
                f"{counts.iloc[0] * 100:.1f}% "
                f"(biến có {df[column].nunique()} giá trị khác nhau)"
            )

    if not found:
        print("  Không có giá trị nào chiếm tỉ lệ bất thường.")


def check_value_vs_label(df, name, column, values, label_column):
    """Những giá trị nghi ngờ đó gắn với nhãn thế nào?

    Nếu một giá trị cho tỉ lệ mắc 0% hoặc 100% thì gần như chắc chắn nó được
    điền dựa trên nhãn — đó là rò rỉ nhãn.
    """

    print(f"\n{name} — {column}")
    overall = df[label_column].mean() * 100

    for value in values:
        subset = df[df[column] == value]
        if len(subset) == 0:
            continue
        rate = subset[label_column].mean() * 100
        alert = " ⚠ RÒ RỈ NHÃN" if rate < 5 or rate > 95 else ""
        print(f"  {column} = {value}: n = {len(subset):4d}, tỉ lệ mắc = {rate:5.1f}%{alert}")

    print(f"  (tỉ lệ mắc chung của toàn bộ dữ liệu: {overall:.1f}%)")


# ============================================================
# 4. THĂM DÒ RÒ RỈ NHÃN
# ============================================================

def check_single_feature_leakage(df, name, features, label_column):
    """Huấn luyện một mô hình cho MỖI biến riêng lẻ.

    Phát hiện: biến nào một mình đã dự đoán nhãn quá tốt thì đáng ngờ —
    hoặc nó là yếu tố quyết định thật sự, hoặc nó đang rò rỉ nhãn.
    """

    print(f"\n{name} — AUC khi chỉ dùng một biến")

    scores = []

    for feature in features:
        auc = forest_auc(df[[feature]], df[label_column], folds=5)
        scores.append((feature, auc))

    for feature, auc in sorted(scores, key=lambda item: -item[1]):
        alert = "  ⚠ NGHI NGỜ" if auc > SINGLE_FEATURE_AUC_ALERT else ""
        print(f"  {feature:26s} AUC = {auc:.3f}{alert}")


# ============================================================
# 5. ADVERSARIAL VALIDATION
# ============================================================

def check_train_test_shift(df, name, features, label_column):
    """Huấn luyện một mô hình để phân biệt tập train và tập test.

    Phát hiện: nếu phân biệt được (AUC cao) thì hai tập lệch phân bố, kết quả
    đánh giá trên tập test sẽ không phản ánh đúng năng lực mô hình.
    AUC quanh 0,5 nghĩa là không phân biệt được — đó là điều mong muốn.
    """

    X_train, X_test = train_test_split(
        df[features], test_size=0.20, random_state=42,
        stratify=df[label_column]
    )

    X = pd.concat([X_train, X_test])
    y = np.r_[np.zeros(len(X_train)), np.ones(len(X_test))]

    auc = forest_auc(X, y)

    verdict = (
        "LỆCH PHÂN BỐ" if auc > ADVERSARIAL_AUC_ALERT
        else "đạt — hai tập cùng phân bố"
    )

    print(f"\n{name}: AUC phân biệt train/test = {auc:.3f} → {verdict}")


# ============================================================
# 6. BIẾN THAY THẾ (PROXY)
# ============================================================

def check_proxy(df, name, features, sensitive_column):
    """Thử đoán thuộc tính nhạy cảm từ các biến còn lại.

    Phát hiện: nếu đoán được, việc bỏ cột nhạy cảm ra khỏi mô hình KHÔNG làm
    mô hình mù với thuộc tính đó — nó vẫn phân biệt được qua biến thay thế.
    """

    others = [feature for feature in features if feature != sensitive_column]

    auc = forest_auc(df[others], df[sensitive_column])

    model = RandomForestClassifier(
        n_estimators=60, random_state=0
    ).fit(df[others], df[sensitive_column])

    top = sorted(
        zip(others, model.feature_importances_),
        key=lambda item: -item[1]
    )[:3]

    print(f"\n{name}: AUC đoán '{sensitive_column}' từ các biến khác = {auc:.3f}")

    if auc > PROXY_AUC_ALERT:
        print(
            f"  ⚠ '{sensitive_column}' bị mã hoá ngầm qua: "
            + ", ".join(f"{feature} ({value:.2f})" for feature, value in top)
        )
        print("    → bỏ cột này ra khỏi mô hình không đủ để đạt công bằng")


# ============================================================
# 7. GIÁ TRỊ NGOÀI KHOẢNG SINH LÝ
# ============================================================

def check_physiological_range(df, name, rules, extra_checks=None):
    """Đối chiếu từng biến với khoảng giá trị hợp lý về mặt sinh lý.

    Phát hiện: dữ liệu bẩn mà thống kê mô tả thông thường dễ bỏ qua.
    """

    print(f"\n{name}")

    for column, (low, high) in rules.items():
        bad = ((df[column] < low) | (df[column] > high)).sum()
        mark = " ⚠" if bad else ""
        print(
            f"  {column:10s} ngoài [{low}, {high}]: {bad:5d} dòng "
            f"({bad / len(df) * 100:.2f}%) — thực tế {df[column].min()} … "
            f"{df[column].max()}{mark}"
        )

    for label, mask in (extra_checks or {}).items():
        count = mask.sum()
        mark = " ⚠" if count else ""
        print(f"  {label}: {count} dòng{mark}")


# ============================================================
# CHẠY TRÊN BA BỘ DỮ LIỆU CỦA DỰ ÁN
# ============================================================

def main():
    """Chạy toàn bộ bảy phép kiểm tra."""

    cardio = pd.read_csv(RAW / "cardio.csv")
    cardio["age"] = cardio["age"] / 365.25

    diabetes = pd.read_csv(RAW / "diabetes.csv")
    hypertension = pd.read_csv(RAW / "hypertension.csv")

    cardio_features = [
        "age", "gender", "height", "weight", "ap_hi", "ap_lo",
        "cholesterol", "gluc", "smoke", "alco", "active",
    ]
    diabetes_features = [
        column for column in diabetes.columns if column != "Outcome"
    ]

    title("1. ĐẠI DIỆN NHÓM — nhóm nào bị thiếu?")
    check_representation(cardio, "CARDIO", ["gender"], "age")
    check_representation(hypertension, "HYPERTENSION", ["male"], "age")
    check_representation(diabetes, "DIABETES (Pima: chỉ nữ)", [], "Age")

    title("2. TỈ LỆ NHÃN THEO NHÓM — nhãn có phụ thuộc nhóm không?")
    check_label_rate(cardio, "CARDIO", "gender", "cardio")

    cardio = cardio.assign(nhom_tuoi=pd.cut(
        cardio["age"], [0, 45, 55, 65, 120],
        labels=["<45", "45-55", "55-65", "65+"]
    ))
    check_label_rate(cardio, "CARDIO", "nhom_tuoi", "cardio")
    check_label_rate(hypertension, "HYPERTENSION", "male", "Risk")

    title("3. GIÁ TRỊ LẶP BẤT THƯỜNG — dữ liệu có bị điền sẵn không?")
    check_dominant_values(diabetes, "DIABETES", diabetes_features)
    check_dominant_values(
        cardio, "CARDIO", ["age", "height", "weight", "ap_hi", "ap_lo"]
    )
    check_value_vs_label(
        diabetes, "DIABETES", "Insulin", [102.5, 169.5], "Outcome"
    )
    check_value_vs_label(
        diabetes, "DIABETES", "SkinThickness", [27, 32], "Outcome"
    )

    title("4. THĂM DÒ RÒ RỈ NHÃN — một biến có dự đoán quá tốt không?")
    check_single_feature_leakage(
        diabetes, "DIABETES", diabetes_features, "Outcome"
    )

    title("5. ADVERSARIAL VALIDATION — train và test có cùng phân bố?")
    check_train_test_shift(cardio, "CARDIO", cardio_features, "cardio")

    title("6. BIẾN THAY THẾ — bỏ cột nhạy cảm có đủ không?")
    check_proxy(cardio, "CARDIO", cardio_features, "gender")

    title("7. GIÁ TRỊ NGOÀI KHOẢNG SINH LÝ")
    check_physiological_range(
        cardio,
        "CARDIO",
        {
            "ap_hi": (60, 250),
            "ap_lo": (40, 150),
            "height": (120, 220),
            "weight": (30, 250),
        },
        {"huyết áp tâm thu ≤ tâm trương (vô lý)": cardio["ap_hi"] <= cardio["ap_lo"]}
    )

    print("\n" + "=" * 68)
    print("Hoàn tất. Mọi dòng có dấu ⚠ đều cần xem xét trước khi huấn luyện.")
    print("=" * 68)


if __name__ == "__main__":
    main()
