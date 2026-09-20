"""
Kiểm thử cho tầng đánh giá mô hình (src/evaluation/model_evaluation.py).

Nội dung:
    1. Kiểm chứng số học của từng chỉ số bằng giá trị tính tay.
    2. Kiểm chứng các trường hợp biên: dữ liệu rỗng, độ dài lệch nhau, chỉ có
       một lớp, xác suất không hợp lệ, tập dữ liệu rất nhỏ.
    3. Kiểm chứng phát hiện dương tính giả / âm tính giả và việc loại cột
       định danh khỏi bảng lỗi.
    4. Kiểm chứng hồi quy: tầng đánh giá dùng ĐÚNG bộ transformer và ĐÚNG
       ngưỡng của pipeline dự đoán đang chạy, và cho ra cùng xác suất như
       prediction_layer.predict_disease().
    5. Kiểm chứng artifact đã lưu trong reports/model_evaluation/ (nếu có).

Không huấn luyện, không fit lại, không ghi đè bất kỳ artifact nào của mô hình.

Chạy::

    python tests/test_model_evaluation.py
"""

import json
import sys
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from evaluation import model_evaluation as me  # noqa: E402
from prediction import prediction_layer  # noqa: E402


REPORTS_DIR = PROJECT_ROOT / "reports" / "model_evaluation"

failures = []


def report(name, passed, detail=""):
    """In kết quả một phép kiểm tra."""

    mark = "PASS" if passed else "FAIL"
    print(f"[{mark}] {name}" + (f" — {detail}" if detail else ""))

    if not passed:
        failures.append(name)


def close(first, second, tolerance=1e-9):
    """So sánh hai số với dung sai cho trước."""

    if first is None or second is None:
        return False

    return abs(float(first) - float(second)) <= tolerance


# ============================================================
# 1. SỐ HỌC CỦA CÁC CHỈ SỐ
# ============================================================


def check_classification_metrics():
    """Accuracy / Macro-F1 / Precision / Recall / F1 / ma trận nhầm lẫn."""

    y_true = [0, 1, 1, 0, 1, 0, 1, 1]
    y_pred = [0, 0, 1, 1, 1, 0, 1, 0]

    result = me.calculate_classification_metrics(
        y_true, y_pred, positive_label=1, threshold=0.4
    )

    # Giá trị tính tay: TP=3, FP=1, FN=2, TN=2
    report(
        "classification: accuracy = 5/8",
        close(result["accuracy"], 5 / 8),
        f"{result['accuracy']}",
    )
    report(
        "classification: ma trận nhầm lẫn [[2, 1], [2, 3]]",
        result["confusion_matrix"]["matrix"] == [[2, 1], [2, 3]],
        f"{result['confusion_matrix']['matrix']}",
    )
    report(
        "classification: TP/FP/FN/TN đúng",
        result["counts"]
        == {
            "true_negative": 2,
            "false_positive": 1,
            "false_negative": 2,
            "true_positive": 3,
        },
        f"{result['counts']}",
    )
    report(
        "classification: positive precision = 0.75",
        close(result["positive_precision"], 0.75),
    )
    report(
        "classification: positive recall = 0.6",
        close(result["positive_recall"], 0.6),
    )
    report(
        "classification: positive F1 = 2PR/(P+R)",
        close(result["positive_f1"], 2 * 0.75 * 0.6 / (0.75 + 0.6)),
    )
    report(
        "classification: macro-F1 = 0.6190476",
        close(result["macro_f1"], 0.6190476190476191),
        f"{result['macro_f1']}",
    )
    report(
        "classification: FPR = 1/3 và FNR = 2/5",
        close(result["rates"]["false_positive_rate"], 1 / 3)
        and close(result["rates"]["false_negative_rate"], 2 / 5),
    )
    report(
        "classification: chỉ số theo lớp đủ precision/recall/f1/support",
        len(result["per_class"]) == 2
        and all(
            {"precision", "recall", "f1", "support"} <= set(item)
            for item in result["per_class"].values()
        ),
    )
    report(
        "classification: vai trò lớp dương lấy từ tham số",
        result["per_class"]["class_1"]["role"] == "positive"
        and result["positive_label"] == 1,
    )

    flipped = me.calculate_classification_metrics(
        y_true, y_pred, positive_label=0
    )
    report(
        "classification: đổi lớp dương sang 0 thì chỉ số đổi theo",
        close(flipped["positive_precision"], 0.5)
        and close(flipped["positive_recall"], 2 / 3),
        f"precision={flipped['positive_precision']} "
        f"recall={flipped['positive_recall']}",
    )


def check_probability_metrics():
    """AUROC và Brier score (dùng xác suất, không dùng nhãn nhị phân)."""

    y_true = [0, 0, 1, 1]
    y_prob = [0.1, 0.2, 0.8, 0.9]

    result = me.calculate_probability_metrics(y_true, y_prob, positive_label=1)

    expected_brier = (
        (0.1 - 0.0) ** 2
        + (0.2 - 0.0) ** 2
        + (0.8 - 1.0) ** 2
        + (0.9 - 1.0) ** 2
    ) / 4

    report(
        "probability: AUROC = 1.0 khi thứ tự hoàn hảo",
        close(result["auroc"], 1.0),
        f"{result['auroc']}",
    )
    report(
        "probability: Brier score khớp công thức xác suất",
        close(result["brier_score"], expected_brier),
        f"{result['brier_score']} vs {expected_brier}",
    )
    report(
        "probability: Brier score KHÁC khi tính từ nhãn nhị phân",
        not close(result["brier_score"], 0.25),
        "0.025 (xác suất) vs 0.25 (nhãn 0/1)",
    )
    report(
        "probability: ghi rõ nguồn dùng xác suất lớp dương",
        result["uses"] == "positive-class predicted probability",
    )


def check_roc_curve():
    """Đường ROC phải lấy từ xác suất và nhất quán với AUROC."""

    rng = np.random.default_rng(7)
    y_true = rng.integers(0, 2, size=400)
    y_prob = np.clip(y_true * 0.35 + rng.random(400) * 0.65, 0.0, 1.0)

    result = me.calculate_roc_curve(y_true, y_prob, positive_label=1)

    fpr = result["fpr"]
    tpr = result["tpr"]

    report("roc: có điểm vẽ", result["n_points"] > 2, f"{result['n_points']}")
    report(
        "roc: FPR và TPR không giảm",
        all(b >= a for a, b in zip(fpr, fpr[1:]))
        and all(b >= a for a, b in zip(tpr, tpr[1:])),
    )
    report(
        "roc: AUROC khớp sklearn.metrics.roc_auc_score",
        close(result["auroc"], me.roc_auc_score(y_true, y_prob)),
        f"{result['auroc']}",
    )
    report(
        "roc: bắt đầu ở (0,0) và kết thúc ở (1,1)",
        close(fpr[0], 0.0)
        and close(tpr[0], 0.0)
        and close(fpr[-1], 1.0)
        and close(tpr[-1], 1.0),
    )


def check_calibration_metrics():
    """Reliability table: bin, số mẫu, tần suất quan sát, ECE."""

    # Hai bin tách biệt: 5 mẫu xác suất 0.1 (nhãn 0) và 5 mẫu 0.9 (nhãn 1).
    y_true = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    y_prob = [0.1] * 5 + [0.9] * 5

    result = me.calculate_calibration_metrics(
        y_true, y_prob, n_bins=10, positive_label=1
    )

    bins = result["bins"]

    report("calibration: 2 bin không rỗng", len(bins) == 2, f"{len(bins)} bin")
    report(
        "calibration: tổng số mẫu trong bin = số mẫu hợp lệ",
        sum(item["sample_count"] for item in bins) == 10,
        f"{[item['sample_count'] for item in bins]}",
    )
    report(
        "calibration: bin 1 (0.0-0.1) chứa 5 mẫu",
        bool(bins) and bins[0]["bin"] == 1 and bins[0]["sample_count"] == 5,
    )
    report(
        "calibration: bin 9 (0.8-0.9) chứa 5 mẫu",
        bool(bins) and bins[-1]["bin"] == 9 and bins[-1]["sample_count"] == 5,
    )
    report(
        "calibration: tần suất quan sát là 0.0 và 1.0",
        close(bins[0]["observed_positive_frequency"], 0.0)
        and close(bins[-1]["observed_positive_frequency"], 1.0),
    )
    report(
        "calibration: mean predicted nằm trong khoảng bin",
        all(
            item["bin_lower"]
            <= item["mean_predicted_probability"]
            <= item["bin_upper"]
            for item in bins
        ),
    )
    report(
        "calibration: ECE = gia quyền |gap| = 0.1",
        close(result["expected_calibration_error"], 0.1),
        f"{result['expected_calibration_error']}",
    )
    report(
        "calibration: có ghi chú nêu rõ không phải giá trị lâm sàng",
        bool(result["interpretation_note_vi"]),
    )


# ============================================================
# 2. PHÂN TÍCH LỖI
# ============================================================


def check_error_analysis():
    """Dương tính giả / âm tính giả và bảo vệ thông tin định danh."""

    features = pd.DataFrame(
        {
            "age": [40, 55, 61, 33],
            "BMI": [22.0, 31.5, 29.0, 24.0],
            "patient_name": ["A", "B", "C", "D"],
            "medical_record_id": [1, 2, 3, 4],
        }
    )

    y_true = [0, 1, 1, 0]
    y_prob = [0.62, 0.20, 0.71, 0.05]
    y_pred = [1, 0, 1, 0]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = me.analyze_prediction_errors(
            features, y_true, y_prob, y_pred, threshold=0.35
        )

    report(
        "error: FP = 1 và FN = 1",
        result["false_positive_count"] == 1
        and result["false_negative_count"] == 1,
        f"FP={result['false_positive_count']} FN={result['false_negative_count']}",
    )
    report(
        "error: TN = 1 và TP = 1",
        result["true_negative_count"] == 1
        and result["true_positive_count"] == 1,
    )
    report(
        "error: FP nằm ở dòng 0",
        result["false_positive_records"]["evaluation_row_index"].tolist() == [0],
        f"{result['false_positive_records']['evaluation_row_index'].tolist()}",
    )
    report(
        "error: FN nằm ở dòng 1",
        result["false_negative_records"]["evaluation_row_index"].tolist() == [1],
        f"{result['false_negative_records']['evaluation_row_index'].tolist()}",
    )
    report(
        "error: bảng lỗi giữ lại đặc trưng của mô hình",
        {"age", "BMI"} <= set(result["false_positive_records"].columns),
    )
    report(
        "error: phát hiện và loại cột định danh",
        set(result["dropped_identifier_columns"])
        == {"patient_name", "medical_record_id"},
        f"{result['dropped_identifier_columns']}",
    )
    report(
        "error: bảng lỗi KHÔNG còn cột định danh",
        "patient_name" not in result["false_negative_records"].columns
        and "medical_record_id"
        not in result["false_negative_records"].columns,
    )
    report(
        "error: có cảnh báo khi loại cột định danh (không im lặng)",
        any("Identifier-like" in str(item.message) for item in caught),
    )
    report(
        "error: FPR = 1/2 và FNR = 1/2",
        close(result["false_positive_rate"], 0.5)
        and close(result["false_negative_rate"], 0.5),
    )
    report(
        "error: FP + FN + TP + TN = số mẫu",
        (
            result["false_positive_count"]
            + result["false_negative_count"]
            + result["true_positive_count"]
            + result["true_negative_count"]
        )
        == 4,
    )
    report(
        "error: bảng lỗi có ngưỡng và mức nguy cơ của ứng dụng",
        {
            "model_threshold",
            "predicted_probability",
            "model_prediction",
            "app_risk_level",
        }
        <= set(result["false_positive_records"].columns),
    )
    report(
        "error: nhãn FN nêu rõ 'bỏ sót ca có nguy cơ'",
        "Bỏ sót" in result["labels_vi"]["false_negative"],
    )
    report(
        "error: có ghi chú sàng lọc thận trọng",
        bool(result["screening_note_vi"]),
    )


# ============================================================
# 3. TRƯỜNG HỢP BIÊN
# ============================================================


def check_empty_data():
    """Dữ liệu rỗng không được gây crash và phải báo rõ lý do."""

    classification = me.calculate_classification_metrics([], [])
    report(
        "edge/empty: classification báo không khả dụng",
        classification["available"] is False
        and "no samples" in classification["status"],
    )

    probability = me.calculate_probability_metrics([], [])
    report(
        "edge/empty: AUROC và Brier không khả dụng kèm lý do",
        probability["auroc"] is None
        and probability["brier_score"] is None
        and "no samples" in probability["auroc_status"]
        and "no samples" in probability["brier_score_status"],
    )

    calibration = me.calculate_calibration_metrics([], [])
    report(
        "edge/empty: calibration không khả dụng",
        calibration["available"] is False and calibration["bins"] == [],
    )

    roc = me.calculate_roc_curve([], [])
    report("edge/empty: ROC không khả dụng", roc["available"] is False)

    errors = me.analyze_prediction_errors(pd.DataFrame({"age": []}), [], [], [])
    report("edge/empty: phân tích lỗi không khả dụng", errors["available"] is False)


def check_length_mismatch():
    """Độ dài lệch nhau phải bị phát hiện."""

    classification = me.calculate_classification_metrics([0, 1], [0])
    report(
        "edge/mismatch: classification báo lệch độ dài",
        classification["available"] is False
        and "different lengths" in classification["status"],
    )

    probability = me.calculate_probability_metrics([0, 1], [0.5])
    report(
        "edge/mismatch: probability báo lệch độ dài",
        probability["available"] is False
        and "different lengths" in probability["status"],
    )

    errors = me.analyze_prediction_errors(
        pd.DataFrame({"age": [1, 2]}), [0, 1], [0.1, 0.9], [0]
    )
    report(
        "edge/mismatch: phân tích lỗi báo lệch độ dài",
        errors["available"] is False,
    )


def check_single_class():
    """Chỉ có một lớp: AUROC/ROC/reliability phải báo không khả dụng."""

    y_true = [1, 1, 1, 1]
    y_prob = [0.9, 0.8, 0.7, 0.6]

    probability = me.calculate_probability_metrics(y_true, y_prob)
    report(
        "edge/one-class: AUROC không khả dụng và không crash",
        probability["auroc"] is None
        and "only one class" in probability["auroc_status"],
        probability["auroc_status"][:60],
    )
    report(
        "edge/one-class: Brier vẫn tính được từ xác suất",
        probability["brier_score"] is not None,
    )

    roc = me.calculate_roc_curve(y_true, y_prob)
    report(
        "edge/one-class: ROC không khả dụng kèm lý do",
        roc["available"] is False and "only one class" in roc["status"],
    )

    calibration = me.calculate_calibration_metrics(y_true, y_prob)
    report(
        "edge/one-class: reliability không khả dụng kèm lý do",
        calibration["available"] is False
        and "only one class" in calibration["status"],
    )

    classification = me.calculate_classification_metrics(y_true, [1, 1, 1, 1])
    report(
        "edge/one-class: classification vẫn tính được (accuracy = 1.0)",
        classification["available"] and close(classification["accuracy"], 1.0),
    )


def check_invalid_probabilities():
    """Xác suất NaN / vô cực / ngoài [0, 1] phải được xử lý an toàn."""

    y_true = [0, 0, 1, 1, 1]
    y_prob = [0.1, 0.2, float("nan"), 2.0, 0.9]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        probability = me.calculate_probability_metrics(y_true, y_prob)

    report(
        "edge/invalid: đếm đúng số xác suất không hợp lệ",
        probability["n_invalid_probabilities"] == 2
        and probability["n_valid_probabilities"] == 3,
        f"invalid={probability['n_invalid_probabilities']} "
        f"valid={probability['n_valid_probabilities']}",
    )
    report(
        "edge/invalid: AUROC vẫn tính trên phần hợp lệ",
        close(probability["auroc"], 1.0),
        f"{probability['auroc']}",
    )
    report(
        "edge/invalid: có cảnh báo, không im lặng",
        any(
            "non-finite" in str(item.message) or "invalid" in str(item.message)
            for item in caught
        ),
    )
    report(
        "edge/invalid: trạng thái ghi rõ đã loại trừ",
        "exclusions" in probability["status"],
        probability["status"],
    )

    all_invalid = me.calculate_probability_metrics(
        [0, 1], [float("nan"), float("inf")]
    )
    report(
        "edge/invalid: toàn bộ không hợp lệ thì báo không khả dụng",
        all_invalid["available"] is False and "invalid" in all_invalid["status"],
    )


def check_tiny_dataset():
    """Tập dữ liệu rất nhỏ vẫn phải cho kết quả nhất quán."""

    classification = me.calculate_classification_metrics([0, 1], [0, 1])
    report(
        "edge/tiny: 2 mẫu vẫn tính được (accuracy = 1.0)",
        classification["available"] and close(classification["accuracy"], 1.0),
    )
    report(
        "edge/tiny: macro-F1 = 1.0 khi dự đoán hoàn hảo",
        close(classification["macro_f1"], 1.0),
    )

    single = me.calculate_classification_metrics([1], [1])
    report(
        "edge/tiny: 1 mẫu không crash",
        single["available"] and single["counts"]["true_positive"] == 1,
    )

    calibration = me.calculate_calibration_metrics([0, 1], [0.2, 0.8])
    report(
        "edge/tiny: reliability với 2 mẫu cho 2 bin",
        len(calibration["bins"]) == 2,
    )
    report(
        "edge/tiny: n_bins < 2 báo không khả dụng",
        me.calculate_calibration_metrics([0, 1], [0.2, 0.8], n_bins=1)[
            "available"
        ]
        is False,
    )


# ============================================================
# 4. NGƯỠNG VÀ ARTIFACT CỦA MÔ HÌNH
# ============================================================


def check_thresholds_loaded_from_artifacts():
    """Ngưỡng phải nạp từ *_threshold.pkl; không hardcode, không dùng 0.5."""

    thresholds = prediction_layer.load_thresholds()

    for disease in me.DISEASE_ORDER:

        config = me.get_disease_config(disease)
        artifacts = me.load_model_artifacts(disease)

        path = prediction_layer.THRESHOLD_PATHS[config.prediction_key]
        stored = float(joblib.load(path))

        report(
            f"threshold[{disease}]: bằng đúng giá trị trong {path.name}",
            close(artifacts["threshold"], stored),
            f"{artifacts['threshold']} vs {stored}",
        )
        report(
            f"threshold[{disease}]: khớp prediction_layer.load_thresholds()",
            close(
                artifacts["threshold"],
                float(thresholds[config.prediction_key]),
            ),
        )
        report(
            f"threshold[{disease}]: KHÔNG phải giá trị 0.5 mặc định",
            not close(artifacts["threshold"], 0.5),
            f"{artifacts['threshold']}",
        )
        report(
            f"threshold[{disease}]: ghi lại được tệp nguồn của ngưỡng",
            artifacts["threshold_path"].endswith(
                f"{config.prediction_key}_threshold.pkl"
            ),
        )
        report(
            f"artifact[{disease}]: lớp dương lấy từ model.classes_",
            artifacts["positive_label"] == int(artifacts["model"].classes_[-1]),
            f"classes={artifacts['classes']} → "
            f"positive={artifacts['positive_label']}",
        )
        report(
            f"artifact[{disease}]: số đặc trưng khớp model.n_features_in_",
            artifacts["n_features"] == int(artifacts["model"].n_features_in_),
            f"{artifacts['n_features']} vs {artifacts['model'].n_features_in_}",
        )


def check_no_refit_in_evaluation_module():
    """Kiểm tra tĩnh: tầng đánh giá không fit và không ghi đè artifact."""

    source = (SRC_DIR / "evaluation" / "model_evaluation.py").read_text(
        encoding="utf-8"
    )

    report(
        "static: không có lời gọi .fit( trong model_evaluation.py",
        ".fit(" not in source,
    )
    report(
        "static: không tạo mô hình mới (không RandomForestClassifier(...))",
        "RandomForestClassifier(" not in source
        and "LogisticRegression(" not in source,
    )
    report(
        "static: không lưu đè artifact bằng joblib.dump",
        "joblib.dump" not in source,
    )
    report(
        "static: không dùng đường dự đoán cũ (healthcare_predictor)",
        "healthcare_predictor" not in source,
    )


# ============================================================
# 5. HỒI QUY: DÙNG ĐÚNG PIPELINE DỰ ĐOÁN ĐANG CHẠY
# ============================================================


def _same_estimator(left: Any, right: Any) -> bool:
    """Same transformer object, or an equal re-load of the same artifact."""

    if left is right:
        return True

    if left is None or right is None:
        return False

    if type(left) is not type(right):
        return False

    try:
        left_params = left.get_params(deep=False)
        right_params = right.get_params(deep=False)
    except AttributeError:
        return False

    if set(left_params) != set(right_params):
        return False

    for key, left_value in left_params.items():
        right_value = right_params[key]
        try:
            left_array = np.asarray(left_value)
            right_array = np.asarray(right_value)
            if left_array.shape == () and right_array.shape == ():
                if left_array.dtype.kind in "fc" and right_array.dtype.kind in "fc":
                    if bool(np.isnan(left_array)) and bool(np.isnan(right_array)):
                        continue
                if bool(left_array == right_array):
                    continue
                return False
            if not np.array_equal(left_array, right_array, equal_nan=True):
                return False
        except Exception:  # noqa: BLE001 - uncomparable params are not equal
            return False

    for attribute in (
        "mean_",
        "var_",
        "scale_",
        "statistics_",
        "n_features_in_",
        "feature_names_in_",
    ):
        has_left = hasattr(left, attribute)
        has_right = hasattr(right, attribute)
        if has_left != has_right:
            return False
        if has_left and not np.array_equal(
            np.asarray(getattr(left, attribute)),
            np.asarray(getattr(right, attribute)),
        ):
            return False

    return True


def check_uses_live_transformers():
    """Tầng đánh giá phải dùng đúng transformer của pipeline dự đoán."""

    transformers = prediction_layer.load_transformers()

    for disease in me.DISEASE_ORDER:

        dataset = me.load_evaluation_dataset(disease)
        config = me.get_disease_config(disease)
        live_group = dict(transformers.get(config.prediction_key, {}))
        eval_group = dict(dataset["artifacts"]["transformers"])
        expected = bool(live_group)
        same_keys = set(eval_group) == set(live_group)
        same_objects = all(
            _same_estimator(live_group.get(key), eval_group.get(key))
            for key in set(live_group) | set(eval_group)
        )

        report(
            f"pipeline[{disease}]: dùng chung bộ transformer của prediction_layer",
            dataset["has_transformer"] == expected and same_keys and same_objects,
            f"has_transformer={dataset['has_transformer']}",
        )
        report(
            f"pipeline[{disease}]: tập đánh giá đã được kiểm chứng",
            dataset["verification"]["verified"] is True
            and dataset["verification"]["y_test_reproduced"] is True,
            "max_diff_vs_processed="
            f"{dataset['verification']['max_abs_diff_vs_processed']}",
        )

        if dataset["has_transformer"]:

            recomputed = prediction_layer.apply_transformers(
                dataset["raw_split"]["X_test"].copy(),
                config.prediction_key,
                transformers,
            )

            difference = float(
                np.max(
                    np.abs(
                        recomputed[dataset["features"]].to_numpy(dtype=float)
                        - dataset["X_model_input"].to_numpy(dtype=float)
                    )
                )
            )

            report(
                f"pipeline[{disease}]: dữ liệu vào mô hình = apply_transformers",
                difference <= 1e-8,
                f"max_diff={difference:.2e}",
            )

        else:

            comparison = (
                dataset["verification"]["non_missing_comparison"] or {}
            )

            report(
                f"pipeline[{disease}]: không cần transformer, khớp dữ liệu thô",
                comparison.get("observed_values_identical") is True,
                f"max_diff={comparison.get('max_abs_diff_where_observed')}",
            )


def check_matches_live_predict_disease():
    """Xác suất của tầng đánh giá phải trùng predict_disease() của ứng dụng."""

    for disease in me.DISEASE_ORDER:

        dataset = me.load_evaluation_dataset(disease)
        artifacts = dataset["artifacts"]
        config = me.get_disease_config(disease)
        threshold = float(artifacts["threshold"])

        batch = me.predict_for_evaluation(
            dataset["X_model_input"], disease, dataset["artifacts"]
        )

        n_rows = len(dataset["X_model_input"])
        step = max(1, n_rows // 60)
        indexes = list(range(0, n_rows, step))[:60]

        mismatches = []

        for index in indexes:

            single = prediction_layer.predict_disease(
                dataset["X_display"].iloc[[index]].reset_index(drop=True),
                config.prediction_key,
                {config.prediction_key: artifacts["model"]},
                {config.prediction_key: threshold},
                {config.prediction_key: dict(artifacts["transformers"])},
            )

            single_probability = float(single["probability"])
            batch_probability = float(batch[index])

            if not close(single_probability, batch_probability, 1e-12):
                mismatches.append(
                    (index, single_probability, batch_probability)
                )
            elif single["prediction"] != int(batch_probability >= threshold):
                mismatches.append((index, "nhãn", single["prediction"]))

        report(
            f"pipeline[{disease}]: xác suất trùng predict_disease() trên "
            f"{len(indexes)} dòng",
            not mismatches,
            f"{mismatches[:3]}",
        )
        report(
            f"pipeline[{disease}]: predict_labels = (probability >= threshold)",
            np.array_equal(
                me.predict_labels(batch, threshold),
                (batch >= threshold).astype(int),
            ),
        )
        report(
            f"pipeline[{disease}]: ngưỡng áp dụng đúng tại điểm biên",
            me.predict_labels(
                [threshold - 1e-9, threshold, threshold + 1e-9], threshold
            ).tolist()
            == [0, 1, 1],
            f"threshold={threshold}",
        )


# ============================================================
# 6. ARTIFACT ĐÃ LƯU
# ============================================================


def check_saved_artifacts():
    """Kiểm tra artifact trong reports/model_evaluation/ (bỏ qua nếu chưa có)."""

    if not REPORTS_DIR.exists():
        print(
            "[SKIP] artifact: chưa có reports/model_evaluation/ — chạy "
            "`python src/evaluation/run_evaluation.py` để sinh artifact."
        )
        return

    required_keys = {
        "model",
        "n_samples",
        "accuracy",
        "macro_f1",
        "auroc",
        "positive_precision",
        "positive_recall",
        "positive_f1",
        "brier_score",
        "false_positive_count",
        "false_negative_count",
        "per_class_metrics",
    }

    for disease in me.DISEASE_ORDER:

        metrics_path = REPORTS_DIR / f"{disease}_metrics.json"

        if not metrics_path.exists():
            report(
                f"artifact[{disease}]: có metrics JSON",
                False,
                str(metrics_path),
            )
            continue

        with metrics_path.open(encoding="utf-8") as handle:
            payload = json.load(handle)

        report(
            f"artifact[{disease}]: metrics JSON đủ khóa bắt buộc",
            required_keys <= set(payload),
            f"thiếu {sorted(required_keys - set(payload))}",
        )
        report(
            f"artifact[{disease}]: có confusion matrix CSV",
            (REPORTS_DIR / f"{disease}_confusion_matrix.csv").exists(),
        )
        report(
            f"artifact[{disease}]: có calibration CSV",
            (REPORTS_DIR / f"{disease}_calibration.csv").exists(),
        )
        report(
            f"artifact[{disease}]: có reliability diagram",
            (REPORTS_DIR / "plots" / f"{disease}_reliability.png").exists(),
        )
        report(
            f"artifact[{disease}]: có ROC curve",
            (REPORTS_DIR / "plots" / f"{disease}_roc.png").exists(),
        )
        report(
            f"artifact[{disease}]: ngưỡng trong JSON khớp *_threshold.pkl",
            close(
                payload.get("threshold"),
                float(
                    joblib.load(
                        prediction_layer.THRESHOLD_PATHS[
                            me.get_disease_config(disease).prediction_key
                        ]
                    )
                ),
            ),
        )

        for label, filename in (
            ("FP", f"{disease}_false_positives.csv"),
            ("FN", f"{disease}_false_negatives.csv"),
        ):

            csv_path = REPORTS_DIR / filename

            if not csv_path.exists():
                report(f"artifact[{disease}]: có bảng {label}", False)
                continue

            columns = [
                str(column).lower()
                for column in pd.read_csv(csv_path, nrows=1).columns
            ]

            report(
                f"artifact[{disease}]: bảng {label} không lộ định danh",
                not any(
                    pattern in column
                    for column in columns
                    for pattern in me.IDENTIFIER_PATTERNS
                ),
                f"{columns[:5]}",
            )
            report(
                f"artifact[{disease}]: bảng {label} có evaluation_row_index",
                "evaluation_row_index" in columns,
            )

    summary_path = REPORTS_DIR / "summary.json"

    if not summary_path.exists():
        report("artifact: có summary.json", False)
        return

    with summary_path.open(encoding="utf-8") as handle:
        summary = json.load(handle)

    consistent = True

    for disease, headline in (summary.get("headline_metrics") or {}).items():

        detail_path = REPORTS_DIR / f"{disease}_metrics.json"

        if not detail_path.exists():
            consistent = False
            continue

        with detail_path.open(encoding="utf-8") as handle:
            detail = json.load(handle)

        for key, value in headline.items():

            if value is None and detail.get(key) is None:
                continue

            if not close(value, detail.get(key)):
                consistent = False

    report("artifact: summary.json khớp metrics chi tiết", consistent)
    report(
        "artifact: summary ghi rõ không huấn luyện lại",
        summary.get("no_retraining") is True,
    )


# ============================================================
# 7. CHẠY TẤT CẢ
# ============================================================

CHECKS = [
    ("số học chỉ số phân loại", check_classification_metrics),
    ("AUROC / Brier score", check_probability_metrics),
    ("đường ROC", check_roc_curve),
    ("hiệu chuẩn / reliability", check_calibration_metrics),
    ("phân tích lỗi FP/FN", check_error_analysis),
    ("dữ liệu rỗng", check_empty_data),
    ("độ dài lệch nhau", check_length_mismatch),
    ("chỉ có một lớp", check_single_class),
    ("xác suất không hợp lệ", check_invalid_probabilities),
    ("tập dữ liệu rất nhỏ", check_tiny_dataset),
    ("ngưỡng và artifact mô hình", check_thresholds_loaded_from_artifacts),
    ("không fit lại (kiểm tra tĩnh)", check_no_refit_in_evaluation_module),
    ("dùng đúng transformer của ứng dụng", check_uses_live_transformers),
    ("trùng khớp predict_disease()", check_matches_live_predict_disease),
    ("artifact đã lưu", check_saved_artifacts),
]


def main():
    """Chạy toàn bộ phép kiểm tra và trả về mã thoát của tiến trình."""

    print("=" * 70)
    print("KIỂM THỬ TẦNG ĐÁNH GIÁ MÔ HÌNH")
    print("=" * 70)

    for name, check in CHECKS:

        print()
        print("-" * 70)
        print(name)
        print("-" * 70)

        try:
            check()
        except Exception as error:  # noqa: BLE001 - báo lỗi, không che giấu
            report(
                f"{name}: chạy được",
                False,
                f"{type(error).__name__}: {error}",
            )

    print()
    print("=" * 70)

    if failures:
        print(f"{len(failures)} PHÉP KIỂM TRA THẤT BẠI:")
        for name in failures:
            print(" -", name)
        print("=" * 70)
        return 1

    print("Tất cả phép kiểm tra đều đạt.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())