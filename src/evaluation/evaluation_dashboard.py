"""
Bảng đánh giá mô hình / Trustworthy AI (Streamlit).

Màn hình này ĐỌC các artifact đã lưu trong ``reports/model_evaluation/`` nên
không tính lại chỉ số trên mỗi lần Streamlit chạy lại. Chỉ khi người dùng bấm
nút "Chạy lại đánh giá" thì ``src/evaluation/model_evaluation.py`` mới được gọi
(đọc mô hình đóng băng, không huấn luyện, không ghi đè artifact của mô hình).

Bật/tắt màn hình bằng cờ ``SHOW_MODEL_EVALUATION`` trong ``src/ui/app.py``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from evaluation.model_evaluation import (  # noqa: E402
    DISEASE_ORDER,
    EVALUATION_DISCLAIMER_VI,
    PLOTS_DIRNAME,
    get_disease_config,
    run_evaluation,
)

# Reuse the existing Trustworthy AI helpers instead of writing a second loader.
from trustworthy_ai.trustworthy_ai_dashboard import (  # noqa: E402
    get_trustworthy_ai_dirs,
    load_csv_files,
)


# ============================================================
# 1. LOADING (cached)
# ============================================================


@st.cache_data(show_spinner=False)
def load_metrics_json(
    reports_dir: str,
    disease: str,
    modified: float,
) -> Optional[Dict[str, Any]]:
    """Read one persisted metrics JSON (invalidated by file mtime)."""

    path = Path(reports_dir) / f"{disease}_metrics.json"

    if not path.exists():
        return None

    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


@st.cache_data(show_spinner=False)
def load_table(path: str, modified: float) -> Optional[pd.DataFrame]:
    """Read one persisted CSV artifact (invalidated by file mtime)."""

    target = Path(path)

    if not target.exists():
        return None

    return pd.read_csv(target)


def file_mtime(path: Path) -> float:
    """Return a file's modification time, or 0.0 when it does not exist."""

    return path.stat().st_mtime if path.exists() else 0.0


def get_metrics(
    reports_dir: Path,
    disease: str,
) -> Optional[Dict[str, Any]]:
    """Load the metrics of one disease, or None when not generated yet."""

    return load_metrics_json(
        str(reports_dir),
        disease,
        file_mtime(reports_dir / f"{disease}_metrics.json"),
    )


# ============================================================
# 2. FORMATTERS
# ============================================================


def fmt(value: Optional[float], digits: int = 4) -> str:
    """Format a metric for display."""

    if value is None:
        return "n/a"

    return f"{value:.{digits}f}"


def fmt_share(value: Optional[float]) -> str:
    """Format a share as a percentage."""

    if value is None:
        return "n/a"

    return f"{value * 100:.1f}%"


def status_note(payload: Dict[str, Any], key: str, status_key: str) -> None:
    """Show an explicit unavailable-status message when a metric is missing."""

    if payload.get(key) is None:
        st.warning(f"Không tính được chỉ số này: {payload.get(status_key)}")


# ============================================================
# 3. SECTION RENDERERS
# ============================================================


def render_overview(results: Dict[str, Dict[str, Any]]) -> None:
    """Model comparison table built from the persisted metrics."""

    st.subheader("Tổng quan các mô hình")

    rows: List[Dict[str, Any]] = []

    for disease in DISEASE_ORDER:

        payload = results.get(disease)

        if not payload:
            continue

        rows.append(
            {
                "Mô hình": (
                    f"{payload['display_name_vi']} "
                    f"({payload['display_name_en']})"
                ),
                "File": payload["model_file"],
                "Ngưỡng": payload["threshold"],
                "Dòng kiểm thử": payload["n_samples"],
                "Accuracy": fmt(payload["accuracy"]),
                "Macro-F1": fmt(payload["macro_f1"]),
                "AUROC": fmt(payload["auroc"]),
                "Brier": fmt(payload["brier_score"]),
                "Recall dương": fmt(payload["positive_recall"]),
                "FP": payload["false_positive_count"],
                "FN": payload["false_negative_count"],
            }
        )

    if not rows:
        st.info(
            "Chưa có artifact đánh giá. Hãy chạy "
            "`python src/evaluation/run_evaluation.py` hoặc bấm nút chạy lại."
        )
        return

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.caption(
        "AUROC và Brier dùng XÁC SUẤT dự đoán; Accuracy, Macro-F1, FP, FN dùng "
        "NGƯỠNG mô hình lấy từ data/models/*_threshold.pkl."
    )


def render_performance(payload: Dict[str, Any]) -> None:
    """Headline metrics of one model."""

    st.subheader("Hiệu năng (theo ngưỡng mô hình)")
    st.caption(
        "Accuracy · Macro-F1 · Precision · Recall · F1 · AUROC · Brier Score · "
        "Positive Precision · Positive Recall · Positive F1"
    )

    st.caption(
        f"Ngưỡng = {payload['threshold']} "
        f"(nguồn: {Path(payload['threshold_source']).name})"
    )

    first_row = st.columns(4)
    first_row[0].metric("Accuracy", fmt(payload["accuracy"]))
    first_row[1].metric("Macro-F1", fmt(payload["macro_f1"]))
    first_row[2].metric("AUROC (xác suất)", fmt(payload["auroc"]))
    first_row[3].metric("Brier score", fmt(payload["brier_score"]))

    second_row = st.columns(4)
    second_row[0].metric("Positive Recall", fmt(payload["positive_recall"]))
    second_row[1].metric(
        "Positive Precision", fmt(payload["positive_precision"])
    )
    second_row[2].metric("Positive F1", fmt(payload["positive_f1"]))
    second_row[3].metric("Số dòng kiểm thử", f"{payload['n_samples']:,}")

    status_note(payload, "auroc", "auroc_status")
    status_note(payload, "brier_score", "brier_score_status")

    st.caption(payload["metric_family_note_vi"])

    with st.expander("Thông tin mô hình và tập dữ liệu đánh giá"):

        st.write(
            f"**Mô hình:** {payload['model']} · "
            f"{payload['n_estimators']} cây · "
            f"{payload['n_features']} đặc trưng"
        )
        st.write(f"**Dữ liệu đánh giá:** {payload['dataset']}")
        st.write(
            f"**Target / lớp dương:** {payload['target_column']} / "
            f"{payload['positive_class']} — "
            f"{payload['positive_class_meaning_vi']}"
        )
        st.write(
            f"**Căn cứ xác định lớp dương:** {payload['positive_class_evidence']}"
        )
        st.write("**Đặc trưng:** " + ", ".join(payload["features"]))
        st.write(f"**Thời điểm tính:** {payload['generated_at']}")
        st.write(f"**Ngưỡng:** {payload['threshold_note_vi']}")


def render_per_class(payload: Dict[str, Any]) -> None:
    """Per-class precision / recall / F1 / support."""

    st.subheader("Chỉ số theo từng lớp")
    st.caption("Per-class metrics")

    per_class = payload.get("per_class_metrics") or {}

    if not per_class:
        st.warning("Không có chỉ số theo lớp.")
        return

    rows = [
        {
            "Lớp": item["label"],
            "Vai trò": (
                "Dương (nguy cơ)" if item["role"] == "positive" else "Âm"
            ),
            "Precision": fmt(item["precision"]),
            "Recall": fmt(item["recall"]),
            "F1": fmt(item["f1"]),
            "Support": item["support"],
        }
        for item in per_class.values()
    ]

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.caption(
        "Mô hình nhị phân: Macro-F1 là trung bình F1 của lớp âm và lớp dương, "
        "nên lớp thiểu số vẫn ảnh hưởng tới chỉ số này."
    )


def render_confusion_matrix(
    payload: Dict[str, Any],
    reports_dir: Path,
) -> None:
    """Confusion matrix with counts and rates."""

    st.subheader("Ma trận nhầm lẫn (theo ngưỡng mô hình)")
    st.caption(
        "Confusion Matrix · True Negative · False Positive · "
        "False Negative · True Positive"
    )

    matrix = payload.get("confusion_matrix")

    if not matrix:
        st.warning("Không có ma trận nhầm lẫn.")
        return

    labels = matrix["labels"]

    frame = pd.DataFrame(
        matrix["matrix"],
        index=[f"Thực tế {label}" for label in labels],
        columns=[f"Dự đoán {label}" for label in labels],
    )

    st.dataframe(frame, use_container_width=True)

    counts = payload["counts"]

    first_row = st.columns(4)
    first_row[0].metric("TN (âm đúng)", f"{counts['true_negative']:,}")
    first_row[1].metric("FP (dương tính giả)", f"{counts['false_positive']:,}")
    first_row[2].metric("FN (âm tính giả)", f"{counts['false_negative']:,}")
    first_row[3].metric("TP (dương đúng)", f"{counts['true_positive']:,}")

    rates = payload["rates"]

    second_row = st.columns(4)
    second_row[0].metric("FPR", fmt_share(rates["false_positive_rate"]))
    second_row[1].metric("FNR", fmt_share(rates["false_negative_rate"]))
    second_row[2].metric("TPR (Recall dương)", fmt_share(rates["true_positive_rate"]))
    second_row[3].metric(
        "Tỷ lệ gọi dương", fmt_share(rates["positive_prediction_rate"])
    )

    plot_path = (
        reports_dir / PLOTS_DIRNAME / f"{payload['disease']}_confusion_matrix.png"
    )

    if plot_path.exists():
        st.image(str(plot_path), caption=plot_path.name)


def render_roc(payload: Dict[str, Any], reports_dir: Path) -> None:
    """ROC curve and AUROC, both from predicted probabilities."""

    st.subheader("ROC / AUROC")
    st.caption("ROC Curve · False Positive Rate · True Positive Rate · AUROC")

    roc = payload.get("roc") or {}

    if not roc.get("available"):
        st.warning(
            "Không vẽ được đường ROC: "
            f"{roc.get('status', 'không có dữ liệu')}. "
            "AUROC cần cả lớp dương và lớp âm trong tập đánh giá."
        )
        return

    st.metric("AUROC (dùng xác suất dự đoán)", fmt(roc.get("auroc")))
    st.caption(f"{roc.get('n_points')} điểm vẽ · {roc.get('note')}")

    plot_path = reports_dir / PLOTS_DIRNAME / f"{payload['disease']}_roc.png"

    if plot_path.exists():
        st.image(str(plot_path), caption=plot_path.name)

    roc_csv = reports_dir / f"{payload['disease']}_roc.csv"
    table = load_table(str(roc_csv), file_mtime(roc_csv))

    if table is not None and not table.empty:
        with st.expander("Bảng điểm ROC (FPR / TPR / ngưỡng)"):
            st.dataframe(table, use_container_width=True, hide_index=True)


def render_calibration(payload: Dict[str, Any], reports_dir: Path) -> None:
    """Brier score and reliability diagram."""

    st.subheader("Hiệu chuẩn (Calibration)")
    st.caption("Calibration / Reliability Diagram")
    st.caption(
        "Reliability diagram so sánh xác suất dự đoán với tỷ lệ dương tính "
        "quan sát được trên các nhóm xác suất."
    )

    st.metric(
        "Brier score (xác suất lớp dương)",
        fmt(payload.get("brier_score")),
    )

    calibration = payload.get("calibration") or {}

    if not calibration.get("available"):
        st.warning(
            "Không tính được đường hiệu chuẩn: "
            f"{calibration.get('status', 'không có dữ liệu')}"
        )
        return

    row = st.columns(4)
    row[0].metric("Số bin", calibration.get("n_bins"))
    row[1].metric("Kiểu chia bin", calibration.get("strategy"))
    row[2].metric(
        "ECE (bins đều)",
        fmt(calibration.get("expected_calibration_error")),
    )
    row[3].metric(
        "Xác suất dự đoán TB",
        fmt(calibration.get("mean_predicted_probability")),
    )

    st.caption(calibration.get("interpretation_note_vi", ""))

    plot_path = (
        reports_dir / PLOTS_DIRNAME / f"{payload['disease']}_reliability.png"
    )

    if plot_path.exists():
        st.image(str(plot_path), caption=plot_path.name)

    calibration_csv = reports_dir / f"{payload['disease']}_calibration.csv"
    table = load_table(str(calibration_csv), file_mtime(calibration_csv))

    if table is not None and not table.empty:
        st.dataframe(table, use_container_width=True, hide_index=True)


def _render_error_table(csv_path: Path, expected_count: Optional[int]) -> None:
    """Show one persisted FP/FN record table with a CSV download."""

    table = load_table(str(csv_path), file_mtime(csv_path))

    if table is None:
        st.warning(f"Chưa có bảng lỗi: {csv_path.name}")
        return

    if table.empty:
        st.info(f"Không có bản ghi nào (số lượng = {expected_count}).")
        return

    st.caption(
        f"{len(table):,} bản ghi · hiển thị tối đa 200 dòng đầu tiên "
        "(chỉ số dòng + đặc trưng của mô hình, không có thông tin định danh)"
    )

    st.dataframe(table.head(200), use_container_width=True, hide_index=True)

    st.download_button(
        "⬇️ Tải CSV",
        data=table.to_csv(index=False).encode("utf-8"),
        file_name=csv_path.name,
        mime="text/csv",
    )


def render_error_analysis(payload: Dict[str, Any], reports_dir: Path) -> None:
    """False-positive and false-negative analysis."""

    st.subheader("Phân tích lỗi")
    st.caption("False Positive analysis · False Negative analysis")

    summary = payload.get("error_analysis") or {}

    if not summary.get("available"):
        st.warning(f"Không phân tích được lỗi: {summary.get('status')}")
        return

    first_row = st.columns(4)
    first_row[0].metric(
        "Dương tính giả (FP)", f"{summary['false_positive_count']:,}"
    )
    first_row[1].metric("FP rate / lớp âm", fmt_share(summary["false_positive_rate"]))
    first_row[2].metric(
        "Âm tính giả (FN)", f"{summary['false_negative_count']:,}"
    )
    first_row[3].metric(
        "FN rate / lớp dương", fmt_share(summary["false_negative_rate"])
    )

    st.caption(
        "FP = thực tế thuộc lớp âm nhưng mô hình báo dương. "
        "FN = thực tế thuộc nhóm nguy cơ nhưng mô hình không báo nguy cơ."
    )

    probability_summary = summary.get("group_probability_summary") or {}

    if probability_summary:
        st.markdown("**Xác suất dự đoán trung bình theo nhóm**")
        st.dataframe(
            pd.DataFrame(
                [
                    {"Nhóm": key, "Xác suất trung bình": fmt(value)}
                    for key, value in probability_summary.items()
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )

    st.info(summary.get("screening_note_vi", ""))

    labels_vi = summary.get("labels_vi") or {}

    tab_false_positive, tab_false_negative = st.tabs(
        [
            labels_vi.get("false_positive", "False Positives"),
            labels_vi.get(
                "false_negative",
                "False Negatives / Missed Positive-Risk Cases",
            ),
        ]
    )

    with tab_false_positive:
        _render_error_table(
            reports_dir / f"{payload['disease']}_false_positives.csv",
            summary["false_positive_count"],
        )

    with tab_false_negative:
        _render_error_table(
            reports_dir / f"{payload['disease']}_false_negatives.csv",
            summary["false_negative_count"],
        )

    dropped = summary.get("dropped_identifier_columns") or []

    if dropped:
        st.caption(f"Cột đã bị loại để tránh lộ định danh: {dropped}")
    else:
        st.caption(
            "Không có cột định danh nào trong bảng lỗi (chỉ số dòng + đặc trưng "
            "của mô hình)."
        )


def render_decision_engine_view(payload: Dict[str, Any]) -> None:
    """Risk-level distribution of the live Decision Engine."""

    st.subheader("Ngưỡng mô hình và mức độ rủi ro")
    st.caption(
        "Threshold vs Decision Engine · Cardiovascular = 0.40 · Diabetes = 0.35 "
        "· Hypertension = 0.35"
    )
    st.caption(
        "Decision Engine risk-level cutoffs: < 0.30 = LOW · "
        "0.30-0.69 = MODERATE · >= 0.70 = HIGH"
    )
    st.caption(
        "Ngưỡng mô hình (model threshold) dùng cho đánh giá nhị phân; "
        "mốc Decision Engine dùng cho mức rủi ro hiển thị. "
        "Hai khái niệm này không nên bị nhầm lẫn."
    )

    st.subheader("Phân mức nguy cơ của Decision Engine")

    view = payload.get("decision_engine_view") or {}

    if not view.get("available"):
        st.warning(f"Không tính được phân bố mức nguy cơ: {view.get('status')}")
        return

    distribution = view["distribution"]
    share = view["distribution_share"]

    columns = st.columns(4)

    for column, level in zip(columns, ("LOW", "MODERATE", "HIGH", "UNKNOWN")):
        column.metric(
            level,
            f"{distribution.get(level, 0):,}",
            fmt_share(share.get(level)),
        )

    cutoffs = view["cutoffs"]

    st.write(
        f"Mốc phân mức: LOW < {cutoffs['low_below']} ≤ MODERATE < "
        f"{cutoffs['high_at_or_above']} ≤ HIGH "
        f"(nguồn: {cutoffs['source']})."
    )

    st.caption(view.get("note_vi", ""))

    mismatch = view.get("model_threshold_vs_high_risk_level")

    if mismatch:
        st.write(
            "**So sánh với ngưỡng mô hình:** mô hình gọi dương "
            f"{mismatch['model_positive_count']:,} dòng; Decision Engine xếp HIGH "
            f"{mismatch['high_risk_level_count']:,} dòng; hai cách khác nhau ở "
            f"{mismatch['rows_where_the_two_rules_disagree']:,} dòng "
            f"({fmt_share(mismatch['disagreement_share'])})."
        )

    high_metrics = view.get("high_as_positive_metrics")

    if high_metrics and high_metrics.get("available"):
        with st.expander(
            "Nếu coi mức HIGH là 'dương tính' (góc nhìn thay thế, KHÔNG thay đổi "
            "logic của ứng dụng)"
        ):
            st.write(
                f"Precision = {fmt(high_metrics['positive_precision'])} · "
                f"Recall = {fmt(high_metrics['positive_recall'])} · "
                f"F1 = {fmt(high_metrics['positive_f1'])}"
            )
            st.write(
                "Thông số này chỉ để so sánh hai cách phân ngưỡng; ứng dụng vẫn "
                "dùng đúng logic hiện có."
            )


def render_dataset_limitations(payload: Dict[str, Any]) -> None:
    """Verified dataset facts plus the honest knowledge gaps."""

    st.subheader("Hạn chế của bộ dữ liệu")
    st.caption("Giới hạn dữ liệu và mô hình · Dataset limitations")

    limitations = payload.get("dataset_limitations")

    if not limitations:
        st.warning("Không có thông tin hạn chế dữ liệu.")
        return

    sizes = limitations["sample_size"]

    row = st.columns(4)
    row[0].metric("Dòng dữ liệu gốc", f"{sizes['raw_rows']:,}")
    row[1].metric("Dòng train", f"{sizes.get('train_rows') or 0:,}")
    row[2].metric("Dòng test", f"{sizes['test_rows']:,}")
    row[3].metric("Số đặc trưng", sizes["feature_count"])

    st.write(f"**Nguồn dữ liệu (trong repo):** {limitations['dataset_source_vi']}")

    provenance = limitations["dataset_provenance"]
    st.warning(f"**Nguồn gốc / giấy phép:** {provenance['status']} — {provenance['detail_vi']}")

    distribution = limitations["class_distribution"]

    st.write(
        "**Phân bố nhãn (tập kiểm thử):** "
        + " · ".join(
            f"lớp {label}: {count:,} "
            f"({fmt_share(distribution['test_share'].get(label))})"
            for label, count in distribution["test_counts"].items()
        )
    )

    st.write(
        "**Phân bố nhãn (toàn bộ dữ liệu gốc):** "
        + " · ".join(
            f"lớp {label}: {count:,}"
            for label, count in distribution["raw_counts"].items()
        )
    )

    missing = limitations["missing_values"]

    st.write(
        f"**Giá trị thiếu trong dữ liệu gốc:** {missing['raw_total_missing']:,} "
        f"(chi tiết: {missing['raw_missing_by_column'] or 'không có'}) · "
        f"tập kiểm thử đã xử lý: {missing['processed_test_missing']} giá trị thiếu"
    )

    if limitations.get("zero_value_counts"):
        st.write(
            f"**Giá trị 0 bất thường trong dữ liệu gốc:** "
            f"{limitations['zero_value_counts']}"
        )

    st.write(f"**Tiền xử lý:** {limitations['preprocessing_steps_vi']}")
    st.write(f"**Transformer khi dự đoán:** {limitations['transformer_vi']}")
    st.write(f"**Cách tạo tập đánh giá:** {limitations['dataset_source_used_for_evaluation']}")

    with st.expander("Kiểm chứng tập đánh giá bằng chính pipeline dự đoán"):
        st.json(limitations["verification"])

    st.markdown("**Mức độ đại diện theo nhóm nhân khẩu học**")

    for note in limitations["representation_limitations_vi"]:
        st.write(f"• {note}")

    with st.expander("Phân bố nhân khẩu học quan sát được trong tập đánh giá"):
        st.json(limitations["demographic_coverage"])

    st.error(
        f"**Kiểm chứng độc lập:** {limitations['external_validation']['statement_vi']}"
    )

    st.markdown("**Hạn chế về nhãn**")

    for item in limitations["label_limitations_vi"]:
        st.write(f"• {item}")

    st.markdown("**Hạn chế đã biết**")

    for item in limitations["known_limitations_vi"]:
        st.write(f"• {item}")


def render_trustworthy_cross_reference(project_root: Any) -> None:
    """Reuse the existing Trustworthy AI CSV outputs instead of duplicating them."""

    st.subheader("Tham chiếu Trustworthy AI hiện có")

    st.caption(
        "Các tệp kết quả công bằng / thiên lệch / độ bền vững có sẵn của dự án, "
        "đọc bằng chính helper của trustworthy_ai_dashboard.py."
    )

    directories = get_trustworthy_ai_dirs(project_root)

    for name in ("fairness", "bias", "robustness", "xai"):

        results = load_csv_files(directories[name])

        if not results:
            st.info(f"Chưa có tệp kết quả cho: {name}")
            continue

        with st.expander(f"{name} ({len(results)} tệp)"):

            for filename, dataframe in results.items():

                st.markdown(f"**{filename}**")

                if dataframe.empty:
                    st.info("Tệp này rỗng.")
                    continue

                st.dataframe(
                    dataframe,
                    use_container_width=True,
                    hide_index=True,
                )


def render_disclaimer(payload: Dict[str, Any]) -> None:
    """Risk-screening and evaluation disclaimers."""

    st.subheader("Cảnh báo sử dụng")
    st.caption("Medical/risk-screening disclaimer")
    st.warning(
        "Đây là hệ thống sàng lọc/phân loại nguy cơ mang tính hỗ trợ, không phải "
        "công cụ chẩn đoán y khoa và không thay thế bác sĩ hoặc nhân viên y tế "
        "có chuyên môn."
    )
    st.info(
        "Các chỉ số đánh giá mô tả hành vi của mô hình trên tập dữ liệu kiểm thử "
        "được giữ lại. Các chỉ số này không chứng minh giá trị lâm sàng hoặc độ "
        "chính xác chẩn đoán."
    )

    st.error(EVALUATION_DISCLAIMER_VI)
    st.info(payload.get("screening_disclaimer_reference_vi", ""))

    st.markdown("**Những điểm phải lưu ý khi đọc các chỉ số này**")

    for item in (
        "Chỉ số đánh giá KHÔNG đồng nghĩa với giá trị lâm sàng.",
        "Hiệu năng trên tập kiểm thử KHÔNG đại diện cho hiệu năng ngoài thực tế.",
        "Thiên lệch trong dữ liệu có thể ảnh hưởng tới kết quả.",
        "Một số nhóm dân số có thể chưa được đại diện đầy đủ trong dữ liệu.",
        "Nhãn của bộ dữ liệu có thể chứa sai số.",
        "Chưa có kiểm chứng độc lập (đã xác nhận trong repository).",
        "Hệ thống chỉ dùng cho sàng lọc/hỗ trợ nguy cơ, không phải chẩn đoán y "
        "khoa và không thay thế nhân viên y tế.",
    ):
        st.write(f"• {item}")


# ============================================================
# 4. MAIN DASHBOARD
# ============================================================


def render_model_evaluation_dashboard(project_root: Any) -> None:
    """Main entry point called from ``src/ui/app.py``.

    Reads the persisted artifacts under ``reports/model_evaluation`` (cached), so
    a Streamlit rerun does not recompute anything. Only the explicit
    "Chạy lại đánh giá" button triggers the evaluation module.
    """

    st.markdown("---")
    st.header("📊 Đánh giá mô hình / Trustworthy AI")
    st.caption("Model Evaluation Dashboard")

    st.write(
        "Các chỉ số được tính trên tập kiểm thử đã giữ lại của chính repository, "
        "bằng chính các mô hình đã đóng băng. Không huấn luyện lại, không fit "
        "lại, không ghi đè mô hình hay ngưỡng."
    )
    st.caption("Evaluation type: Held-out test set")

    reports_dir = Path(project_root) / "reports" / "model_evaluation"
    summary_path = reports_dir / "summary.json"

    header_columns = st.columns([3, 1])

    with header_columns[0]:

        if summary_path.exists():

            with summary_path.open(encoding="utf-8") as handle:
                summary = json.load(handle)

            st.caption(
                f"Kết quả sinh lúc {summary.get('generated_at')} · {reports_dir}"
            )

            if summary.get("failed_diseases"):
                st.error(
                    "Mô hình không đánh giá được: "
                    + "; ".join(
                        f"{disease}: {message}"
                        for disease, message in summary[
                            "failed_diseases"
                        ].items()
                    )
                )

        else:

            st.info(
                "Chưa có kết quả đánh giá. Bấm 'Chạy lại đánh giá' hoặc chạy "
                "`python src/evaluation/run_evaluation.py`."
            )

    with header_columns[1]:

        run_now = st.button(
            "🔄 Chạy lại đánh giá",
            help=(
                "Đọc mô hình đóng băng và tính lại chỉ số trên tập kiểm thử đã "
                "giữ lại (không huấn luyện lại)."
            ),
        )

    if run_now:

        with st.spinner("Đang đánh giá các mô hình trên tập kiểm thử..."):

            output = run_evaluation(
                reports_dir=str(reports_dir),
                save_plots=True,
            )

        if output["failures"]:
            st.error(
                "Evaluation incomplete — required artifact missing: "
                + "; ".join(
                    f"{disease}: {message}"
                    for disease, message in output["failures"].items()
                )
            )
        else:
            st.success(
                f"Đã cập nhật kết quả cho {len(output['results'])} mô hình."
            )

        st.cache_data.clear()

    loaded = {
        disease: get_metrics(reports_dir, disease) for disease in DISEASE_ORDER
    }

    available = {
        disease: payload for disease, payload in loaded.items() if payload
    }

    render_overview(available)

    missing = [disease for disease in DISEASE_ORDER if disease not in available]

    if missing:
        st.warning(
            "Chưa có artifact đánh giá cho: " + ", ".join(missing) + "."
        )

    if not available:
        st.info(
            "Bảng điều khiển sẽ hiển thị ngay khi có artifact đánh giá. "
            "Chạy `python src/evaluation/run_evaluation.py` để sinh artifact."
        )
        return

    model_labels = {
        disease: (
            f"{payload['display_name_vi']} ({payload['display_name_en']})"
        )
        for disease, payload in available.items()
    }

    selected_disease = st.radio(
        "Chọn mô hình",
        options=list(available),
        format_func=lambda key: model_labels[key],
        horizontal=True,
        key="model_evaluation_selector",
    )

    payload = available[selected_disease]

    tabs = st.tabs(
        [
            "📈 Hiệu năng",
            "🧮 Theo lớp",
            "🔲 Ma trận nhầm lẫn",
            "📉 ROC / AUROC",
            "🎯 Hiệu chuẩn",
            "⚠️ Phân tích lỗi",
            "🧭 Decision Engine",
            "📋 Hạn chế dữ liệu",
            "📚 Tham chiếu Trustworthy AI",
            "⚖️ Cảnh báo",
        ]
    )

    with tabs[0]:
        render_performance(payload)

    with tabs[1]:
        render_per_class(payload)

    with tabs[2]:
        render_confusion_matrix(payload, reports_dir)

    with tabs[3]:
        render_roc(payload, reports_dir)

    with tabs[4]:
        render_calibration(payload, reports_dir)

    with tabs[5]:
        render_error_analysis(payload, reports_dir)

    with tabs[6]:
        render_decision_engine_view(payload)

    with tabs[7]:
        render_dataset_limitations(payload)

    with tabs[8]:
        render_trustworthy_cross_reference(project_root)

    with tabs[9]:
        render_disclaimer(payload)