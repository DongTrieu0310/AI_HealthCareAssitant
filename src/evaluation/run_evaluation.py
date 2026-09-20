"""
Đánh giá 3 mô hình sàng lọc nguy cơ (đã đóng băng) trên tập kiểm thử giữ lại.

Cách chạy::

    python src/evaluation/run_evaluation.py
    python src/evaluation/run_evaluation.py --diseases diabetes
    python src/evaluation/run_evaluation.py --no-plots

Script này KHÔNG huấn luyện, KHÔNG fit lại, KHÔNG ghi đè mô hình, scaler,
imputer hay ngưỡng. Nó chỉ đọc các artifact hiện có, tính chỉ số trên tập
kiểm thử đã giữ lại và ghi kết quả vào ``reports/model_evaluation/``.

Mã thoát: 0 nếu tất cả mô hình được đánh giá xong, 1 nếu có mô hình không thể
đánh giá (thiếu artifact) - khi đó thông báo ghi rõ
"Evaluation incomplete — required artifact missing."
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import List, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from evaluation.model_evaluation import (  # noqa: E402
    DISEASE_ORDER,
    EVALUATION_DISCLAIMER_VI,
    run_evaluation,
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Đánh giá mô hình sàng lọc nguy cơ trên tập kiểm thử đã giữ lại "
            "(không huấn luyện lại)."
        )
    )

    parser.add_argument(
        "--diseases",
        default=",".join(DISEASE_ORDER),
        help=(
            "Danh sách bệnh, cách nhau bởi dấu phẩy. "
            f"Mặc định: {','.join(DISEASE_ORDER)}"
        ),
    )

    parser.add_argument(
        "--reports-dir",
        default=None,
        help=(
            "Thư mục ghi kết quả. "
            "Mặc định: reports/model_evaluation"
        ),
    )

    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Chỉ sinh số liệu (JSON/CSV), không tạo biểu đồ.",
    )

    return parser.parse_args(argv)


def format_metric(value: Optional[float], digits: int = 4) -> str:
    """Format a metric value, showing n/a when it is unavailable."""

    if value is None:
        return "n/a"

    return f"{value:.{digits}f}"


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the evaluation for every selected model."""

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):  # pragma: no cover - old interpreters
        pass

    args = parse_args(argv)

    diseases: List[str] = [
        item.strip() for item in args.diseases.split(",") if item.strip()
    ]

    # Warnings must stay visible: they carry the "unavailable metric" reasons.
    warnings.simplefilter("default")

    print("=" * 78)
    print("AI HEALTHCARE ASSISTANT - MODEL EVALUATION")
    print("=" * 78)
    print("Không huấn luyện lại, không fit lại, không ghi đè mô hình/ngưỡng.")
    print(f"Mô hình được đánh giá: {', '.join(diseases)}")

    output = run_evaluation(
        diseases,
        reports_dir=args.reports_dir,
        save_plots=not args.no_plots,
    )

    results = output["results"]
    failures = output["failures"]

    for disease, payload in results.items():

        print()
        print("-" * 78)
        print(f"{payload['display_name_vi']} ({disease}) — {payload['model_file']}")
        print("-" * 78)
        print(f"Dataset            : {payload['dataset']}")
        print(f"So dòng kiểm thử   : {payload['n_samples']}")
        print(f"Target / lớp dương : {payload['target_column']} / "
              f"{payload['positive_class']} ({payload['positive_class_meaning_vi']})")
        print(f"Ngưỡng mô hình     : {payload['threshold']} "
              f"(nguồn: {payload['threshold_source']})")
        print(f"Accuracy           : {format_metric(payload['accuracy'])}")
        print(f"Macro-F1           : {format_metric(payload['macro_f1'])}")

        auroc_text = format_metric(payload["auroc"])
        if payload["auroc"] is None:
            auroc_text += f"  [{payload['auroc_status']}]"
        print(f"AUROC              : {auroc_text}")

        brier_text = format_metric(payload["brier_score"])
        if payload["brier_score"] is None:
            brier_text += f"  [{payload['brier_score_status']}]"
        print(f"Brier score        : {brier_text}")

        print(
            "Positive P/R/F1    : "
            f"{format_metric(payload['positive_precision'])} / "
            f"{format_metric(payload['positive_recall'])} / "
            f"{format_metric(payload['positive_f1'])}"
        )

        matrix = payload["confusion_matrix"]
        print(f"Confusion matrix   : {matrix['matrix']} (nhãn {matrix['labels']})")
        print(f"False positives    : {payload['false_positive_count']}")
        print(f"False negatives    : {payload['false_negative_count']}")

        distribution = payload["decision_engine_view"]["distribution"]
        print(
            "Decision Engine    : "
            f"LOW={distribution.get('LOW')} "
            f"MODERATE={distribution.get('MODERATE')} "
            f"HIGH={distribution.get('HIGH')}"
        )

        print("Artifact đã ghi    :")
        for name, path in payload["artifacts_written"].items():
            print(f"  - {name}: {path}")

    if failures:

        print()
        print("=" * 78)
        print("EVALUATION INCOMPLETE")
        print("=" * 78)

        for disease, message in failures.items():
            print(
                f"[FAILED] {disease}: Evaluation incomplete — required artifact "
                f"missing. {message}"
            )

    print()
    print("=" * 78)
    print(f"Summary JSON: {output['summary_path']}")
    print(EVALUATION_DISCLAIMER_VI)
    print("=" * 78)

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())