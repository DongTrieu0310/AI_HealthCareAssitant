"""
AI Healthcare Assistant - Model Evaluation Layer
================================================

Offline evaluation of the three FROZEN risk-screening models against the
held-out test splits that already exist in the repository.

Guarantees
----------
* No retraining, no refitting, no model replacement, no threshold invention.
* No duplicated preprocessing: the live prediction pipeline
  (``src/prediction/prediction_layer.py``) is reused for feature validation
  (``validate_input``) and for applying the training-time scaler / imputer
  (``apply_transformers``).
* No second prediction implementation: probabilities come from the same frozen
  ``*_random_forest.pkl`` files the application loads at runtime, through the
  vectorized form of the live call ``model.predict_proba(X)[0, 1]``.
* Every reported number is computed from the repository's own test split and its
  own ``*_threshold.pkl`` artifact. Nothing is hardcoded; if a required artifact
  is missing the evaluation fails loudly instead of substituting a value.

Metric families are kept apart on purpose
-----------------------------------------
probability-based : AUROC, Brier score, reliability (calibration) curve,
                    ROC curve
threshold-based   : accuracy, precision, recall, F1, confusion matrix - using the
                    threshold loaded from ``data/models/<disease>_threshold.pkl``

The Risk Decision Engine (``src/decision_engine/decision_engine.py``) uses
different cut-offs (LOW < 0.30 <= MODERATE < 0.70 <= HIGH). That mismatch is NOT
changed here; the Decision Engine risk-level distribution is reported next to the
model-threshold metrics so the difference stays visible.

Evaluation results describe model behaviour on a held-out sample of the same
data distribution the models were trained on. They do NOT establish clinical
validity or diagnostic accuracy.
"""

from __future__ import annotations

import inspect
import json
import math
import sys
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd

import matplotlib

# The evaluation runs head-less (scripts, tests and Streamlit). A non-interactive
# backend keeps `plt.show()` calls from other project scripts out of the way.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split  # noqa: E402


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# The live prediction pipeline is the single source of truth for preprocessing.
from prediction import prediction_layer  # noqa: E402

# The live risk-level cut-offs are reused (never redefined) so the documented
# difference between thresholds and risk levels cannot drift.
from decision_engine.decision_engine import (  # noqa: E402
    HIGH_RISK_THRESHOLD,
    LOW_RISK_THRESHOLD,
    classify_risk,
)


DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_MODELS = PROJECT_ROOT / "data" / "models"

DEFAULT_REPORTS_DIR = PROJECT_ROOT / "reports" / "model_evaluation"
PLOTS_DIRNAME = "plots"

TOLERANCE = 1e-8

# The held-out split parameters that the existing preprocessing scripts used.
# Verified against src/preprocessing/cardio_preprocessing.py and
# src/preprocessing/diabetes_preprocessing.py (test_size=0.20, random_state=42,
# stratify=y). The split is re-derived, never re-fitted.
TEST_SIZE = 0.20
RANDOM_STATE = 42

# Calibration convention: sklearn's default uniform binning over [0, 1].
# The project itself has no other justified convention.
N_CALIBRATION_BINS = 10
CALIBRATION_STRATEGY = "uniform"

POSITIVE_LABEL = 1
NEGATIVE_LABEL = 0

# Column names that must never reach an error-analysis table.
IDENTIFIER_PATTERNS = (
    "id",
    "name",
    "phone",
    "email",
    "address",
    "ssn",
    "medical_record",
    "patient_number",
)

_CALIBRATION_HAS_POS_LABEL = (
    "pos_label" in inspect.signature(calibration_curve).parameters
)
_BRIER_HAS_POS_LABEL = (
    "pos_label" in inspect.signature(brier_score_loss).parameters
)

# ============================================================
# 2. POSITIVE-CLASS DETERMINATION
# ============================================================
#
# The positive class is NOT assumed. It was read from the existing project:
#
# * `prediction_layer.predict_disease()` computes
#   `model.predict_proba(validated_data)[0, 1]` and then
#   `prediction = int(probability >= threshold)`, i.e. class index 1 is the
#   class the application reports as "risk present".
# * The existing evaluation scripts name class 1 as the disease class:
#       src/models/cardio_evalution.py        ["No Cardiovascular Disease", "Cardiovascular Disease"]
#       src/models/diabetes_evaluation.py     ["No Diabetes", "Diabetes"]
#       src/models/hypertension_evaluasion.py ["No Hypertension Risk", "Hypertension Risk"]
# * Target columns: `cardio`, `Outcome`, `Risk` - 1 means the risk/condition.
#
# `load_model_artifacts()` additionally reads `model.classes_` from the frozen
# pickle and uses `model.classes_[-1]` as the positive label, so the mapping is
# also data-driven at run time and a mismatch is reported instead of ignored.

POSITIVE_CLASS_EVIDENCE = (
    "prediction_layer.predict_disease() uses predict_proba(X)[0, 1] and "
    "prediction = int(probability >= threshold); existing evaluation scripts "
    "label class 1 as the disease class (cardio_evalution.py, "
    "diabetes_evaluation.py, hypertension_evaluasion.py)."
)


# ============================================================
# 3. DISEASE CONFIGURATION
# ============================================================


@dataclass(frozen=True)
class DiseaseConfig:
    """Static, verified description of one evaluated model."""

    key: str                 # repository/report key, e.g. "cardiovascular"
    prediction_key: str      # key used by data/models + prediction_layer ("cardio")
    file_prefix: str         # prefix of data/processed/<prefix>_X_test.csv
    display_name_vi: str
    display_name_en: str
    target_column: str
    positive_meaning_vi: str
    negative_meaning_vi: str
    raw_file: str
    raw_drop_duplicates: bool
    raw_age_divisor: Optional[float]
    preprocessing_vi: str
    transformer_vi: str
    demographic_columns: Tuple[str, ...]


DISEASE_CONFIGS: Dict[str, DiseaseConfig] = {
    "cardiovascular": DiseaseConfig(
        key="cardiovascular",
        prediction_key="cardio",
        file_prefix="cardio",
        display_name_vi="Tim mạch",
        display_name_en="Cardiovascular",
        target_column="cardio",
        positive_meaning_vi="Có nguy cơ / dấu hiệu bệnh tim mạch",
        negative_meaning_vi="Không thuộc nhóm nguy cơ tim mạch",
        raw_file="cardio.csv",
        raw_drop_duplicates=True,
        raw_age_divisor=365.25,
        preprocessing_vi=(
            "Loại bỏ dòng trùng lặp, đổi tuổi từ ngày sang năm (chia 365.25), "
            "chia train/test 80/20 (random_state=42, stratify), chuẩn hoá "
            "StandardScaler fit trên tập train."
        ),
        transformer_vi="StandardScaler (cardio_scaler.pkl) - bắt buộc khi dự đoán.",
        demographic_columns=("gender", "age"),
    ),
    "diabetes": DiseaseConfig(
        key="diabetes",
        prediction_key="diabetes",
        file_prefix="diabetes",
        display_name_vi="Đái tháo đường",
        display_name_en="Diabetes",
        target_column="Outcome",
        positive_meaning_vi="Có nguy cơ / dấu hiệu đái tháo đường",
        negative_meaning_vi="Không thuộc nhóm nguy cơ đái tháo đường",
        raw_file="diabetes.csv",
        raw_drop_duplicates=False,
        raw_age_divisor=None,
        preprocessing_vi=(
            "Giá trị Insulin 102.5 và 169.5 được coi là bất thường và đổi thành "
            "NaN, điền khuyết bằng SimpleImputer(strategy='median') fit trên tập "
            "train, chia train/test 80/20 (random_state=42, stratify), chuẩn hoá "
            "StandardScaler fit trên tập train."
        ),
        transformer_vi=(
            "SimpleImputer(median) (diabetes_imputer.pkl) + StandardScaler "
            "(diabetes_scaler.pkl)."
        ),
        demographic_columns=("Age",),
    ),
    "hypertension": DiseaseConfig(
        key="hypertension",
        prediction_key="hypertension",
        file_prefix="hypertension",
        display_name_vi="Tăng huyết áp",
        display_name_en="Hypertension",
        target_column="Risk",
        positive_meaning_vi="Có nguy cơ tăng huyết áp",
        negative_meaning_vi="Không thuộc nhóm nguy cơ tăng huyết áp",
        raw_file="hypertension.csv",
        raw_drop_duplicates=False,
        raw_age_divisor=None,
        preprocessing_vi=(
            "Chia train/test 80/20 (random_state=42, stratify) - đã kiểm chứng "
            "lại đúng từng dòng với data/raw/hypertension.csv. Các giá trị thiếu "
            "của dữ liệu gốc đã được điền khuyết TRƯỚC khi lưu, nhưng script "
            "điền khuyết không còn trong repository => "
            "NOT FOUND / NEEDS VERIFICATION."
        ),
        transformer_vi=(
            "Không có transformer: mô hình được huấn luyện trên đơn vị gốc "
            "(sysBP/diaBP tính bằng mmHg)."
        ),
        demographic_columns=("male", "age"),
    ),
}

DISEASE_ORDER: Tuple[str, ...] = ("cardiovascular", "diabetes", "hypertension")


def get_disease_config(disease: str) -> DiseaseConfig:
    """Return the verified configuration of one evaluated disease."""

    try:
        return DISEASE_CONFIGS[disease]
    except KeyError:
        raise KeyError(
            f"Unknown disease '{disease}'. "
            f"Known diseases: {list(DISEASE_CONFIGS)}"
        ) from None
# ============================================================
# 4. LOAD FROZEN MODEL ARTIFACTS (read only)
# ============================================================


@lru_cache(maxsize=1)
def _cached_models() -> Dict[str, Any]:
    """Load the frozen models through the live loader.

    Cached because the pickles are large. Callers must never mutate them.
    """

    return prediction_layer.load_models()


@lru_cache(maxsize=1)
def _cached_thresholds() -> Dict[str, float]:
    """Load the real thresholds through the live loader.

    The values always come from ``data/models/<disease>_threshold.pkl``. They are
    never hardcoded and never replaced by 0.5.
    """

    return prediction_layer.load_thresholds()


@lru_cache(maxsize=1)
def _cached_transformers() -> Dict[str, Dict[str, Any]]:
    """Load the training-time scalers / imputers through the live loader."""

    return prediction_layer.load_transformers()


def clear_artifact_cache() -> None:
    """Drop cached artifacts so the next call re-reads them from disk."""

    _cached_models.cache_clear()
    _cached_thresholds.cache_clear()
    _cached_transformers.cache_clear()


def load_model_artifacts(disease: str) -> Dict[str, Any]:
    """Load the frozen model, its feature list and its real threshold.

    Raises
    ------
    FileNotFoundError
        When a required artifact is missing. No substitute value is created -
        the evaluation must fail loudly instead of reporting invented numbers.
    """

    cfg = get_disease_config(disease)
    prediction_key = cfg.prediction_key

    model_path = prediction_layer.MODEL_PATHS.get(prediction_key)

    if model_path is None:
        raise KeyError(
            f"[{disease}] No model path registered in "
            f"prediction_layer.MODEL_PATHS for '{prediction_key}'."
        )

    if not model_path.exists():
        raise FileNotFoundError(
            f"[{disease}] Required model artifact is missing: {model_path}. "
            "Evaluation incomplete - required artifact missing."
        )

    model = _cached_models()[prediction_key]

    # Feature list: prefer the artifact saved next to the model (it is what the
    # existing model scripts wrote), fall back to the live pipeline definition.
    features_path = DATA_MODELS / f"{prediction_key}_features.pkl"
    if features_path.exists():
        features = [str(name) for name in joblib.load(features_path)]
        features_source = str(features_path)
    else:
        features = list(prediction_layer.EXPECTED_FEATURES[prediction_key])
        features_source = "prediction_layer.EXPECTED_FEATURES"

    # Threshold: loaded from the real artifact or the evaluation stops here.
    threshold_path = prediction_layer.THRESHOLD_PATHS.get(prediction_key)

    if threshold_path is None or not threshold_path.exists():
        raise FileNotFoundError(
            f"[{disease}] Required threshold artifact is missing: "
            f"{threshold_path}. Refusing to substitute a threshold "
            "(0.5 is NOT used). Evaluation incomplete - required artifact "
            "missing."
        )

    try:
        threshold = float(_cached_thresholds()[prediction_key])
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"[{disease}] Threshold artifact could not be loaded: {error}. "
            "Refusing to substitute a threshold (0.5 is NOT used)."
        ) from error

    classes = [int(label) for label in getattr(model, "classes_", [])]
    positive_label = classes[-1] if classes else POSITIVE_LABEL

    if classes and classes != [NEGATIVE_LABEL, POSITIVE_LABEL]:
        warnings.warn(
            f"[{disease}] Unexpected class labels {classes} in the frozen "
            f"model; assuming the last label ({positive_label}) is the "
            "positive/risk class. Verified project evidence says class 1 is "
            "the risk class.",
            RuntimeWarning,
            stacklevel=2,
        )

    n_features_expected = getattr(model, "n_features_in_", len(features))

    if int(n_features_expected) != len(features):
        raise ValueError(
            f"[{disease}] Feature mismatch: model expects "
            f"{n_features_expected} features but the feature list has "
            f"{len(features)} ({features})."
        )

    return {
        "disease": disease,
        "prediction_key": prediction_key,
        "model": model,
        "model_type": type(model).__name__,
        "model_path": str(model_path),
        "n_estimators": getattr(model, "n_estimators", None),
        "features": features,
        "features_source": features_source,
        "n_features": len(features),
        "threshold": threshold,
        "threshold_path": str(threshold_path),
        "classes": classes,
        "positive_label": positive_label,
        "negative_label": (
            classes[0] if classes else NEGATIVE_LABEL
        ),
        "positive_class_evidence": POSITIVE_CLASS_EVIDENCE,
        "transformers": _cached_transformers().get(prediction_key, {}),
    }


# ============================================================
# 5. EVALUATION DATASET
# ============================================================


def _reconstruct_raw_test_split(cfg: DiseaseConfig) -> Dict[str, Any]:
    """Re-derive the held-out rows from the repository's raw dataset.

    The split parameters are the ones the existing preprocessing scripts used
    (``test_size=0.20, random_state=42, stratify=y``). Nothing is re-fitted here:
    the rows that were held out are simply recovered, so the LIVE preprocessing
    pipeline can be applied to raw-unit inputs.
    """

    raw_path = DATA_RAW / cfg.raw_file

    if not raw_path.exists():
        raise FileNotFoundError(
            f"[{cfg.key}] Raw dataset is missing: {raw_path}. "
            "Evaluation incomplete - required artifact missing."
        )

    frame = pd.read_csv(raw_path)

    duplicates_dropped = 0

    if cfg.raw_drop_duplicates:
        duplicates_dropped = int(frame.duplicated().sum())
        if duplicates_dropped:
            frame = frame.drop_duplicates()

    if cfg.raw_age_divisor:
        frame["age"] = frame["age"] / cfg.raw_age_divisor

    features = list(prediction_layer.EXPECTED_FEATURES[cfg.prediction_key])

    missing_columns = [
        column for column in features if column not in frame.columns
    ]

    if missing_columns:
        raise ValueError(
            f"[{cfg.key}] Raw dataset {raw_path} is missing expected feature "
            f"columns: {missing_columns}"
        )

    if cfg.target_column not in frame.columns:
        raise ValueError(
            f"[{cfg.key}] Raw dataset {raw_path} has no target column "
            f"'{cfg.target_column}'."
        )

    X = frame[features].copy()
    y = frame[cfg.target_column].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    return {
        "raw_path": str(raw_path),
        "raw_rows": int(len(frame)),
        "duplicates_dropped": duplicates_dropped,
        "features": features,
        "X_test": X_test.reset_index(drop=True),
        "y_test": y_test.reset_index(drop=True).astype(int),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
    }


def _load_stored_test_set(
    cfg: DiseaseConfig,
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, str]]:
    """Read the held-out test files that already exist in the repository."""

    x_path = DATA_PROCESSED / f"{cfg.file_prefix}_X_test.csv"
    y_path = DATA_PROCESSED / f"{cfg.file_prefix}_y_test.csv"

    for path in (x_path, y_path):
        if not path.exists():
            raise FileNotFoundError(
                f"[{cfg.key}] Held-out test file is missing: {path}. "
                "Evaluation incomplete - required artifact missing."
            )

    X_test = pd.read_csv(x_path)
    y_test = pd.read_csv(y_path).squeeze().astype(int)

    if X_test.isna().any().any():
        raise ValueError(
            f"[{cfg.key}] {x_path} contains missing values. The frozen model "
            "cannot consume them, and the evaluation must not apply a different "
            "imputation than the live pipeline."
        )

    if len(X_test) != len(y_test):
        raise ValueError(
            f"[{cfg.key}] Row mismatch between {x_path} ({len(X_test)}) and "
            f"{y_path} ({len(y_test)})."
        )

    return X_test, y_test, {"x_path": str(x_path), "y_path": str(y_path)}


def _compare_non_missing(
    raw_frame: pd.DataFrame,
    processed_frame: pd.DataFrame,
) -> Dict[str, Any]:
    """Compare two frames position by position, ignoring raw missing values.

    Used for diseases whose held-out file was imputed by a script that is no
    longer present. It proves row order and all observed values are identical.
    """

    raw = raw_frame.to_numpy(dtype=float)
    processed = processed_frame.to_numpy(dtype=float)

    if raw.shape != processed.shape:
        return {
            "comparable": False,
            "reason": f"shape mismatch {raw.shape} vs {processed.shape}",
        }

    observed = ~np.isnan(raw)

    differences = np.abs(raw[observed] - processed[observed])

    return {
        "comparable": True,
        "n_raw_missing_in_test": int(np.isnan(raw).sum()),
        "max_abs_diff_where_observed": (
            float(differences.max()) if differences.size else 0.0
        ),
        "observed_values_identical": bool(
            differences.size == 0 or differences.max() <= TOLERANCE
        ),
    }


def load_evaluation_dataset(
    disease: str,
    *,
    verify: bool = True,
) -> Dict[str, Any]:
    """Load the held-out evaluation data for one disease.

    ``X_model_input`` is returned in the exact form the frozen model consumes:

    * diseases with a transformer -> the recovered raw held-out rows pushed
      through the LIVE pipeline (``validate_input`` + ``apply_transformers``),
      verified element-wise against ``data/processed/<prefix>_X_test.csv``.
      The stored processed file is already scaled, so feeding it through the
      scaler again would scale twice; this function prevents that mistake.
    * diseases without a transformer (hypertension) -> the stored processed file
      itself, because the live pipeline applies no transformation either. That
      file was additionally verified against the raw split: row order and every
      observed value are identical; only the 103 missing entries were imputed
      before saving by a script that is no longer in the repository.

    ``X_display`` holds the same rows in evaluation units for the error-analysis
    tables. Only model features are ever included - no identifier column.
    """

    cfg = get_disease_config(disease)
    artifacts = load_model_artifacts(disease)

    stored_X, stored_y, stored_paths = _load_stored_test_set(cfg)

    features = list(artifacts["features"])
    transformers = {"cardio": {}, "diabetes": {}, "hypertension": {}}
    transformers.update(_cached_transformers())
    group = transformers.get(cfg.prediction_key, {})
    has_transformer = bool(group)

    raw_split: Optional[Dict[str, Any]] = None

    try:
        raw_split = _reconstruct_raw_test_split(cfg)
    except FileNotFoundError as error:
        if has_transformer and verify:
            raise
        warnings.warn(str(error), RuntimeWarning, stacklevel=2)

    verification: Dict[str, Any] = {
        "mode": None,
        "verified": False,
        "max_abs_diff_vs_processed": None,
        "y_test_reproduced": None,
        "non_missing_comparison": None,
        "split": {
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "stratify": True,
        },
        "live_pipeline": (
            "prediction_layer.validate_input + "
            "prediction_layer.apply_transformers"
        ),
    }

    if has_transformer:

        if raw_split is None:
            raise RuntimeError(
                f"[{disease}] Raw dataset could not be recovered, so the live "
                "preprocessing pipeline cannot be applied to the held-out rows."
            )

        live_input = prediction_layer.apply_transformers(
            raw_split["X_test"].copy(),
            cfg.prediction_key,
            transformers,
        )
        live_input = live_input[features].reset_index(drop=True)

        difference = float(
            np.max(
                np.abs(
                    live_input.to_numpy(dtype=float)
                    - stored_X[features].to_numpy(dtype=float)
                )
            )
        )

        y_reproduced = bool(
            np.array_equal(raw_split["y_test"].to_numpy(), stored_y.to_numpy())
        )

        verification.update(
            mode=(
                "raw held-out rows -> validate_input -> apply_transformers "
                "(live pipeline)"
            ),
            verified=difference <= TOLERANCE,
            max_abs_diff_vs_processed=difference,
            y_test_reproduced=y_reproduced,
        )

        if verify and not verification["verified"]:
            raise ValueError(
                f"[{disease}] Reconstructing the held-out features with the live "
                f"pipeline does not match {stored_paths['x_path']} "
                f"(max abs diff {difference:.3e}). Refusing to evaluate: the "
                "evaluation must use exactly the preprocessing of the running "
                "application. Re-run "
                "`python src/preprocessing/export_transformers.py` to check the "
                "saved transformers."
            )

        if verify and not y_reproduced:
            raise ValueError(
                f"[{disease}] The held-out labels could not be reproduced from "
                f"the raw dataset with test_size={TEST_SIZE}, "
                f"random_state={RANDOM_STATE}, stratify=y. Refusing to evaluate "
                "against a different split."
            )

        X_model_input = live_input
        X_display = raw_split["X_test"][features].reset_index(drop=True)
        dataset_source = (
            f"{cfg.raw_file} (held-out {TEST_SIZE:.0%} split) -> live pipeline; "
            f"verified against {stored_paths['x_path']}"
        )

    else:

        X_model_input = stored_X[features].reset_index(drop=True)
        X_display = X_model_input.copy()

        comparison = None
        y_reproduced = None

        if raw_split is not None:
            comparison = _compare_non_missing(
                raw_split["X_test"], stored_X[features]
            )
            y_reproduced = bool(
                np.array_equal(
                    raw_split["y_test"].to_numpy(), stored_y.to_numpy()
                )
            )

        verification.update(
            mode=(
                "stored processed file (the live pipeline applies no transformer "
                "for this disease)"
            ),
            verified=True,
            non_missing_comparison=comparison,
            y_test_reproduced=y_reproduced,
        )

        dataset_source = (
            f"{stored_paths['x_path']} (raw units; no transformer in the live "
            "pipeline)"
        )

    return {
        "disease": disease,
        "config": asdict(cfg),
        "X_model_input": X_model_input,
        "X_display": X_display,
        "y": stored_y.reset_index(drop=True),
        "n_samples": int(len(stored_y)),
        "n_features": int(X_model_input.shape[1]),
        "features": features,
        "has_transformer": has_transformer,
        "dataset_source": dataset_source,
        "stored_paths": stored_paths,
        "raw_split": raw_split,
        "verification": verification,
        "artifacts": artifacts,
    }


def predict_for_evaluation(
    X_model_input: pd.DataFrame,
    disease: str,
    artifacts: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """Return positive-class probabilities for the whole held-out split.

    This is the vectorized form of the live call in
    ``prediction_layer.predict_disease`` (``model.predict_proba(X)[0, 1]``): the
    same frozen model and the same positive-class column, applied to every row.
    ``tests/test_model_evaluation.py`` proves the two agree row by row.

    ``X_model_input`` must already be in the exact form produced by
    :func:`load_evaluation_dataset`; applying the transformers a second time
    would standardize already-standardized values.
    """

    if artifacts is None:
        artifacts = load_model_artifacts(disease)

    cfg = get_disease_config(disease)

    validated = prediction_layer.validate_input(
        X_model_input.copy(),
        cfg.prediction_key,
    )

    probabilities = artifacts["model"].predict_proba(validated)[:, 1]

    return np.asarray(probabilities, dtype=float)


def predict_labels(
    probabilities: Sequence[float],
    threshold: float,
) -> np.ndarray:
    """Apply a model threshold exactly like the live pipeline does.

    ``prediction_layer.predict_disease`` uses
    ``prediction = int(probability >= threshold)``; the same comparison is used
    here so the evaluation matches the application for every row.
    """

    return (
        np.asarray(probabilities, dtype=float) >= float(threshold)
    ).astype(int)


# ============================================================
# 6. THRESHOLD-BASED METRICS
# ============================================================


def _safe_rate(numerator: int, denominator: int) -> Optional[float]:
    """Return numerator/denominator or None when it is undefined."""

    if denominator <= 0:
        return None

    return float(numerator) / float(denominator)


def calculate_classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    positive_label: int = POSITIVE_LABEL,
    threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """Classification metrics at the model's own decision threshold.

    Positive/negative roles come from ``positive_label`` - never assumed, see
    section 2 of this module.

    Returns a JSON-friendly dictionary. Unusable input never raises a wrong
    number: ``available`` becomes ``False`` and ``status`` explains why.
    """

    true = np.asarray(y_true).astype(int).ravel()
    predicted = np.asarray(y_pred).astype(int).ravel()

    positive = int(positive_label)
    negative = int(NEGATIVE_LABEL) if positive != NEGATIVE_LABEL else 1

    result: Dict[str, Any] = {
        "status": "ok",
        "available": True,
        "n_samples": int(true.size),
        "positive_label": positive,
        "negative_label": negative,
        "threshold": threshold,
        "accuracy": None,
        "macro_f1": None,
        "positive_precision": None,
        "positive_recall": None,
        "positive_f1": None,
        "per_class": {},
        "confusion_matrix": None,
        "counts": {
            "true_negative": None,
            "false_positive": None,
            "false_negative": None,
            "true_positive": None,
        },
        "rates": {
            "false_positive_rate": None,
            "false_negative_rate": None,
            "true_positive_rate": None,
            "true_negative_rate": None,
            "positive_prediction_rate": None,
            "prevalence": None,
        },
    }

    def unavailable(reason: str) -> Dict[str, Any]:
        result["status"] = f"unavailable: {reason}"
        result["available"] = False
        return result

    if true.size == 0:
        return unavailable("the evaluation set contains no samples")

    if true.shape != predicted.shape:
        return unavailable(
            "y_true and y_pred have different lengths "
            f"({true.size} vs {predicted.size})"
        )

    observed_labels = set(np.unique(true).tolist()) | set(
        np.unique(predicted).tolist()
    )
    unknown_labels = sorted(observed_labels - {negative, positive})

    if unknown_labels:
        return unavailable(
            f"unexpected labels {unknown_labels}; expected {negative} and "
            f"{positive}"
        )

    labels = [negative, positive]

    matrix = confusion_matrix(true, predicted, labels=labels)
    true_negative, false_positive, false_negative, true_positive = (
        int(value) for value in matrix.ravel()
    )

    precision, recall, f1, support = precision_recall_fscore_support(
        true,
        predicted,
        labels=labels,
        zero_division=0,
    )

    result["accuracy"] = float(accuracy_score(true, predicted))
    result["macro_f1"] = float(
        f1_score(true, predicted, labels=labels, average="macro", zero_division=0)
    )

    result["positive_precision"] = float(precision[1])
    result["positive_recall"] = float(recall[1])
    result["positive_f1"] = float(f1[1])

    result["per_class"] = {
        f"class_{negative}": {
            "label": negative,
            "role": "negative",
            "meaning": "negative class",
            "precision": float(precision[0]),
            "recall": float(recall[0]),
            "f1": float(f1[0]),
            "support": int(support[0]),
        },
        f"class_{positive}": {
            "label": positive,
            "role": "positive",
            "meaning": "positive / risk class",
            "precision": float(precision[1]),
            "recall": float(recall[1]),
            "f1": float(f1[1]),
            "support": int(support[1]),
        },
    }

    result["confusion_matrix"] = {
        "labels": labels,
        "matrix": [
            [true_negative, false_positive],
            [false_negative, true_positive],
        ],
        "row_semantics": "rows = actual class, columns = predicted class",
    }

    result["counts"] = {
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_positive": true_positive,
    }

    result["rates"] = {
        "false_positive_rate": _safe_rate(
            false_positive, false_positive + true_negative
        ),
        "false_negative_rate": _safe_rate(
            false_negative, false_negative + true_positive
        ),
        "true_positive_rate": _safe_rate(
            true_positive, true_positive + false_negative
        ),
        "true_negative_rate": _safe_rate(
            true_negative, true_negative + false_positive
        ),
        "positive_prediction_rate": _safe_rate(
            true_positive + false_positive, int(true.size)
        ),
        "prevalence": _safe_rate(
            true_positive + false_negative, int(true.size)
        ),
    }

    return result


# ============================================================
# 7. PROBABILITY-BASED METRICS
# ============================================================


def _prepare_probability_arrays(
    y_true: Sequence[int],
    y_prob: Sequence[float],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict[str, Any]]:
    """Validate labels/probabilities and drop unusable probability values.

    Invalid values (non-finite or outside [0, 1]) are never used to produce a
    metric: they are counted, warned about (not silently swallowed) and excluded.
    """

    true = np.asarray(y_true).astype(int).ravel()
    probability = np.asarray(y_prob, dtype=float).ravel()

    info: Dict[str, Any] = {
        "reason": None,
        "warning": None,
        "n_samples": int(true.size),
        "n_valid_probabilities": 0,
        "n_invalid_probabilities": 0,
    }

    if true.size == 0 or probability.size == 0:
        info["reason"] = "the evaluation set contains no samples"
        return None, None, info

    if true.shape != probability.shape:
        info["reason"] = (
            "y_true and y_prob have different lengths "
            f"({true.size} vs {probability.size})"
        )
        return None, None, info

    valid = np.isfinite(probability) & (probability >= 0.0) & (probability <= 1.0)

    info["n_invalid_probabilities"] = int((~valid).sum())
    info["n_valid_probabilities"] = int(valid.sum())

    if info["n_invalid_probabilities"]:

        warnings.warn(
            f"{info['n_invalid_probabilities']} predicted probability value(s) "
            "are non-finite or outside [0, 1] and were excluded from the "
            "probability metrics.",
            RuntimeWarning,
            stacklevel=2,
        )

        info["warning"] = (
            f"{info['n_invalid_probabilities']} invalid probability value(s) "
            "excluded"
        )

    if info["n_valid_probabilities"] == 0:
        info["reason"] = (
            "every predicted probability is invalid (non-finite or outside "
            "[0, 1])"
        )
        return None, None, info

    return true[valid], probability[valid], info


def calculate_probability_metrics(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    positive_label: int = POSITIVE_LABEL,
) -> Dict[str, Any]:
    """Probability metrics: AUROC and Brier score.

    AUROC uses the probability scores (never the thresholded predictions) and is
    reported as explicitly unavailable when ``y_true`` contains only one class.
    The Brier score always uses the positive-class probability, never the binary
    prediction.
    """

    result: Dict[str, Any] = {
        "status": "ok",
        "available": True,
        "n_samples": 0,
        "n_valid_probabilities": 0,
        "n_invalid_probabilities": 0,
        "positive_label": int(positive_label),
        "classes_present": [],
        "auroc": None,
        "auroc_status": "ok",
        "brier_score": None,
        "brier_score_status": "ok",
        "mean_predicted_probability": None,
        "observed_positive_rate": None,
        "uses": "positive-class predicted probability",
    }

    true, probability, info = _prepare_probability_arrays(y_true, y_prob)

    result.update(
        n_samples=info["n_samples"],
        n_valid_probabilities=info["n_valid_probabilities"],
        n_invalid_probabilities=info["n_invalid_probabilities"],
    )

    if info["reason"] is not None:

        result["available"] = False
        result["status"] = f"unavailable: {info['reason']}"
        result["auroc_status"] = f"unavailable: {info['reason']}"
        result["brier_score_status"] = f"unavailable: {info['reason']}"

        return result

    if info["warning"]:
        result["status"] = f"ok with exclusions: {info['warning']}"

    classes_present = sorted(int(value) for value in np.unique(true))
    result["classes_present"] = classes_present

    result["mean_predicted_probability"] = float(probability.mean())
    result["observed_positive_rate"] = float(true.mean())

    # --------------------------------------------------------
    # AUROC
    # --------------------------------------------------------

    if len(classes_present) < 2:

        result["auroc"] = None
        result["auroc_status"] = (
            "unavailable: y_true contains only one class "
            f"({classes_present[0] if classes_present else 'n/a'}). AUROC needs "
            "at least one positive and one negative sample, so ranking quality "
            "cannot be measured on this data."
        )

    else:

        result["auroc"] = float(roc_auc_score(true, probability))
        result["auroc_status"] = "ok"

    # --------------------------------------------------------
    # BRIER SCORE
    # --------------------------------------------------------

    try:

        brier_kwargs: Dict[str, Any] = {}
        if _BRIER_HAS_POS_LABEL:
            brier_kwargs["pos_label"] = int(positive_label)

        result["brier_score"] = float(
            brier_score_loss(true, probability, **brier_kwargs)
        )
        result["brier_score_status"] = "ok"

    except ValueError as error:

        result["brier_score"] = None
        result["brier_score_status"] = (
            f"unavailable: {error} (labels must be 0/1 to score probabilities)"
        )

    return result


def calculate_roc_curve(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    positive_label: int = POSITIVE_LABEL,
) -> Dict[str, Any]:
    """ROC curve points from the predicted probabilities.

    Uses ``sklearn.metrics.roc_curve`` with the default
    ``drop_intermediate=True`` (collinear points are removed, the curve stays
    identical). When only one class is present the curve is reported as
    unavailable with an explanation instead of raising.
    """

    result: Dict[str, Any] = {
        "status": "ok",
        "available": True,
        "n_samples": 0,
        "n_valid_probabilities": 0,
        "n_invalid_probabilities": 0,
        "n_points": 0,
        "fpr": [],
        "tpr": [],
        "thresholds": [],
        "auroc": None,
        "uses": "positive-class predicted probability",
        "note": (
            "sklearn.metrics.roc_curve, drop_intermediate=True "
            "(collinear points removed; curve shape preserved)"
        ),
    }

    true, probability, info = _prepare_probability_arrays(y_true, y_prob)

    result.update(
        n_samples=info["n_samples"],
        n_valid_probabilities=info["n_valid_probabilities"],
        n_invalid_probabilities=info["n_invalid_probabilities"],
    )

    if info["reason"] is not None:
        result["available"] = False
        result["status"] = f"unavailable: {info['reason']}"
        return result

    classes_present = sorted(int(value) for value in np.unique(true))

    if len(classes_present) < 2:
        result["available"] = False
        result["status"] = (
            "unavailable: y_true contains only one class "
            f"({classes_present[0] if classes_present else 'n/a'}). A ROC curve "
            "needs both a positive and a negative class."
        )
        return result

    false_positive_rate, true_positive_rate, thresholds = roc_curve(
        true,
        probability,
        pos_label=int(positive_label),
        drop_intermediate=True,
    )

    result.update(
        fpr=[float(value) for value in false_positive_rate],
        tpr=[float(value) for value in true_positive_rate],
        thresholds=[float(value) for value in thresholds],
        n_points=int(len(false_positive_rate)),
        auroc=float(roc_auc_score(true, probability)),
    )

    return result


def calculate_calibration_metrics(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    n_bins: int = N_CALIBRATION_BINS,
    strategy: str = CALIBRATION_STRATEGY,
    positive_label: int = POSITIVE_LABEL,
) -> Dict[str, Any]:
    """Reliability table: mean predicted probability vs observed frequency.

    ``sklearn.calibration_curve`` produces the calibration curve. Because it
    omits empty bins, the per-bin sample counts are computed from the same
    uniform edges (``np.linspace(0, 1, n_bins + 1)``) and cross-checked against
    sklearn's values; a disagreement raises instead of producing mis-aligned
    bins.

    Calibration describes agreement between predicted probability and observed
    frequency on this held-out sample only. It is not evidence of clinical
    validity.
    """

    bins_count = int(n_bins)

    result: Dict[str, Any] = {
        "status": "ok",
        "available": True,
        "n_samples": 0,
        "n_valid_probabilities": 0,
        "n_invalid_probabilities": 0,
        "n_bins": bins_count,
        "strategy": str(strategy),
        "positive_label": int(positive_label),
        "bins": [],
        "expected_calibration_error": None,
        "mean_predicted_probability": None,
        "observed_positive_rate": None,
        "bin_sample_counts_available": str(strategy) == "uniform",
        "interpretation_note_vi": (
            "Đường hiệu chuẩn chỉ cho biết mức độ khớp giữa xác suất dự đoán và "
            "tần suất thực tế trên tập kiểm thử này. Nó KHÔNG chứng minh giá trị "
            "lâm sàng."
        ),
    }

    if bins_count < 2:
        result["available"] = False
        result["status"] = "unavailable: n_bins must be at least 2"
        return result

    true, probability, info = _prepare_probability_arrays(y_true, y_prob)

    result.update(
        n_samples=info["n_samples"],
        n_valid_probabilities=info["n_valid_probabilities"],
        n_invalid_probabilities=info["n_invalid_probabilities"],
    )

    if info["reason"] is not None:
        result["available"] = False
        result["status"] = f"unavailable: {info['reason']}"
        return result

    classes_present = sorted(int(value) for value in np.unique(true))

    if len(classes_present) < 2:
        result["available"] = False
        result["status"] = (
            "unavailable: y_true contains only one class "
            f"({classes_present[0] if classes_present else 'n/a'}). The observed "
            "positive frequency would be constant, so a reliability diagram "
            "would not describe the data."
        )
        return result

    curve_kwargs: Dict[str, Any] = {
        "n_bins": bins_count,
        "strategy": str(strategy),
    }

    if _CALIBRATION_HAS_POS_LABEL:
        curve_kwargs["pos_label"] = int(positive_label)

    observed_frequency, mean_predicted = calibration_curve(
        true,
        probability,
        **curve_kwargs,
    )

    result["mean_predicted_probability"] = float(probability.mean())
    result["observed_positive_rate"] = float(true.mean())

    edges = np.linspace(0.0, 1.0, bins_count + 1)
    bin_ids = np.searchsorted(edges[1:-1], probability)
    sample_counts = np.bincount(bin_ids, minlength=bins_count)

    bins: List[Dict[str, Any]] = []

    if str(strategy) == "uniform":

        # Per-bin values recomputed from the same bin assignment sklearn uses,
        # then cross-checked so the persisted bins cannot be misaligned.
        per_bin_sum = np.bincount(
            bin_ids, weights=probability, minlength=bins_count
        )
        per_bin_positives = np.bincount(
            bin_ids, weights=true.astype(float), minlength=bins_count
        )

        non_empty = sample_counts > 0

        own_mean = per_bin_sum[non_empty] / sample_counts[non_empty]
        own_frequency = per_bin_positives[non_empty] / sample_counts[non_empty]

        if (
            int(non_empty.sum()) != len(mean_predicted)
            or not np.allclose(own_mean, mean_predicted, atol=1e-9)
            or not np.allclose(own_frequency, observed_frequency, atol=1e-9)
        ):
            raise ValueError(
                "Calibration bin alignment check failed: per-bin values computed "
                "from the uniform bin edges disagree with "
                "sklearn.calibration_curve. Refusing to persist mis-aligned "
                "reliability data."
            )

        for position, bin_index in enumerate(np.flatnonzero(non_empty)):

            gap = abs(
                float(observed_frequency[position])
                - float(mean_predicted[position])
            )

            bins.append(
                {
                    "bin": int(bin_index) + 1,
                    "bin_lower": float(edges[bin_index]),
                    "bin_upper": float(edges[bin_index + 1]),
                    "sample_count": int(sample_counts[bin_index]),
                    "mean_predicted_probability": float(
                        mean_predicted[position]
                    ),
                    "observed_positive_frequency": float(
                        observed_frequency[position]
                    ),
                    "absolute_gap": gap,
                }
            )

        total = float(sample_counts.sum())

        if total > 0:
            result["expected_calibration_error"] = float(
                sum(
                    (item["sample_count"] / total) * item["absolute_gap"]
                    for item in bins
                )
            )

    else:

        bins = [
            {
                "bin": int(position) + 1,
                "bin_lower": None,
                "bin_upper": None,
                "sample_count": None,
                "mean_predicted_probability": float(mean_value),
                "observed_positive_frequency": float(frequency),
                "absolute_gap": abs(float(frequency) - float(mean_value)),
            }
            for position, (frequency, mean_value) in enumerate(
                zip(observed_frequency, mean_predicted)
            )
        ]

    result["bins"] = bins

    return result


# ============================================================
# 8. ERROR ANALYSIS
# ============================================================


def _strip_identifier_columns(
    frame: pd.DataFrame,
) -> Tuple[pd.DataFrame, List[str]]:
    """Drop any identifier-like column before data is displayed or persisted.

    The evaluation frames are built from model features only, so normally
    nothing is dropped. The check is defensive: no name, phone number, medical
    record number or similar column may reach an error-analysis table.
    """

    def looks_like_identifier(column: str) -> bool:

        name = str(column).strip().lower()

        if name in {"age", "bmi", "male", "gender", "diabetes"}:
            return False

        return any(pattern in name for pattern in IDENTIFIER_PATTERNS)

    flagged = [column for column in frame.columns if looks_like_identifier(column)]

    if not flagged:
        return frame, []

    cleaned = frame.drop(columns=flagged)

    warnings.warn(
        f"Identifier-like columns dropped from the evaluation tables: {flagged}",
        RuntimeWarning,
        stacklevel=2,
    )

    return cleaned, flagged


def analyze_prediction_errors(
    X_display: pd.DataFrame,
    y_true: Sequence[int],
    y_prob: Sequence[float],
    y_pred: Sequence[int],
    positive_label: int = POSITIVE_LABEL,
    negative_label: int = NEGATIVE_LABEL,
    threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """Split the held-out rows into errors for manual inspection.

    * False positives: actual negative, predicted positive.
    * False negatives: actual positive, predicted negative - reported as
      "missed positive-risk cases" because that is what the label means, without
      claiming any clinical consequence that the repository has not established.

    The record tables hold: evaluation row index, model feature values in
    evaluation units, predicted probability, the model threshold, the binary
    model prediction and the risk level the application would display.
    """

    true = np.asarray(y_true).astype(int).ravel()
    probability = np.asarray(y_prob, dtype=float).ravel()
    predicted = np.asarray(y_pred).astype(int).ravel()

    positive = int(positive_label)
    negative = int(negative_label)

    result: Dict[str, Any] = {
        "status": "ok",
        "available": True,
        "n_samples": int(true.size),
        "positive_label": positive,
        "negative_label": negative,
        "threshold": threshold,
        "false_positive_count": None,
        "false_negative_count": None,
        "true_positive_count": None,
        "true_negative_count": None,
        "actual_positive_count": None,
        "actual_negative_count": None,
        "false_positive_rate": None,
        "false_negative_rate": None,
        "false_positive_share_of_all": None,
        "false_negative_share_of_all": None,
        "group_probability_summary": {},
        "dropped_identifier_columns": [],
        "false_positive_records": pd.DataFrame(),
        "false_negative_records": pd.DataFrame(),
        "labels_vi": {
            "false_positive": "Dương tính giả (False Positives)",
            "false_negative": (
                "Âm tính giả / Bỏ sót ca có nguy cơ "
                "(False Negatives / Missed Positive-Risk Cases)"
            ),
        },
        "screening_note_vi": (
            "Các ca âm tính giả là những dòng thực tế thuộc nhóm nguy cơ nhưng mô "
            "hình không báo nguy cơ trên tập kiểm thử. Con số này mô tả hành vi "
            "của mô hình trên dữ liệu kiểm thử; repository không có bằng chứng về "
            "hậu quả lâm sàng thực tế, và đây không phải công cụ chẩn đoán."
        ),
    }

    def unavailable(reason: str) -> Dict[str, Any]:
        result["status"] = f"unavailable: {reason}"
        result["available"] = False
        return result

    if true.size == 0:
        return unavailable("the evaluation set contains no samples")

    if not (true.shape == probability.shape == predicted.shape):
        return unavailable(
            "labels, probabilities and predictions have different lengths "
            f"({true.size}, {probability.size}, {predicted.size})"
        )

    if len(X_display) != true.size:
        return unavailable(
            "the feature frame and the labels have different lengths "
            f"({len(X_display)} vs {true.size})"
        )

    safe_frame, dropped_columns = _strip_identifier_columns(
        X_display.reset_index(drop=True)
    )
    result["dropped_identifier_columns"] = dropped_columns

    risk_levels = np.array(
        [
            (
                classify_risk(float(value))
                if np.isfinite(value) and 0.0 <= float(value) <= 1.0
                else "UNKNOWN"
            )
            for value in probability
        ],
        dtype=object,
    )

    false_positive_mask = (true == negative) & (predicted == positive)
    false_negative_mask = (true == positive) & (predicted == negative)
    true_positive_mask = (true == positive) & (predicted == positive)
    true_negative_mask = (true == negative) & (predicted == negative)

    false_positive_count = int(false_positive_mask.sum())
    false_negative_count = int(false_negative_mask.sum())

    actual_positive_count = int((true == positive).sum())
    actual_negative_count = int((true == negative).sum())
    n_samples = int(true.size)

    result.update(
        false_positive_count=false_positive_count,
        false_negative_count=false_negative_count,
        true_positive_count=int(true_positive_mask.sum()),
        true_negative_count=int(true_negative_mask.sum()),
        actual_positive_count=actual_positive_count,
        actual_negative_count=actual_negative_count,
        false_positive_rate=_safe_rate(
            false_positive_count, actual_negative_count
        ),
        false_negative_rate=_safe_rate(
            false_negative_count, actual_positive_count
        ),
        false_positive_share_of_all=_safe_rate(
            false_positive_count, n_samples
        ),
        false_negative_share_of_all=_safe_rate(
            false_negative_count, n_samples
        ),
    )

    result["group_probability_summary"] = {
        "true_negative_mean_probability": _mean_or_none(
            probability[true_negative_mask]
        ),
        "false_positive_mean_probability": _mean_or_none(
            probability[false_positive_mask]
        ),
        "false_negative_mean_probability": _mean_or_none(
            probability[false_negative_mask]
        ),
        "true_positive_mean_probability": _mean_or_none(
            probability[true_positive_mask]
        ),
    }

    result["false_positive_records"] = _build_error_records(
        safe_frame,
        false_positive_mask,
        probability,
        threshold,
        predicted,
        risk_levels,
    )

    result["false_negative_records"] = _build_error_records(
        safe_frame,
        false_negative_mask,
        probability,
        threshold,
        predicted,
        risk_levels,
    )

    return result


def _mean_or_none(values: Sequence[float]) -> Optional[float]:
    """Mean of the finite values, or None when there is nothing to average."""

    array = np.asarray(values, dtype=float).ravel()
    finite = array[np.isfinite(array)]

    if finite.size == 0:
        return None

    return float(finite.mean())


def _build_error_records(
    frame: pd.DataFrame,
    mask: np.ndarray,
    probability: np.ndarray,
    threshold: Optional[float],
    predicted: np.ndarray,
    risk_levels: np.ndarray,
) -> pd.DataFrame:
    """Build one inspection table for the rows selected by ``mask``."""

    indices = np.flatnonzero(mask)

    records = frame.iloc[indices].reset_index(drop=True).copy()

    records.insert(0, "evaluation_row_index", indices)
    records.insert(1, "predicted_probability", probability[indices])
    records.insert(2, "model_threshold", threshold)
    records.insert(3, "model_prediction", predicted[indices])
    records.insert(4, "app_risk_level", risk_levels[indices])

    return records


def calculate_decision_engine_view(
    probabilities: Sequence[float],
    y_true: Optional[Sequence[int]] = None,
    model_predictions: Optional[Sequence[int]] = None,
    positive_label: int = POSITIVE_LABEL,
) -> Dict[str, Any]:
    """Risk-level distribution produced by the LIVE Decision Engine cut-offs.

    The application classifies risk with ``decision_engine.classify_risk``
    (LOW below 0.30, MODERATE below 0.70, HIGH at or above 0.70). That rule is
    NOT the same as the model threshold stored in
    ``data/models/<disease>_threshold.pkl`` (0.40 / 0.35 / 0.35). Both views are
    reported side by side so the difference stays visible; the application logic
    is reused, never redefined.
    """

    probability = np.asarray(probabilities, dtype=float).ravel()

    result: Dict[str, Any] = {
        "status": "ok",
        "available": True,
        "n_samples": int(probability.size),
        "cutoffs": {
            "low_below": float(LOW_RISK_THRESHOLD),
            "high_at_or_above": float(HIGH_RISK_THRESHOLD),
            "source": "src/decision_engine/decision_engine.py",
        },
        "distribution": {},
        "distribution_share": {},
        "high_prediction_rate": None,
        "high_as_positive_metrics": None,
        "model_threshold_vs_high_risk_level": None,
        "note_vi": (
            "Decision Engine dùng mốc 0.30 / 0.70 để phân mức nguy cơ, khác với "
            "ngưỡng mô hình lấy từ data/models/*_threshold.pkl. Hai cách nhìn "
            "được báo cáo song song; logic của ứng dụng KHÔNG bị thay đổi."
        ),
    }

    if probability.size == 0:
        result["status"] = "unavailable: the evaluation set contains no samples"
        result["available"] = False
        return result

    levels = np.array(
        [
            (
                classify_risk(float(value))
                if np.isfinite(value) and 0.0 <= float(value) <= 1.0
                else "UNKNOWN"
            )
            for value in probability
        ],
        dtype=object,
    )

    n_samples = int(probability.size)

    distribution = {
        level: int((levels == level).sum())
        for level in ("LOW", "MODERATE", "HIGH", "UNKNOWN")
    }

    result["distribution"] = distribution
    result["distribution_share"] = {
        level: _safe_rate(count, n_samples)
        for level, count in distribution.items()
    }
    result["high_prediction_rate"] = _safe_rate(
        distribution["HIGH"], n_samples
    )

    high_mask = levels == "HIGH"

    if y_true is not None:
        result["high_as_positive_metrics"] = calculate_classification_metrics(
            y_true,
            high_mask.astype(int),
            positive_label=positive_label,
            threshold=float(HIGH_RISK_THRESHOLD),
        )

    if model_predictions is not None:

        predictions = np.asarray(model_predictions).astype(int).ravel()

        if predictions.shape == high_mask.shape:

            model_positive = predictions == 1

            result["model_threshold_vs_high_risk_level"] = {
                "model_positive_count": int(model_positive.sum()),
                "high_risk_level_count": int(high_mask.sum()),
                "rows_where_the_two_rules_disagree": int(
                    (high_mask != model_positive).sum()
                ),
                "disagreement_share": _safe_rate(
                    int((high_mask != model_positive).sum()), n_samples
                ),
            }

    return result


# ============================================================
# 9. DATASET LIMITATIONS
# ============================================================

EVALUATION_DISCLAIMER_VI = (
    "Các chỉ số dưới đây mô tả hành vi của mô hình trên tập dữ liệu kiểm thử "
    "đã được giữ lại. Chúng KHÔNG chứng minh giá trị lâm sàng hay độ chính xác "
    "chẩn đoán. Hệ thống được thiết kế cho mục đích sàng lọc/hỗ trợ nguy cơ, "
    "không phải chẩn đoán y khoa và không thay thế nhân viên y tế có chuyên môn."
)

SCREENING_DISCLAIMER_REFERENCE_VI = (
    "Cảnh báo chung của ứng dụng vẫn giữ nguyên và được hiển thị ở màn hình "
    "đánh giá nguy cơ (nguồn: "
    "src/recommendation/recommendation_engine.py, khóa 'disclaimer')."
)


def collect_dataset_limitations(
    disease: str,
    dataset: Dict[str, Any],
) -> Dict[str, Any]:
    """Collect verified dataset facts and the honest knowledge gaps.

    Every number here is read from the repository's own files. Anything the
    repository does not document is reported as
    "NOT FOUND / NEEDS VERIFICATION" instead of being invented.
    """

    cfg = get_disease_config(disease)

    raw_path = DATA_RAW / cfg.raw_file

    if not raw_path.exists():
        raise FileNotFoundError(
            f"[{disease}] Raw dataset is missing: {raw_path}. "
            "Evaluation incomplete - required artifact missing."
        )

    raw = pd.read_csv(raw_path)
    target = cfg.target_column

    if target not in raw.columns:
        raise ValueError(
            f"[{disease}] Raw dataset {raw_path} has no target column '{target}'."
        )

    raw_class_counts = {
        str(int(label)): int(count)
        for label, count in raw[target].value_counts().sort_index().items()
    }
    raw_total = int(len(raw))
    raw_class_share = {
        label: _safe_rate(count, raw_total)
        for label, count in raw_class_counts.items()
    }

    missing_by_column = {
        str(column): int(count)
        for column, count in raw.isna().sum().items()
        if int(count) > 0
    }

    # Zero values that are not physiologically plausible. The repository's
    # src/data/data_quanlity.py analyses these explicitly.
    zero_columns = [
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
    ]
    zero_value_counts = {
        column: int((raw[column] == 0).sum())
        for column in zero_columns
        if column in raw.columns and int((raw[column] == 0).sum()) > 0
    }

    y_test = dataset["y"]

    test_class_counts = {
        str(int(label)): int(count)
        for label, count in y_test.value_counts().sort_index().items()
    }

    test_total = int(len(y_test))

    processed_test_path = DATA_PROCESSED / f"{cfg.file_prefix}_X_test.csv"
    processed_train_path = DATA_PROCESSED / f"{cfg.file_prefix}_X_train.csv"

    processed_missing = None
    if processed_test_path.exists():
        processed_missing = int(
            pd.read_csv(processed_test_path).isna().sum().sum()
        )

    # ----------------------------
    # Observed demographic coverage
    # ----------------------------

    display_frame = dataset["X_display"]

    demographics: Dict[str, Any] = {}

    for column in cfg.demographic_columns:

        if column not in display_frame.columns:
            demographics[column] = {"available": False}
            continue

        series = display_frame[column]
        n_unique = int(series.nunique(dropna=True))

        if n_unique > 10:
            demographics[column] = {
                "available": True,
                "type": "continuous",
                "n_unique": n_unique,
                "min": float(series.min()),
                "median": float(series.median()),
                "max": float(series.max()),
            }
        else:
            demographics[column] = {
                "available": True,
                "type": "categorical",
                "n_unique": n_unique,
                "counts": {
                    str(int(value)): int(count)
                    for value, count in series.value_counts()
                    .sort_index()
                    .items()
                },
            }

    absent_demographics = [
        name
        for name in (
            "gender",
            "male",
            "sex",
            "race",
            "ethnicity",
            "income",
            "region",
            "education",
        )
        if name not in display_frame.columns
    ]

    representation_notes = [
        (
            "Các trường nhân khẩu học KHÔNG có trong bộ dữ liệu này: "
            + ", ".join(absent_demographics)
            + " => không thể đánh giá mức độ đại diện hay công bằng theo các "
            "nhóm đó."
        ),
        (
            "Không có trường dân tộc, thu nhập, khu vực sinh sống hay trình độ "
            "học vấn trong bất kỳ bộ dữ liệu nào của repository."
        ),
        (
            "Không có thông tin về phương pháp lấy mẫu nên không thể khẳng định "
            "mẫu đại diện cho dân số mục tiêu."
        ),
    ]

    if "gender" not in display_frame.columns and "male" not in display_frame.columns:
        representation_notes.append(
            "Bộ dữ liệu này không có biến giới tính (xác minh từ danh sách cột)."
        )

    external_validation = {
        "exists": False,
        "statement_vi": (
            "KHÔNG có tập kiểm chứng độc lập/bên ngoài trong repository. Toàn bộ "
            "chỉ số được tính trên 20% dữ liệu giữ lại của CHÍNH bộ dữ liệu đã "
            "dùng để huấn luyện (test_size=0.20, random_state=42, stratify=y)."
        ),
        "how_verified": (
            "Đã rà soát data/ (raw, processed) và mã nguồn src/; không tồn tại "
            "tệp dữ liệu kiểm chứng độc lập nào."
        ),
    }

    label_limitations = [
        (
            "Repository không có data dictionary hay tài liệu định nghĩa nhãn => "
            "định nghĩa nhãn: NOT FOUND / NEEDS VERIFICATION."
        ),
        (
            f"Nhãn dùng để đánh giá là cột '{target}', lớp dương = "
            f"{int(dataset['artifacts']['positive_label'])} "
            f"({cfg.positive_meaning_vi}). Ý nghĩa này suy ra từ mã nguồn hiện "
            "có, không phải từ tài liệu dữ liệu."
        ),
    ]

    if disease == "hypertension":
        label_limitations.append(
            "src/data/data_audit.py có mục 'POTENTIAL LABEL RULE' kiểm tra xem "
            "nhãn 'Risk' có được suy ra từ ngưng sysBP hay không; repository "
            "không kết luận => nhãn có thể mang tính quy tắc => NEEDS VERIFICATION."
        )

    if zero_value_counts:
        label_limitations.append(
            "Bộ dữ liệu dùng giá trị 0 thay cho giá trị thiếu ở một số chỉ số; "
            "đây là dữ liệu thiếu không được ghi nhãn và có thể làm nhiễu cả đặc "
            "trưng lẫn nhãn."
        )

    known_limitations_vi = [
        (
            "Tập kiểm thử chỉ là 20% dữ liệu giữ lại của cùng một bộ dữ liệu, "
            "không phải tập kiểm chứng độc lập."
        ),
        (
            "Hiệu năng trên tập kiểm thử KHÔNG đại diện cho hiệu năng thực tế "
            "ngoài dân số của bộ dữ liệu này."
        ),
        (
            "Không xác minh được nguồn gốc/giấy phép dữ liệu nên không thể khẳng "
            "định tính phù hợp cho mục đích lâm sàng."
        ),
        (
            "Dữ liệu thiếu và các giá trị 0 bất thường có thể làm sai lệch chỉ số."
        ),
        (
            "Nhãn có thể chứa sai số; định nghĩa nhãn không được tài liệu hóa."
        ),
        (
            "Thiếu nhiều trường nhân khẩu học nên không thể đánh giá đầy đủ tính "
            "đại diện và công bằng giữa các nhóm."
        ),
    ]

    return {
        "disease": disease,
        "display_name_vi": cfg.display_name_vi,
        "target_column": target,
        "positive_class": int(dataset["artifacts"]["positive_label"]),
        "negative_class": int(dataset["artifacts"]["negative_label"]),
        "positive_meaning_vi": cfg.positive_meaning_vi,
        "negative_meaning_vi": cfg.negative_meaning_vi,
        "source_file": str(raw_path),
        "source_columns": [str(column) for column in raw.columns],
        "dataset_source_vi": (
            f"Bộ dữ liệu thô trong repository: {raw_path.name} "
            f"({raw_total} dòng, {len(raw.columns)} cột)."
        ),
        "dataset_provenance": {
            "status": "NOT FOUND / NEEDS VERIFICATION",
            "detail_vi": (
                "Repository không có tài liệu về nguồn gốc, đơn vị thu thập, "
                "phương pháp lấy mẫu, thời điểm thu thập hay giấy phép của bộ dữ "
                "liệu. Không thể xác minh từ mã nguồn => "
                "NOT FOUND / NEEDS VERIFICATION."
            ),
        },
        "sample_size": {
            "raw_rows": raw_total,
            "raw_columns": int(len(raw.columns)),
            "duplicate_rows": int(raw.duplicated().sum()),
            "train_rows": (
                int(pd.read_csv(processed_train_path).shape[0])
                if processed_train_path.exists()
                else None
            ),
            "test_rows": test_total,
            "feature_count": int(dataset["n_features"]),
        },
        "class_distribution": {
            "raw_counts": raw_class_counts,
            "raw_share": raw_class_share,
            "test_counts": test_class_counts,
            "test_share": {
                label: _safe_rate(count, test_total)
                for label, count in test_class_counts.items()
            },
        },
        "missing_values": {
            "raw_total_missing": int(raw.isna().sum().sum()),
            "raw_missing_by_column": missing_by_column,
            "processed_test_missing": processed_missing,
        },
        "zero_value_counts": zero_value_counts,
        "preprocessing_steps_vi": cfg.preprocessing_vi,
        "transformer_vi": cfg.transformer_vi,
        "dataset_source_used_for_evaluation": dataset["dataset_source"],
        "verification": dataset["verification"],
        "demographic_coverage": demographics,
        "demographics_absent": absent_demographics,
        "representation_limitations_vi": representation_notes,
        "external_validation": external_validation,
        "label_limitations_vi": label_limitations,
        "known_limitations_vi": known_limitations_vi,
        "split_note_vi": (
            "Tập đánh giá là 20% dữ liệu giữ lại (test_size=0.20, "
            "random_state=42, stratify=y). Đây là hold-out nội bộ, không phải "
            "kiểm chứng độc lập."
        ),
        "evaluation_disclaimer_vi": EVALUATION_DISCLAIMER_VI,
        "screening_disclaimer_reference_vi": SCREENING_DISCLAIMER_REFERENCE_VI,
    }


# ============================================================
# 10. PLOTS
# ============================================================

PLOT_ACCENT = "#0B6E75"
PLOT_TEXT = "#101E27"


def _save_figure(figure: Any, output_path: Any) -> str:
    """Write a matplotlib figure to disk and close it."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)

    return str(path)


def generate_reliability_diagram(
    calibration: Dict[str, Any],
    output_path: Any,
    *,
    disease: Optional[str] = None,
    title: Optional[str] = None,
) -> Optional[str]:
    """Save a reliability (calibration) diagram.

    Shows the perfect-calibration reference line, the model curve, the mean
    predicted probability on the x-axis, the observed positive fraction on the
    y-axis, the sample count per bin and a note that calibration is not clinical
    validity.

    Returns the written path, or ``None`` when calibration is unavailable.
    """

    if not calibration.get("available") or not calibration.get("bins"):

        warnings.warn(
            f"[{disease or 'unknown'}] Reliability diagram skipped: "
            f"{calibration.get('status', 'no calibration data')}",
            RuntimeWarning,
            stacklevel=2,
        )

        return None

    bins = calibration["bins"]

    mean_predicted = [item["mean_predicted_probability"] for item in bins]
    observed = [item["observed_positive_frequency"] for item in bins]
    counts = [item.get("sample_count") or 0 for item in bins]

    figure, axes = plt.subplots(figsize=(7.0, 6.2))

    axes.plot(
        [0.0, 1.0],
        [0.0, 1.0],
        linestyle="--",
        color="#9AA5B1",
        label="Hiệu chuẩn hoàn hảo (perfect calibration)",
    )

    axes.plot(
        mean_predicted,
        observed,
        marker="o",
        markersize=6,
        color=PLOT_ACCENT,
        label="Mô hình (model)",
    )

    for x_value, y_value, count in zip(mean_predicted, observed, counts):
        if count:
            axes.annotate(
                f"n={count}",
                (x_value, y_value),
                textcoords="offset points",
                xytext=(6, 6),
                fontsize=7,
                color=PLOT_TEXT,
            )

    axes.set_xlabel("Mean predicted probability")
    axes.set_ylabel("Fraction of positives")
    axes.set_title(
        title
        or (
            f"Reliability diagram - {disease} "
            f"({calibration.get('n_bins')} bins, {calibration.get('strategy')})"
        )
    )
    axes.set_xlim(0.0, 1.0)
    axes.set_ylim(0.0, 1.0)
    axes.grid(True, linestyle=":", alpha=0.6)
    axes.legend(loc="best", fontsize=9)

    ece = calibration.get("expected_calibration_error")

    caption = (
        f"n = {calibration.get('n_samples')} | "
        f"bins = {calibration.get('n_bins')} ({calibration.get('strategy')})"
    )

    if ece is not None:
        caption += f" | ECE = {ece:.4f}"

    axes.text(
        0.02,
        0.98,
        caption,
        transform=axes.transAxes,
        va="top",
        fontsize=8,
        color=PLOT_TEXT,
    )

    axes.text(
        0.02,
        0.03,
        "Hiệu chuẩn chỉ mô tả tập kiểm thử này, KHÔNG chứng minh giá trị lâm sàng.",
        transform=axes.transAxes,
        fontsize=8,
        color="#5A6B76",
    )

    return _save_figure(figure, output_path)


def generate_roc_curve(
    roc_result: Dict[str, Any],
    output_path: Any,
    *,
    disease: Optional[str] = None,
    title: Optional[str] = None,
) -> Optional[str]:
    """Save a ROC curve built from the probability-based curve points.

    Returns the written path, or ``None`` when the curve is unavailable (for
    example when the evaluation labels contain only one class).
    """

    if not roc_result.get("available") or not roc_result.get("fpr"):

        warnings.warn(
            f"[{disease or 'unknown'}] ROC curve skipped: "
            f"{roc_result.get('status', 'no ROC data')}",
            RuntimeWarning,
            stacklevel=2,
        )

        return None

    figure, axes = plt.subplots(figsize=(7.0, 6.2))

    auroc = roc_result.get("auroc")

    label = "Mô hình (model)"
    if auroc is not None:
        label += f" - AUROC = {auroc:.4f}"

    axes.plot(
        roc_result["fpr"],
        roc_result["tpr"],
        color=PLOT_ACCENT,
        label=label,
    )

    axes.plot(
        [0.0, 1.0],
        [0.0, 1.0],
        linestyle="--",
        color="#9AA5B1",
        label="Phân loại ngẫu nhiên (chance)",
    )

    axes.set_xlabel("False Positive Rate")
    axes.set_ylabel("True Positive Rate")
    axes.set_title(title or f"ROC curve - {disease}")
    axes.set_xlim(0.0, 1.0)
    axes.set_ylim(0.0, 1.0)
    axes.grid(True, linestyle=":", alpha=0.6)
    axes.legend(loc="lower right", fontsize=9)

    axes.text(
        0.02,
        0.98,
        (
            f"n = {roc_result.get('n_samples')} | "
            f"điểm vẽ = {roc_result.get('n_points')}"
        ),
        transform=axes.transAxes,
        va="top",
        fontsize=8,
        color=PLOT_TEXT,
    )

    return _save_figure(figure, output_path)


def generate_confusion_matrix_plot(
    classification: Dict[str, Any],
    output_path: Any,
    *,
    disease: Optional[str] = None,
    title: Optional[str] = None,
) -> Optional[str]:
    """Save a confusion-matrix plot for the threshold-based predictions."""

    matrix_payload = classification.get("confusion_matrix")

    if not classification.get("available") or not matrix_payload:

        warnings.warn(
            f"[{disease or 'unknown'}] Confusion matrix plot skipped: "
            f"{classification.get('status', 'no confusion matrix')}",
            RuntimeWarning,
            stacklevel=2,
        )

        return None

    matrix = np.asarray(matrix_payload["matrix"], dtype=int)
    labels = [str(value) for value in matrix_payload["labels"]]

    figure, axes = plt.subplots(figsize=(6.2, 5.4))

    image = axes.imshow(matrix, cmap="Blues")

    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axes.text(
                column,
                row,
                f"{matrix[row, column]:,}",
                ha="center",
                va="center",
                fontsize=11,
                color=PLOT_TEXT,
            )

    axes.set_xticks(range(len(labels)))
    axes.set_yticks(range(len(labels)))
    axes.set_xticklabels([f"predicted {value}" for value in labels])
    axes.set_yticklabels([f"actual {value}" for value in labels])
    axes.set_xlabel("Predicted class")
    axes.set_ylabel("Actual class")
    axes.set_title(
        title
        or (
            f"Confusion matrix - {disease} "
            f"(threshold = {classification.get('threshold')})"
        )
    )

    figure.colorbar(image, ax=axes, fraction=0.046, pad=0.04)

    return _save_figure(figure, output_path)


# ============================================================
# 11. PERSISTENCE
# ============================================================


def utc_timestamp() -> str:
    """Return an ISO-8601 UTC timestamp for the persisted artifacts."""

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _to_builtin(value: Any) -> Any:
    """Convert numpy / pandas values into plain JSON-serializable Python types.

    Non-finite floats (NaN, inf) become ``None`` so an unavailable metric is
    never written as a misleading number.
    """

    if isinstance(value, dict):
        return {str(key): _to_builtin(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [_to_builtin(item) for item in value]

    if isinstance(value, np.ndarray):
        return [_to_builtin(item) for item in value.tolist()]

    if isinstance(value, np.bool_):
        return bool(value)

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        number = float(value)
        return number if math.isfinite(number) else None

    if isinstance(value, float):
        return value if math.isfinite(value) else None

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    return value


def write_json(payload: Dict[str, Any], path: Any) -> str:
    """Write a JSON artifact with stable formatting. Never overwrites models."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    with target.open("w", encoding="utf-8") as handle:
        json.dump(_to_builtin(payload), handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    return str(target)


def write_dataframe_csv(frame: pd.DataFrame, path: Any) -> str:
    """Write a CSV artifact (headers are written even when there are no rows)."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    frame.to_csv(target, index=False, encoding="utf-8")

    return str(target)


def save_evaluation_results(
    disease: str,
    payload: Dict[str, Any],
    *,
    classification: Dict[str, Any],
    calibration: Dict[str, Any],
    roc: Dict[str, Any],
    errors: Dict[str, Any],
    plots: Optional[Dict[str, Optional[str]]] = None,
    reports_dir: Any = None,
) -> Dict[str, str]:
    """Persist every evaluation artifact for one disease.

    Writes into ``reports/model_evaluation/`` (and ``plots/`` inside it) only:
    model pickles, scalers, imputers, thresholds and processed datasets are never
    touched.
    """

    reports = Path(reports_dir) if reports_dir else DEFAULT_REPORTS_DIR
    reports.mkdir(parents=True, exist_ok=True)

    written: Dict[str, str] = {}

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    matrix_payload = classification.get("confusion_matrix")

    if matrix_payload:

        labels = matrix_payload["labels"]

        matrix_frame = pd.DataFrame(
            matrix_payload["matrix"],
            index=[f"actual_{label}" for label in labels],
            columns=[f"predicted_{label}" for label in labels],
        )
        matrix_frame.index.name = "actual_class"

        written["confusion_matrix_csv"] = write_dataframe_csv(
            matrix_frame,
            reports / f"{disease}_confusion_matrix.csv",
        )

    # --------------------------------------------------------
    # Per-class metrics
    # --------------------------------------------------------

    per_class = classification.get("per_class") or {}

    if per_class:

        per_class_frame = pd.DataFrame(
            [
                {
                    "class_label": item["label"],
                    "role": item["role"],
                    "precision": item["precision"],
                    "recall": item["recall"],
                    "f1": item["f1"],
                    "support": item["support"],
                }
                for item in per_class.values()
            ]
        )

        written["per_class_csv"] = write_dataframe_csv(
            per_class_frame,
            reports / f"{disease}_per_class_metrics.csv",
        )

    # --------------------------------------------------------
    # Calibration (reliability) table
    # --------------------------------------------------------

    bins = calibration.get("bins") or []

    if bins:
        written["calibration_csv"] = write_dataframe_csv(
            pd.DataFrame(bins),
            reports / f"{disease}_calibration.csv",
        )

    # --------------------------------------------------------
    # ROC curve points
    # --------------------------------------------------------

    if roc.get("available") and roc.get("fpr"):

        roc_frame = pd.DataFrame(
            {
                "false_positive_rate": roc["fpr"],
                "true_positive_rate": roc["tpr"],
                "threshold": roc["thresholds"],
            }
        )

        written["roc_csv"] = write_dataframe_csv(
            roc_frame,
            reports / f"{disease}_roc.csv",
        )

    # --------------------------------------------------------
    # Error records (no identifier columns - already stripped)
    # --------------------------------------------------------

    written["false_positives_csv"] = write_dataframe_csv(
        errors.get("false_positive_records", pd.DataFrame()),
        reports / f"{disease}_false_positives.csv",
    )

    written["false_negatives_csv"] = write_dataframe_csv(
        errors.get("false_negative_records", pd.DataFrame()),
        reports / f"{disease}_false_negatives.csv",
    )

    # --------------------------------------------------------
    # Plots
    # --------------------------------------------------------

    for name, plot_path in (plots or {}).items():
        if plot_path:
            written[f"plot_{name}"] = str(plot_path)

    # --------------------------------------------------------
    # Metrics JSON (written last so it can list every other artifact)
    # --------------------------------------------------------

    metrics_path = reports / f"{disease}_metrics.json"
    written["metrics_json"] = str(metrics_path)

    payload = dict(payload)
    payload["artifacts_written"] = written

    write_json(payload, metrics_path)

    return written


# ============================================================
# 12. PER-DISEASE EVALUATION
# ============================================================


def evaluate_disease(
    disease: str,
    *,
    reports_dir: Any = None,
    save_plots: bool = True,
) -> Dict[str, Any]:
    """Evaluate one frozen model on its held-out split and persist the results.

    Nothing is fitted here: the model, the threshold and the transformers are
    loaded from the existing artifacts, and the recovered held-out rows are
    pushed through the live preprocessing pipeline.
    """

    cfg = get_disease_config(disease)

    dataset = load_evaluation_dataset(disease)
    artifacts = dataset["artifacts"]

    positive_label = int(artifacts["positive_label"])
    negative_label = int(artifacts["negative_label"])
    threshold = float(artifacts["threshold"])

    y_true = dataset["y"].to_numpy()

    probabilities = predict_for_evaluation(
        dataset["X_model_input"],
        disease,
        artifacts,
    )

    predictions = predict_labels(probabilities, threshold)

    classification = calculate_classification_metrics(
        y_true,
        predictions,
        positive_label=positive_label,
        threshold=threshold,
    )

    probability_metrics = calculate_probability_metrics(
        y_true,
        probabilities,
        positive_label=positive_label,
    )

    roc = calculate_roc_curve(
        y_true,
        probabilities,
        positive_label=positive_label,
    )

    calibration = calculate_calibration_metrics(
        y_true,
        probabilities,
        positive_label=positive_label,
    )

    errors = analyze_prediction_errors(
        dataset["X_display"],
        y_true,
        probabilities,
        predictions,
        positive_label=positive_label,
        negative_label=negative_label,
        threshold=threshold,
    )

    decision_view = calculate_decision_engine_view(
        probabilities,
        y_true=y_true,
        model_predictions=predictions,
        positive_label=positive_label,
    )

    limitations = collect_dataset_limitations(disease, dataset)

    reports = Path(reports_dir) if reports_dir else DEFAULT_REPORTS_DIR

    plots: Dict[str, Optional[str]] = {}

    if save_plots:

        plots_dir = reports / PLOTS_DIRNAME

        plots["reliability"] = generate_reliability_diagram(
            calibration,
            plots_dir / f"{disease}_reliability.png",
            disease=disease,
            title=(
                f"Reliability diagram - {cfg.display_name_vi} "
                f"({cfg.display_name_en})"
            ),
        )

        plots["roc"] = generate_roc_curve(
            roc,
            plots_dir / f"{disease}_roc.png",
            disease=disease,
            title=f"ROC curve - {cfg.display_name_vi} ({cfg.display_name_en})",
        )

        plots["confusion_matrix"] = generate_confusion_matrix_plot(
            classification,
            plots_dir / f"{disease}_confusion_matrix.png",
            disease=disease,
            title=(
                f"Confusion matrix - {cfg.display_name_vi} "
                f"(threshold = {threshold})"
            ),
        )

    payload: Dict[str, Any] = {
        # ---- required identification -------------------------
        "model": artifacts["model_type"],
        "model_file": Path(artifacts["model_path"]).name,
        "model_path": artifacts["model_path"],
        "n_estimators": artifacts["n_estimators"],
        "disease": disease,
        "display_name_vi": cfg.display_name_vi,
        "display_name_en": cfg.display_name_en,
        "dataset": dataset["dataset_source"],
        "n_samples": dataset["n_samples"],
        "n_features": dataset["n_features"],
        "features": dataset["features"],
        # ---- target definition -------------------------------
        "target_column": cfg.target_column,
        "positive_class": positive_label,
        "negative_class": negative_label,
        "positive_class_meaning_vi": cfg.positive_meaning_vi,
        "negative_class_meaning_vi": cfg.negative_meaning_vi,
        "positive_class_evidence": artifacts["positive_class_evidence"],
        # ---- threshold ---------------------------------------
        "threshold": threshold,
        "threshold_source": artifacts["threshold_path"],
        "threshold_note_vi": (
            "Ngưỡng được nạp từ tệp *_threshold.pkl hiện có qua "
            "prediction_layer.load_thresholds(); không hardcode và không thay "
            "bằng 0.5."
        ),
        # ---- required headline metrics -----------------------
        "accuracy": classification["accuracy"],
        "macro_f1": classification["macro_f1"],
        "auroc": probability_metrics["auroc"],
        "auroc_status": probability_metrics["auroc_status"],
        "positive_precision": classification["positive_precision"],
        "positive_recall": classification["positive_recall"],
        "positive_f1": classification["positive_f1"],
        "brier_score": probability_metrics["brier_score"],
        "brier_score_status": probability_metrics["brier_score_status"],
        "false_positive_count": errors["false_positive_count"],
        "false_negative_count": errors["false_negative_count"],
        # ---- detail ------------------------------------------
        "classification": classification,
        "per_class_metrics": classification["per_class"],
        "confusion_matrix": classification["confusion_matrix"],
        "counts": classification["counts"],
        "rates": classification["rates"],
        "probability_metrics": probability_metrics,
        "roc": {
            "available": roc["available"],
            "status": roc["status"],
            "n_points": roc["n_points"],
            "auroc": roc["auroc"],
            "note": roc["note"],
        },
        "calibration": calibration,
        "error_analysis": {
            key: value
            for key, value in errors.items()
            if not key.endswith("_records")
        },
        "decision_engine_view": decision_view,
        "dataset_limitations": limitations,
        # ---- safety wording ----------------------------------
        "metric_family_note_vi": (
            "AUROC, Brier và đường hiệu chuẩn dùng XÁC SUẤT dự đoán. Accuracy, "
            "Precision, Recall, F1 và ma trận nhầm lẫn dùng NGƯỠNG mô hình lấy "
            "từ *_threshold.pkl."
        ),
        "evaluation_disclaimer_vi": EVALUATION_DISCLAIMER_VI,
        "screening_disclaimer_reference_vi": SCREENING_DISCLAIMER_REFERENCE_VI,
        "records_note_vi": (
            "Bảng lỗi hiển thị giá trị đặc trưng theo đơn vị của tập đánh giá và "
            "chỉ dùng chỉ số dòng (evaluation_row_index); không có cột định danh."
        ),
        "generated_at": utc_timestamp(),
    }

    written = save_evaluation_results(
        disease,
        payload,
        classification=classification,
        calibration=calibration,
        roc=roc,
        errors=errors,
        plots=plots,
        reports_dir=reports,
    )

    payload["artifacts_written"] = written

    return payload


# ============================================================
# 13. RUN ALL DISEASES
# ============================================================


def run_evaluation(
    diseases: Optional[Iterable[str]] = None,
    *,
    reports_dir: Any = None,
    save_plots: bool = True,
) -> Dict[str, Any]:
    """Evaluate the selected models and write every evaluation artifact.

    Returns ``{"results": ..., "failures": ..., "summary": ..., }``. A disease
    that cannot be evaluated (missing artifact) is reported in ``failures`` with
    the original error message - it is never replaced by invented numbers.
    """

    selected = list(diseases) if diseases is not None else list(DISEASE_ORDER)

    unknown = [item for item in selected if item not in DISEASE_CONFIGS]

    if unknown:
        raise KeyError(
            f"Unknown disease(ies): {unknown}. "
            f"Known diseases: {list(DISEASE_CONFIGS)}"
        )

    reports = Path(reports_dir) if reports_dir else DEFAULT_REPORTS_DIR
    reports.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Any] = {}
    failures: Dict[str, str] = {}

    for disease in selected:

        try:

            results[disease] = evaluate_disease(
                disease,
                reports_dir=reports,
                save_plots=save_plots,
            )

        except (FileNotFoundError, ValueError, RuntimeError, KeyError) as error:

            failures[disease] = f"{type(error).__name__}: {error}"

            warnings.warn(
                f"[{disease}] Evaluation incomplete — required artifact missing "
                f"or invalid: {failures[disease]}",
                RuntimeWarning,
                stacklevel=2,
            )

    headline_keys = (
        "accuracy",
        "macro_f1",
        "auroc",
        "positive_precision",
        "positive_recall",
        "positive_f1",
        "brier_score",
        "false_positive_count",
        "false_negative_count",
    )

    summary: Dict[str, Any] = {
        "generated_at": utc_timestamp(),
        "reports_dir": str(reports),
        "evaluated_diseases": list(results),
        "failed_diseases": failures,
        "models": {
            disease: {
                "model_file": payload["model_file"],
                "model_type": payload["model"],
                "threshold": payload["threshold"],
                "threshold_source": payload["threshold_source"],
                "n_samples": payload["n_samples"],
                "positive_class": payload["positive_class"],
                "target_column": payload["target_column"],
            }
            for disease, payload in results.items()
        },
        "headline_metrics": {
            disease: {
                key: payload.get(key) for key in headline_keys
            }
            for disease, payload in results.items()
        },
        "status_notes": {
            disease: {
                "auroc_status": payload.get("auroc_status"),
                "brier_score_status": payload.get("brier_score_status"),
                "calibration_status": payload.get("calibration", {}).get("status"),
                "classification_status": payload.get("classification", {}).get("status"),
            }
            for disease, payload in results.items()
        },
        "no_retraining": True,
        "no_model_artifacts_modified": True,
        "evaluation_disclaimer_vi": EVALUATION_DISCLAIMER_VI,
        "screening_disclaimer_reference_vi": SCREENING_DISCLAIMER_REFERENCE_VI,
    }

    summary_path = write_json(summary, reports / "summary.json")

    return {
        "results": results,
        "failures": failures,
        "summary": summary,
        "summary_path": summary_path,
    }