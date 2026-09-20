"""
Model evaluation layer for the AI Healthcare Assistant.

This package contains the offline evaluation of the three FROZEN risk-screening
models (cardiovascular, diabetes, hypertension) against the held-out test splits
that already exist in the repository.

Modules
-------
``model_evaluation.py``
    Library: dataset / artifact loading, classification metrics, probability
    metrics, calibration, error analysis and artifact persistence. It reuses
    ``src/prediction/prediction_layer.py`` so neither preprocessing nor
    prediction logic is duplicated.

``run_evaluation.py``
    Command line entry point::

        python src/evaluation/run_evaluation.py

``evaluation_dashboard.py``
    Streamlit dashboard that reads the persisted artifacts from
    ``reports/model_evaluation/``.

Nothing in this package trains, refits or replaces a model, and nothing here is
part of the runtime patient-prediction workflow.
"""

__all__ = [
    "model_evaluation",
    "evaluation_dashboard",
    "run_evaluation",
]