import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


print("=" * 60)
print("CARDIO - TUNED MODEL EVALUATION")
print("=" * 60)


# ============================================================
# 1. PATHS
# ============================================================

X_TEST_PATH = "../../data/processed/cardio_X_test.csv"
Y_TEST_PATH = "../../data/processed/cardio_y_test.csv"

MODEL_PATH = "../../data/models/cardio_tuned_random_forest.pkl"


# ============================================================
# 2. LOAD TEST DATA
# ============================================================

X_test = pd.read_csv(
    X_TEST_PATH
)

y_test = pd.read_csv(
    Y_TEST_PATH
).squeeze()


print("\nTest data:")
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)


# ============================================================
# 3. LOAD TUNED MODEL
# ============================================================

model = joblib.load(
    MODEL_PATH
)


print("\nTuned model loaded successfully.")

print("\nModel:")
print(model)


# ============================================================
# 4. PREDICTION
# ============================================================

y_pred = model.predict(
    X_test
)


y_probability = model.predict_proba(
    X_test
)[:, 1]


# ============================================================
# 5. METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    y_probability
)


# ============================================================
# 6. DISPLAY METRICS
# ============================================================

print("\n")
print("=" * 60)
print("TUNED MODEL METRICS")
print("=" * 60)

print(
    f"Accuracy : {accuracy:.4f}"
)

print(
    f"Precision: {precision:.4f}"
)

print(
    f"Recall   : {recall:.4f}"
)

print(
    f"F1-score : {f1:.4f}"
)

print(
    f"ROC-AUC  : {roc_auc:.4f}"
)


# ============================================================
# 7. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)


tn, fp, fn, tp = cm.ravel()


print("\n")
print("=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

print(cm)

print("\nTrue Negative (TN):", tn)
print("False Positive (FP):", fp)
print("False Negative (FN):", fn)
print("True Positive (TP):", tp)


# ============================================================
# 8. CLASSIFICATION REPORT
# ============================================================

print("\n")
print("=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "No Cardiovascular Disease",
            "Cardiovascular Disease"
        ],
        zero_division=0
    )
)


# ============================================================
# 9. COMPARE WITH BASELINE
# ============================================================

baseline_f1 = 0.7095
baseline_roc_auc = 0.7705
baseline_accuracy = 0.7130


print("\n")
print("=" * 60)
print("BASELINE VS TUNED")
print("=" * 60)


print(
    f"Accuracy: "
    f"{baseline_accuracy:.4f} → {accuracy:.4f}"
)

print(
    f"F1-score: "
    f"{baseline_f1:.4f} → {f1:.4f}"
)

print(
    f"ROC-AUC : "
    f"{baseline_roc_auc:.4f} → {roc_auc:.4f}"
)


print("\nF1 improvement:")

print(
    f"{f1 - baseline_f1:+.4f}"
)


print("\nROC-AUC improvement:")

print(
    f"{roc_auc - baseline_roc_auc:+.4f}"
)


# ============================================================
# 10. COMPLETED
# ============================================================

print("\n")
print("=" * 60)
print("CARDIO TUNED EVALUATION COMPLETED")
print("=" * 60)