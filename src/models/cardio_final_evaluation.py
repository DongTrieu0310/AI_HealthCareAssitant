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
print("CARDIO - FINAL MODEL EVALUATION")
print("=" * 60)


# ============================================================
# 1. PATHS
# ============================================================

X_TEST_PATH = "../../data/processed/cardio_X_test.csv"
Y_TEST_PATH = "../../data/processed/cardio_y_test.csv"

MODEL_PATH = "../../data/models/cardio_random_forest.pkl"
THRESHOLD_PATH = "../../data/models/cardio_threshold.pkl"
FEATURES_PATH = "../../data/models/cardio_features.pkl"


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
# 3. CHECK MISSING VALUES
# ============================================================

print("\nMissing values:")

print(
    "X_test:",
    X_test.isnull().sum().sum()
)

print(
    "y_test:",
    y_test.isnull().sum()
)


if X_test.isnull().sum().sum() > 0:

    raise ValueError(
        "X_test contains missing values!"
    )


if y_test.isnull().sum() > 0:

    raise ValueError(
        "y_test contains missing values!"
    )


# ============================================================
# 4. LOAD FINAL MODEL
# ============================================================

model = joblib.load(
    MODEL_PATH
)

threshold = joblib.load(
    THRESHOLD_PATH
)

features = joblib.load(
    FEATURES_PATH
)


print("\nFinal model loaded successfully.")

print(
    "Threshold:",
    threshold
)


# ============================================================
# 5. ENSURE FEATURE ORDER
# ============================================================

X_test = X_test[
    features
]


print(
    "\nFeature order verified."
)


# ============================================================
# 6. PREDICT PROBABILITY
# ============================================================

probability = model.predict_proba(
    X_test
)[:, 1]


# ============================================================
# 7. APPLY FINAL THRESHOLD
# ============================================================

y_pred = (
    probability >= threshold
).astype(int)


# ============================================================
# 8. CALCULATE METRICS
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
    probability
)


# ============================================================
# 9. DISPLAY METRICS
# ============================================================

print("\n")
print("=" * 60)
print("FINAL EVALUATION METRICS")
print("=" * 60)

print(
    f"Threshold: {threshold:.2f}"
)

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
# 10. CONFUSION MATRIX
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

print(
    "\nTrue Negative (TN):",
    tn
)

print(
    "False Positive (FP):",
    fp
)

print(
    "False Negative (FN):",
    fn
)

print(
    "True Positive (TP):",
    tp
)


# ============================================================
# 11. CLASSIFICATION REPORT
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
# 12. FINAL MODEL SUMMARY
# ============================================================

print("\n")
print("=" * 60)
print("FINAL CARDIO MODEL SUMMARY")
print("=" * 60)

print(
    "Model:",
    model
)

print(
    "\nThreshold:",
    threshold
)

print(
    "\nFeatures:"
)

for feature in features:

    print(
        f"- {feature}"
    )


# ============================================================
# 13. COMPLETED
# ============================================================

print("\n")
print("=" * 60)
print("CARDIO FINAL EVALUATION COMPLETED")
print("=" * 60)