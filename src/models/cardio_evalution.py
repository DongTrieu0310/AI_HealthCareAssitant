import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve
)


print("=" * 60)
print("CARDIO - MODEL EVALUATION")
print("=" * 60)


# ============================================================
# 1. PATHS
# ============================================================

X_TRAIN_PATH = "../../data/processed/cardio_X_train.csv"
X_TEST_PATH = "../../data/processed/cardio_X_test.csv"

Y_TRAIN_PATH = "../../data/processed/cardio_y_train.csv"
Y_TEST_PATH = "../../data/processed/cardio_y_test.csv"


# ============================================================
# 2. LOAD DATA
# ============================================================

X_train = pd.read_csv(
    X_TRAIN_PATH
)

X_test = pd.read_csv(
    X_TEST_PATH
)

y_train = pd.read_csv(
    Y_TRAIN_PATH
).squeeze()

y_test = pd.read_csv(
    Y_TEST_PATH
).squeeze()


print("\nDataset:")
print("X_train:", X_train.shape)
print("X_test :", X_test.shape)
print("y_train:", y_train.shape)
print("y_test :", y_test.shape)


# ============================================================
# 3. TRAIN RANDOM FOREST BASELINE
# ============================================================

print("\n")
print("=" * 60)
print("TRAINING RANDOM FOREST")
print("=" * 60)


model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


model.fit(
    X_train,
    y_train
)


print("\nRandom Forest trained successfully.")


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
# 5. BASIC METRICS
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
print("EVALUATION METRICS")
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
# 9. ROC CURVE
# ============================================================

fpr, tpr, thresholds = roc_curve(
    y_test,
    y_probability
)


plt.figure(
    figsize=(8, 6)
)


plt.plot(
    fpr,
    tpr,
    label=f"Random Forest (AUC = {roc_auc:.4f})"
)


plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)


plt.xlabel(
    "False Positive Rate"
)


plt.ylabel(
    "True Positive Rate"
)


plt.title(
    "CARDIO - Random Forest ROC Curve"
)


plt.legend()


plt.grid(
    True
)


plt.tight_layout()


plt.show()


# ============================================================
# 10. EVALUATION SUMMARY
# ============================================================

print("\n")
print("=" * 60)
print("EVALUATION SUMMARY")
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


print("\nConfusion Matrix:")

print(
    f"TN = {tn}"
)

print(
    f"FP = {fp}"
)

print(
    f"FN = {fn}"
)

print(
    f"TP = {tp}"
)


# ============================================================
# 11. COMPLETED
# ============================================================

print("\n")
print("=" * 60)
print("CARDIO EVALUATION COMPLETED")
print("=" * 60)