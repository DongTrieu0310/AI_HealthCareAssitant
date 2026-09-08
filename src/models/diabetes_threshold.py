import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


print("=" * 60)
print("DIABETES - THRESHOLD OPTIMIZATION")
print("=" * 60)


# ============================================================
# 1. LOAD PROCESSED DATA
# ============================================================

X_train_full = pd.read_csv(
    "../../data/processed/diabetes_X_train.csv"
)

X_test = pd.read_csv(
    "../../data/processed/diabetes_X_test.csv"
)

y_train_full = pd.read_csv(
    "../../data/processed/diabetes_y_train.csv"
).squeeze()

y_test = pd.read_csv(
    "../../data/processed/diabetes_y_test.csv"
).squeeze()


print("\nData:")
print("Full training data:", X_train_full.shape)
print("Test data         :", X_test.shape)


# ============================================================
# 2. SPLIT TRAINING DATA INTO TRAIN + VALIDATION
# ============================================================

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full,
    y_train_full,
    test_size=0.20,
    random_state=42,
    stratify=y_train_full
)


print("\nTrain / Validation split:")
print("X_train:", X_train.shape)
print("X_val  :", X_val.shape)
print("X_test :", X_test.shape)


# ============================================================
# 3. TRAIN RANDOM FOREST
# ============================================================

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
# 4. VALIDATION PROBABILITIES
# ============================================================

y_val_prob = model.predict_proba(
    X_val
)[:, 1]


print("Validation probability prediction completed.")


# ============================================================
# 5. TEST DIFFERENT THRESHOLDS ON VALIDATION SET
# ============================================================

thresholds = [
    0.30,
    0.35,
    0.40,
    0.45,
    0.50
]


results = []


print("\n")
print("=" * 60)
print("VALIDATION THRESHOLD COMPARISON")
print("=" * 60)


for threshold in thresholds:

    y_val_pred = (
        y_val_prob >= threshold
    ).astype(int)

    accuracy = accuracy_score(
        y_val,
        y_val_pred
    )

    precision = precision_score(
        y_val,
        y_val_pred,
        zero_division=0
    )

    recall = recall_score(
        y_val,
        y_val_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_val,
        y_val_pred,
        zero_division=0
    )

    cm = confusion_matrix(
        y_val,
        y_val_pred
    )

    tn, fp, fn, tp = cm.ravel()

    results.append({
        "Threshold": threshold,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp
    })


results_df = pd.DataFrame(results)


print(
    results_df.to_string(index=False)
)


# ============================================================
# 6. SELECT BEST THRESHOLD
# ============================================================

best_result = results_df.loc[
    results_df["F1"].idxmax()
]

best_threshold = best_result["Threshold"]


print("\n")
print("=" * 60)
print("BEST THRESHOLD ON VALIDATION SET")
print("=" * 60)

print(
    f"Threshold : {best_threshold:.2f}"
)

print(
    f"Accuracy  : {best_result['Accuracy']:.4f}"
)

print(
    f"Precision : {best_result['Precision']:.4f}"
)

print(
    f"Recall    : {best_result['Recall']:.4f}"
)

print(
    f"F1-score  : {best_result['F1']:.4f}"
)

print(
    f"FN        : {int(best_result['FN'])}"
)

print(
    f"TP        : {int(best_result['TP'])}"
)


# ============================================================
# 7. RETRAIN MODEL ON FULL TRAINING DATA
# ============================================================

final_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

final_model.fit(
    X_train_full,
    y_train_full
)


print("\n")
print("=" * 60)
print("FINAL MODEL RETRAINED")
print("=" * 60)

print(
    "Training data used:",
    X_train_full.shape
)


# ============================================================
# 8. FINAL TEST PREDICTION
# ============================================================

y_test_prob = final_model.predict_proba(
    X_test
)[:, 1]


y_test_pred = (
    y_test_prob >= best_threshold
).astype(int)


# ============================================================
# 9. FINAL TEST EVALUATION
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_test_pred
)

precision = precision_score(
    y_test,
    y_test_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_test_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_test_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    y_test_prob
)


print("\n")
print("=" * 60)
print("FINAL TEST PERFORMANCE")
print("=" * 60)

print(
    f"Threshold : {best_threshold:.2f}"
)

print(
    f"Accuracy  : {accuracy:.4f}"
)

print(
    f"Precision : {precision:.4f}"
)

print(
    f"Recall    : {recall:.4f}"
)

print(
    f"F1-score  : {f1:.4f}"
)

print(
    f"ROC-AUC   : {roc_auc:.4f}"
)


# ============================================================
# 10. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_test_pred
)

tn, fp, fn, tp = cm.ravel()


print("\n")
print("=" * 60)
print("FINAL CONFUSION MATRIX")
print("=" * 60)

print(cm)

print("\nTrue Negative :", tn)
print("False Positive:", fp)
print("False Negative:", fn)
print("True Positive :", tp)


# ============================================================
# 11. VISUALIZATION
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    results_df["Threshold"],
    results_df["F1"],
    marker="o",
    label="F1-score"
)

plt.plot(
    results_df["Threshold"],
    results_df["Recall"],
    marker="o",
    label="Recall"
)

plt.plot(
    results_df["Threshold"],
    results_df["Precision"],
    marker="o",
    label="Precision"
)

plt.xlabel("Classification Threshold")
plt.ylabel("Score")

plt.title(
    "Diabetes - Threshold Optimization"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()


# ============================================================
# 12. COMPLETED
# ============================================================

print("\n")
print("=" * 60)
print("THRESHOLD OPTIMIZATION COMPLETED")
print("=" * 60)

print(
    f"Selected threshold: {best_threshold:.2f}"
)

print(
    "Threshold selected using validation data."
)

print(
    "Final performance evaluated on untouched test data."
)