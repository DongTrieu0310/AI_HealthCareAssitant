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
    ConfusionMatrixDisplay,
    RocCurveDisplay
)


print("=" * 60)
print("DIABETES - RANDOM FOREST EVALUATION")
print("=" * 60)


# ============================================================
# 1. LOAD PROCESSED DATA
# ============================================================

X_train = pd.read_csv(
    "../../data/processed/diabetes_X_train.csv"
)

X_test = pd.read_csv(
    "../../data/processed/diabetes_X_test.csv"
)

y_train = pd.read_csv(
    "../../data/processed/diabetes_y_train.csv"
).squeeze()

y_test = pd.read_csv(
    "../../data/processed/diabetes_y_test.csv"
).squeeze()


print("\nData:")
print("X_train:", X_train.shape)
print("X_test :", X_test.shape)
print("y_train:", y_train.shape)
print("y_test :", y_test.shape)


# ============================================================
# 2. CHECK MISSING VALUES
# ============================================================

print("\nMissing values:")
print("X_train:", X_train.isna().sum().sum())
print("X_test :", X_test.isna().sum().sum())


# ============================================================
# 3. TRAIN RANDOM FOREST
# ============================================================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

print("\nRandom Forest trained successfully.")


# ============================================================
# 4. PREDICTION
# ============================================================

y_pred = model.predict(X_test)

y_prob = model.predict_proba(X_test)[:, 1]


# ============================================================
# 5. CALCULATE METRICS
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
    y_prob
)


# ============================================================
# 6. DISPLAY METRICS
# ============================================================

print("\n")
print("=" * 60)
print("RANDOM FOREST EVALUATION")
print("=" * 60)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")


# ============================================================
# 7. CLASSIFICATION REPORT
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
            "No Diabetes",
            "Diabetes"
        ],
        zero_division=0
    )
)


# ============================================================
# 8. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\n")
print("=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

print(cm)


# ============================================================
# 9. CONFUSION MATRIX VISUALIZATION
# ============================================================

ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "No Diabetes",
        "Diabetes"
    ]
).plot()

plt.title("Random Forest - Confusion Matrix")

plt.tight_layout()

plt.show()


# ============================================================
# 10. ROC CURVE
# ============================================================

RocCurveDisplay.from_predictions(
    y_test,
    y_prob
)

plt.title("Random Forest - ROC Curve")

plt.tight_layout()

plt.show()


# ============================================================
# 11. FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 60)
print("FINAL MODEL SUMMARY")
print("=" * 60)

print("Model    : Random Forest")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")

print("\n")
print("=" * 60)
print("EVALUATION COMPLETED SUCCESSFULLY")
print("=" * 60)