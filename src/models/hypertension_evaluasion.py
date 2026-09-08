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
    ConfusionMatrixDisplay
)


print("=" * 60)
print("HYPERTENSION - TUNED RANDOM FOREST EVALUATION")
print("=" * 60)


# ============================================================
# 1. LOAD PROCESSED DATA
# ============================================================

X_train = pd.read_csv(
    "../../data/processed/hypertension_X_train.csv"
)

X_test = pd.read_csv(
    "../../data/processed/hypertension_X_test.csv"
)

y_train = pd.read_csv(
    "../../data/processed/hypertension_y_train.csv"
).squeeze()

y_test = pd.read_csv(
    "../../data/processed/hypertension_y_test.csv"
).squeeze()


print("\nData:")
print("X_train:", X_train.shape)
print("X_test :", X_test.shape)
print("y_train:", y_train.shape)
print("y_test :", y_test.shape)


print("\nMissing values:")
print("X_train:", X_train.isnull().sum().sum())
print("X_test :", X_test.isnull().sum().sum())


# ============================================================
# 2. TRAIN TUNED RANDOM FOREST
# ============================================================

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    max_features="sqrt",
    min_samples_leaf=1,
    min_samples_split=5,
    random_state=42
)

model.fit(X_train, y_train)

print("\nTuned Random Forest trained successfully.")


# ============================================================
# 3. PREDICTION
# ============================================================

y_pred = model.predict(X_test)

y_prob = model.predict_proba(X_test)[:, 1]


# ============================================================
# 4. PERFORMANCE METRICS
# ============================================================

accuracy = accuracy_score(y_test, y_pred)

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


print("\n")
print("=" * 60)
print("TUNED MODEL PERFORMANCE")
print("=" * 60)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")


# ============================================================
# 5. CONFUSION MATRIX
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

print("\nTrue Negative :", tn)
print("False Positive:", fp)
print("False Negative:", fn)
print("True Positive :", tp)


# ============================================================
# 6. CLASSIFICATION REPORT
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
            "No Hypertension Risk",
            "Hypertension Risk"
        ],
        zero_division=0
    )
)


# ============================================================
# 7. CONFUSION MATRIX VISUALIZATION
# ============================================================

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "No Risk",
        "Risk"
    ]
)

disp.plot()

plt.title("Tuned Random Forest - Hypertension Confusion Matrix")

plt.tight_layout()

plt.show()


# ============================================================
# 8. FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 60)
print("FINAL TUNED MODEL SUMMARY")
print("=" * 60)

print("Model    : Tuned Random Forest")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1       : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")


print("\n")
print("=" * 60)
print("HYPERTENSION EVALUATION COMPLETED")
print("=" * 60)