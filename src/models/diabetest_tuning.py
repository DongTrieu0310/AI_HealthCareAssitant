import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


# ============================================================
# 1. HEADER
# ============================================================

print("=" * 60)
print("DIABETES - RANDOM FOREST TUNING")
print("=" * 60)


# ============================================================
# 2. LOAD PROCESSED DATA
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
# 3. BASE RANDOM FOREST
# ============================================================

rf = RandomForestClassifier(
    random_state=42
)


# ============================================================
# 4. HYPERPARAMETER GRID
# ============================================================

param_grid = {

    "n_estimators": [
        100,
        200,
        300
    ],

    "max_depth": [
        None,
        5,
        10,
        15
    ],

    "min_samples_split": [
        2,
        5,
        10
    ],

    "min_samples_leaf": [
        1,
        2,
        4
    ],

    "max_features": [
        "sqrt",
        "log2"
    ]
}


# ============================================================
# 5. CROSS VALIDATION
# ============================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# ============================================================
# 6. GRID SEARCH
# ============================================================

grid_search = GridSearchCV(

    estimator=rf,

    param_grid=param_grid,

    scoring="f1",

    cv=cv,

    n_jobs=-1,

    verbose=1
)


print("\nStarting GridSearchCV...")

grid_search.fit(
    X_train,
    y_train
)


# ============================================================
# 7. BEST PARAMETERS
# ============================================================

print("\n")
print("=" * 60)
print("BEST PARAMETERS")
print("=" * 60)

print(grid_search.best_params_)

print(
    f"\nBest CV F1: "
    f"{grid_search.best_score_:.4f}"
)


# ============================================================
# 8. BEST MODEL
# ============================================================

best_model = grid_search.best_estimator_


# ============================================================
# 9. TEST SET EVALUATION
# ============================================================

y_pred = best_model.predict(X_test)

y_prob = best_model.predict_proba(
    X_test
)[:, 1]


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
# 10. RESULTS
# ============================================================

print("\n")
print("=" * 60)
print("TUNED RANDOM FOREST RESULTS")
print("=" * 60)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")


# ============================================================
# 11. COMPARE WITH BASELINE
# ============================================================

print("\n")
print("=" * 60)
print("BASELINE vs TUNED")
print("=" * 60)

print("\nBaseline Random Forest:")
print("Accuracy : 0.8117")
print("Precision: 0.7273")
print("Recall   : 0.7407")
print("F1       : 0.7339")
print("ROC-AUC  : 0.8753")

print("\nTuned Random Forest:")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1       : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")


print("\n" + "=" * 60)
print("TUNING COMPLETED SUCCESSFULLY")
print("=" * 60)