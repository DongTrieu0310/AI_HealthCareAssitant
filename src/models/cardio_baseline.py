"""
CARDIO - BASELINE MODELS

Purpose:
    Build baseline machine learning models for cardiovascular
    disease prediction.

Models:
    1. Logistic Regression
    2. Decision Tree
    3. Random Forest

Evaluation:
    - Accuracy
    - Precision
    - Recall
    - F1-score
    - ROC-AUC
"""

import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


print("=" * 60)
print("CARDIO - BASELINE MODELS")
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
# 3. CHECK DATA
# ============================================================

print("\nMissing values:")

print(
    "X_train:",
    X_train.isnull().sum().sum()
)

print(
    "X_test :",
    X_test.isnull().sum().sum()
)

print(
    "y_train:",
    y_train.isnull().sum()
)

print(
    "y_test :",
    y_test.isnull().sum()
)


# ============================================================
# 4. BASELINE MODELS
# ============================================================

models = {

    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42
    ),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
}


# ============================================================
# 5. TRAIN AND EVALUATE
# ============================================================

results = []


for model_name, model in models.items():

    print("\n")
    print("-" * 60)
    print(model_name)
    print("-" * 60)

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.fit(
        X_train,
        y_train
    )

    print("Model trained successfully.")

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    y_pred = model.predict(
        X_test
    )

    # --------------------------------------------------------
    # PROBABILITY
    # --------------------------------------------------------

    y_probability = model.predict_proba(
        X_test
    )[:, 1]

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # PRINT
    # --------------------------------------------------------

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")

    # --------------------------------------------------------
    # SAVE RESULT
    # --------------------------------------------------------

    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC_AUC": roc_auc
    })


# ============================================================
# 6. COMPARE MODELS
# ============================================================

results_df = pd.DataFrame(
    results
)


print("\n")
print("=" * 60)
print("BASELINE MODEL COMPARISON")
print("=" * 60)

print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# 7. BEST MODEL BY F1
# ============================================================

best_model = results_df.loc[
    results_df["F1"].idxmax()
]


print("\n")
print("=" * 60)
print("BEST BASELINE MODEL")
print("=" * 60)

print(
    "Model:",
    best_model["Model"]
)

print(
    f"F1-score: {best_model['F1']:.4f}"
)

print(
    f"ROC-AUC: {best_model['ROC_AUC']:.4f}"
)


# ============================================================
# 8. COMPLETED
# ============================================================

print("\n")
print("=" * 60)
print("CARDIO BASELINE COMPLETED")
print("=" * 60)