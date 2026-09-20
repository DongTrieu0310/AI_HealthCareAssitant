import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


print("=" * 60)
print("HYPERTENSION - BASELINE MODEL")
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


print("\nLoaded data:")
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
# 3. TARGET DISTRIBUTION
# ============================================================

print("\nTarget distribution:")
print(y_train.value_counts())


print("\nTarget percentage:")
print(
    (y_train.value_counts(normalize=True) * 100)
    .round(2)
)


# ============================================================
# 4. DEFINE MODELS
# ============================================================

models = {

    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42
    ),

    "KNN": KNeighborsClassifier(
        n_neighbors=5
    ),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ),

    "SVM": SVC(
        probability=True,
        random_state=42
    )
}


# ============================================================
# 5. TRAIN + EVALUATE
# ============================================================

results = []


for name, model in models.items():

    print("\n" + "-" * 60)
    print(name)
    print("-" * 60)

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    model.fit(
        X_train,
        y_train
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    y_pred = model.predict(
        X_test
    )

    # --------------------------------------------------------
    # Probability
    # --------------------------------------------------------

    y_prob = model.predict_proba(
        X_test
    )[:, 1]

    # --------------------------------------------------------
    # Metrics
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
        y_prob
    )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC-AUC": roc_auc
    })


# ============================================================
# 6. MODEL COMPARISON
# ============================================================

results_df = pd.DataFrame(
    results
)


print("\n")
print("=" * 60)
print("BASELINE MODEL COMPARISON")
print("=" * 60)


print(
    results_df
    .sort_values(
        "F1",
        ascending=False
    )
    .to_string(index=False)
)


# ============================================================
# 7. BEST MODEL
# ============================================================

best_model = results_df.loc[
    results_df["F1"].idxmax()
]


print("\n")
print("=" * 60)
print("BEST BASELINE MODEL")
print("=" * 60)


print(
    f"Model   : {best_model['Model']}"
)

print(
    f"Accuracy: {best_model['Accuracy']:.4f}"
)

print(
    f"Precision: {best_model['Precision']:.4f}"
)

print(
    f"Recall  : {best_model['Recall']:.4f}"
)

print(
    f"F1      : {best_model['F1']:.4f}"
)

print(
    f"ROC-AUC : {best_model['ROC-AUC']:.4f}"
)


# ============================================================
# 8. COMPLETED
# ============================================================

print("\n")
print("=" * 60)
print("BASELINE COMPLETED SUCCESSFULLY")
print("=" * 60)