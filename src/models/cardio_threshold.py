import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


print("=" * 60)
print("CARDIO - THRESHOLD OPTIMIZATION")
print("=" * 60)


# ============================================================
# 1. PATHS
# ============================================================

X_TRAIN_PATH = "../../data/processed/cardio_X_train.csv"
Y_TRAIN_PATH = "../../data/processed/cardio_y_train.csv"


# ============================================================
# 2. LOAD TRAINING DATA
# ============================================================

X_train = pd.read_csv(
    X_TRAIN_PATH
)

y_train = pd.read_csv(
    Y_TRAIN_PATH
).squeeze()


print("\nOriginal training data:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)


# ============================================================
# 3. CREATE VALIDATION SET
# ============================================================

X_model_train, X_validation, y_model_train, y_validation = train_test_split(

    X_train,
    y_train,

    test_size=0.20,

    random_state=42,

    stratify=y_train
)


print("\nValidation split:")

print(
    "Model training:",
    X_model_train.shape
)

print(
    "Validation:",
    X_validation.shape
)


# ============================================================
# 4. TRAIN TUNED RANDOM FOREST
# ============================================================

model = RandomForestClassifier(

    n_estimators=200,

    max_depth=10,

    min_samples_split=5,

    min_samples_leaf=2,

    max_features="sqrt",

    random_state=42,

    n_jobs=-1
)


model.fit(
    X_model_train,
    y_model_train
)


print("\nTuned Random Forest trained successfully.")


# ============================================================
# 5. VALIDATION PROBABILITIES
# ============================================================

validation_probability = model.predict_proba(
    X_validation
)[:, 1]


# ============================================================
# 6. THRESHOLDS
# ============================================================

thresholds = [

    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60

]


print("\nThresholds:")
print(thresholds)


# ============================================================
# 7. TEST THRESHOLDS ON VALIDATION SET
# ============================================================

results = []


print("\n")
print("=" * 85)
print("VALIDATION THRESHOLD COMPARISON")
print("=" * 85)


for threshold in thresholds:

    validation_prediction = (
        validation_probability >= threshold
    ).astype(int)


    accuracy = accuracy_score(
        y_validation,
        validation_prediction
    )


    precision = precision_score(
        y_validation,
        validation_prediction,
        zero_division=0
    )


    recall = recall_score(
        y_validation,
        validation_prediction,
        zero_division=0
    )


    f1 = f1_score(
        y_validation,
        validation_prediction,
        zero_division=0
    )


    tn, fp, fn, tp = confusion_matrix(

        y_validation,

        validation_prediction

    ).ravel()


    results.append({

        "Threshold": threshold,

        "Accuracy": accuracy,

        "Precision": precision,

        "Recall": recall,

        "F1": f1,

        "False_Negative": fn,

        "False_Positive": fp

    })


    print(
        f"\nThreshold: {threshold:.2f}"
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
        f"FN       : {fn}"
    )

    print(
        f"FP       : {fp}"
    )


# ============================================================
# 8. RESULTS TABLE
# ============================================================

results_df = pd.DataFrame(
    results
)


print("\n")
print("=" * 85)
print("VALIDATION THRESHOLD TABLE")
print("=" * 85)


print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# 9. BEST F1 THRESHOLD
# ============================================================

best_row = results_df.loc[
    results_df["F1"].idxmax()
]


best_threshold = float(
    best_row["Threshold"]
)


print("\n")
print("=" * 60)
print("BEST VALIDATION THRESHOLD")
print("=" * 60)


print(
    f"Threshold: {best_threshold:.2f}"
)

print(
    f"Accuracy : {best_row['Accuracy']:.4f}"
)

print(
    f"Precision: {best_row['Precision']:.4f}"
)

print(
    f"Recall   : {best_row['Recall']:.4f}"
)

print(
    f"F1-score : {best_row['F1']:.4f}"
)

print(
    f"FN       : {int(best_row['False_Negative'])}"
)

print(
    f"FP       : {int(best_row['False_Positive'])}"
)


# ============================================================
# 10. SAVE THRESHOLD
# ============================================================

THRESHOLD_PATH = "../../data/models/cardio_threshold.pkl"


joblib.dump(
    best_threshold,
    THRESHOLD_PATH
)


print("\n")
print("=" * 60)
print("THRESHOLD SAVED")
print("=" * 60)


print(
    "Threshold:",
    best_threshold
)

print(
    "Path:",
    THRESHOLD_PATH
)


# ============================================================
# 11. COMPLETED
# ============================================================

print("\n")
print("=" * 60)
print("CARDIO THRESHOLD OPTIMIZATION COMPLETED")
print("=" * 60)