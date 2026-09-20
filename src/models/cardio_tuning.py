import os
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


print("=" * 60)
print("CARDIO - RANDOM FOREST TUNING")
print("=" * 60)


# ============================================================
# 1. PATHS
# ============================================================

X_TRAIN_PATH = "../../data/processed/cardio_X_train.csv"
Y_TRAIN_PATH = "../../data/processed/cardio_y_train.csv"

MODEL_DIR = "../../data/models"

TUNED_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "cardio_tuned_random_forest.pkl"
)


# ============================================================
# 2. CREATE MODEL DIRECTORY
# ============================================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# ============================================================
# 3. LOAD TRAINING DATA
# ============================================================

X_train = pd.read_csv(
    X_TRAIN_PATH
)

y_train = pd.read_csv(
    Y_TRAIN_PATH
).squeeze()


print("\nTraining data:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)


# ============================================================
# 4. BASE MODEL
# ============================================================

model = RandomForestClassifier(
    random_state=42,
    n_jobs=-1
)


# ============================================================
# 5. PARAMETER GRID
# ============================================================

param_grid = {

    "n_estimators": [
        100,
        200
    ],

    "max_depth": [
        8,
        10,
        None
    ],

    "min_samples_split": [
        2,
        5
    ],

    "min_samples_leaf": [
        1,
        2
    ],

    "max_features": [
        "sqrt"
    ]
}


print("\nParameter grid:")

for parameter, values in param_grid.items():

    print(
        f"{parameter}: {values}"
    )


# ============================================================
# 6. GRID SEARCH
# ============================================================

grid_search = GridSearchCV(

    estimator=model,

    param_grid=param_grid,

    scoring="f1",

    cv=3,

    n_jobs=-1,

    verbose=1
)


print("\n")
print("=" * 60)
print("STARTING GRID SEARCH")
print("=" * 60)


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


print(
    "Best parameters:"
)

print(
    grid_search.best_params_
)


print(
    f"\nBest CV F1-score: "
    f"{grid_search.best_score_:.4f}"
)


# ============================================================
# 8. BEST MODEL
# ============================================================

best_model = grid_search.best_estimator_


print("\n")
print("=" * 60)
print("BEST RANDOM FOREST MODEL")
print("=" * 60)


print(
    best_model
)


# ============================================================
# 9. SAVE TUNED MODEL
# ============================================================

joblib.dump(
    best_model,
    TUNED_MODEL_PATH
)


print("\n")
print("=" * 60)
print("TUNED MODEL SAVED")
print("=" * 60)


print(
    "Model:",
    os.path.abspath(TUNED_MODEL_PATH)
)


# ============================================================
# 10. VERIFY LOAD
# ============================================================

test_model = joblib.load(
    TUNED_MODEL_PATH
)


print("\n")
print("=" * 60)
print("VERIFYING SAVED MODEL")
print("=" * 60)


print(
    "Model load: SUCCESS"
)


print(
    "Loaded model:"
)

print(
    test_model
)


# ============================================================
# 11. COMPLETED
# ============================================================

print("\n")
print("=" * 60)
print("CARDIO RANDOM FOREST TUNING COMPLETED")
print("=" * 60)