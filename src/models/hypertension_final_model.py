import os
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier


print("=" * 60)
print("HYPERTENSION - FINAL MODEL")
print("=" * 60)


# ============================================================
# 1. PATHS
# ============================================================

X_TRAIN_PATH = "../../data/processed/hypertension_X_train.csv"
Y_TRAIN_PATH = "../../data/processed/hypertension_y_train.csv"

MODEL_DIR = "../../data/models"

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "hypertension_random_forest.pkl"
)

THRESHOLD_PATH = os.path.join(
    MODEL_DIR,
    "hypertension_threshold.pkl"
)

FEATURES_PATH = os.path.join(
    MODEL_DIR,
    "hypertension_features.pkl"
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
# 4. FINAL RANDOM FOREST
# ============================================================

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    max_features="sqrt",
    min_samples_leaf=1,
    min_samples_split=5,
    random_state=42
)

model.fit(
    X_train,
    y_train
)

print("\nFinal Random Forest trained successfully.")


# ============================================================
# 5. FINAL THRESHOLD
# ============================================================

threshold = 0.35

print(f"\nFinal threshold: {threshold}")


# ============================================================
# 6. FEATURES
# ============================================================

features = list(
    X_train.columns
)


# ============================================================
# 7. SAVE MODEL
# ============================================================

joblib.dump(
    model,
    MODEL_PATH
)

joblib.dump(
    threshold,
    THRESHOLD_PATH
)

joblib.dump(
    features,
    FEATURES_PATH
)


# ============================================================
# 8. VERIFY FILES
# ============================================================

print("\n")
print("=" * 60)
print("FINAL MODEL SAVED")
print("=" * 60)

print("Model     :", os.path.abspath(MODEL_PATH))
print("Threshold :", os.path.abspath(THRESHOLD_PATH))
print("Features  :", os.path.abspath(FEATURES_PATH))

print("\nFeatures used by model:")

for feature in features:
    print(f"- {feature}")


# ============================================================
# 9. VERIFY LOAD
# ============================================================

print("\n")
print("=" * 60)
print("VERIFYING SAVED FILES")
print("=" * 60)

test_model = joblib.load(
    MODEL_PATH
)

test_threshold = joblib.load(
    THRESHOLD_PATH
)

test_features = joblib.load(
    FEATURES_PATH
)

print("Model load    : SUCCESS")
print("Threshold load: SUCCESS")
print("Features load : SUCCESS")

print("\nLoaded threshold:", test_threshold)
print("Loaded features :", test_features)


# ============================================================
# 10. COMPLETED
# ============================================================

print("\n")
print("=" * 60)
print("FINAL HYPERTENSION MODEL COMPLETED")
print("=" * 60)