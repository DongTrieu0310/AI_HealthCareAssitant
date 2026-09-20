import os
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier


print("=" * 60)
print("CARDIO - FINAL MODEL")
print("=" * 60)


# ============================================================
# 1. PATHS
# ============================================================

X_TRAIN_PATH = "../../data/processed/cardio_X_train.csv"
Y_TRAIN_PATH = "../../data/processed/cardio_y_train.csv"

MODEL_DIR = "../../data/models"

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "cardio_random_forest.pkl"
)

THRESHOLD_PATH = os.path.join(
    MODEL_DIR,
    "cardio_threshold.pkl"
)

FEATURES_PATH = os.path.join(
    MODEL_DIR,
    "cardio_features.pkl"
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
# 4. CHECK MISSING VALUES
# ============================================================

print("\nMissing values:")

print(
    "X_train:",
    X_train.isnull().sum().sum()
)

print(
    "y_train:",
    y_train.isnull().sum()
)


if X_train.isnull().sum().sum() > 0:

    raise ValueError(
        "X_train contains missing values!"
    )


if y_train.isnull().sum() > 0:

    raise ValueError(
        "y_train contains missing values!"
    )


# ============================================================
# 5. FINAL TUNED RANDOM FOREST
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
    X_train,
    y_train
)


print(
    "\nFinal Random Forest trained successfully."
)


# ============================================================
# 6. LOAD FINAL THRESHOLD
# ============================================================

if not os.path.exists(THRESHOLD_PATH):

    raise FileNotFoundError(
        "cardio_threshold.pkl not found!"
    )


threshold = joblib.load(
    THRESHOLD_PATH
)


print(
    f"\nFinal threshold: {threshold}"
)


# ============================================================
# 7. FEATURES
# ============================================================

features = list(
    X_train.columns
)


print("\nFeatures:")

for feature in features:

    print(
        f"- {feature}"
    )


# ============================================================
# 8. SAVE FINAL MODEL
# ============================================================

joblib.dump(
    model,
    MODEL_PATH
)


joblib.dump(
    features,
    FEATURES_PATH
)


print("\n")
print("=" * 60)
print("FINAL MODEL SAVED")
print("=" * 60)


print(
    "Model     :",
    os.path.abspath(MODEL_PATH)
)

print(
    "Threshold :",
    os.path.abspath(THRESHOLD_PATH)
)

print(
    "Features  :",
    os.path.abspath(FEATURES_PATH)
)


# ============================================================
# 9. VERIFY MODEL
# ============================================================

test_model = joblib.load(
    MODEL_PATH
)

test_threshold = joblib.load(
    THRESHOLD_PATH
)

test_features = joblib.load(
    FEATURES_PATH
)


print("\n")
print("=" * 60)
print("VERIFYING SAVED FILES")
print("=" * 60)


print(
    "Model load    : SUCCESS"
)

print(
    "Threshold load: SUCCESS"
)

print(
    "Features load : SUCCESS"
)


print(
    "\nLoaded threshold:",
    test_threshold
)

print(
    "Loaded features:",
    test_features
)


# ============================================================
# 10. FINAL CHECK
# ============================================================

if test_threshold != 0.40:

    raise ValueError(
        "Unexpected CARDIO threshold!"
    )


if test_features != features:

    raise ValueError(
        "Saved features do not match training features!"
    )


print("\nFinal verification: PASSED")


# ============================================================
# 11. COMPLETED
# ============================================================

print("\n")
print("=" * 60)
print("FINAL CARDIO MODEL COMPLETED")
print("=" * 60)
