import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


# ============================================================
# DIABETES - PREPROCESSING
# ============================================================

print("=" * 60)
print("DIABETES - PREPROCESSING")
print("=" * 60)


# ============================================================
# 1. LOAD RAW DATA
# ============================================================

diabetes = pd.read_csv("../../data/raw/diabetes.csv")

print("\nOriginal shape:")
print(diabetes.shape)


# ============================================================
# 2. DEFINE FEATURES AND TARGET
# ============================================================

features = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age"
]

target = "Outcome"

X = diabetes[features].copy()
y = diabetes[target].copy()

print("\nFeatures:")
print(features)

print("\nTarget:")
print(target)


# ============================================================
# 3. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTrain shape:")
print(X_train.shape)

print("\nTest shape:")
print(X_test.shape)


# ============================================================
# 4. HANDLE SUSPICIOUS INSULIN VALUES
# ============================================================

suspicious_insulin = [102.5, 169.5]

print("\nSuspicious Insulin values:")
print(suspicious_insulin)


# Replace suspicious values with NaN
X_train["Insulin"] = X_train["Insulin"].replace(
    suspicious_insulin,
    float("nan")
)

X_test["Insulin"] = X_test["Insulin"].replace(
    suspicious_insulin,
    float("nan")
)


# ============================================================
# 5. IMPUTATION
# ============================================================

imputer = SimpleImputer(strategy="median")


# IMPORTANT:
# Fit only on training data
X_train = pd.DataFrame(
    imputer.fit_transform(X_train),
    columns=X_train.columns,
    index=X_train.index
)


# Test data only uses the median learned from training data
X_test = pd.DataFrame(
    imputer.transform(X_test),
    columns=X_test.columns,
    index=X_test.index
)


# ============================================================
# 6. CHECK MISSING VALUES
# ============================================================

train_missing = X_train.isnull().sum().sum()
test_missing = X_test.isnull().sum().sum()

print("\nMissing values after imputation:")
print("Train:", train_missing)
print("Test :", test_missing)


# Safety check
if train_missing > 0:
    raise ValueError("X_train still contains missing values!")

if test_missing > 0:
    raise ValueError("X_test still contains missing values!")


# ============================================================
# 7. STANDARDIZATION
# ============================================================

scaler = StandardScaler()


X_train = pd.DataFrame(
    scaler.fit_transform(X_train),
    columns=X_train.columns,
    index=X_train.index
)


X_test = pd.DataFrame(
    scaler.transform(X_test),
    columns=X_test.columns,
    index=X_test.index
)


# ============================================================
# 8. CREATE PROCESSED DIRECTORY
# ============================================================

processed_dir = "../../data/processed"

os.makedirs(
    processed_dir,
    exist_ok=True
)


# ============================================================
# 9. SAVE PROCESSED DATA
# ============================================================

X_train.to_csv(
    f"{processed_dir}/diabetes_X_train.csv",
    index=False
)

X_test.to_csv(
    f"{processed_dir}/diabetes_X_test.csv",
    index=False
)

y_train.to_csv(
    f"{processed_dir}/diabetes_y_train.csv",
    index=False
)

y_test.to_csv(
    f"{processed_dir}/diabetes_y_test.csv",
    index=False
)


# ============================================================
# 10. FINAL CHECK
# ============================================================

print("\nProcessed data shape:")
print("X_train:", X_train.shape)
print("X_test :", X_test.shape)
print("y_train:", y_train.shape)
print("y_test :", y_test.shape)


print("\nFinal missing values:")
print("Train:", X_train.isnull().sum().sum())
print("Test :", X_test.isnull().sum().sum())


print("\nProcessed files saved successfully.")


print("\n" + "=" * 60)
print("PREPROCESSING COMPLETED SUCCESSFULLY")
print("=" * 60)