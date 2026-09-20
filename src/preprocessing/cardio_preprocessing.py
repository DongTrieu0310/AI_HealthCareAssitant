import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ============================================================
# CARDIO - PREPROCESSING
# ============================================================

print("=" * 60)
print("CARDIO - PREPROCESSING")
print("=" * 60)


# ============================================================
# 1. LOAD RAW DATA
# ============================================================

cardio = pd.read_csv(
    "../../data/raw/cardio.csv"
)

print("\nOriginal shape:")
print(cardio.shape)


# ============================================================
# 2. CHECK COLUMNS
# ============================================================

print("\nOriginal columns:")

for column in cardio.columns:
    print("-", column)


# ============================================================
# 3. BASIC DATA CHECK
# ============================================================

print("\nMissing values:")
print(
    cardio.isnull().sum()
)

print(
    "\nDuplicate rows:",
    cardio.duplicated().sum()
)


# ============================================================
# 4. REMOVE DUPLICATES
# ============================================================

duplicate_count = cardio.duplicated().sum()

if duplicate_count > 0:

    print(
        f"\nRemoving {duplicate_count} duplicate rows..."
    )

    cardio = cardio.drop_duplicates()

else:

    print("\nNo duplicate rows found.")


# ============================================================
# 5. AGE CONVERSION
# ============================================================

# CARDIO dataset stores age in days.
# Convert age from days to years.

cardio["age"] = cardio["age"] / 365.25


print("\nAge converted from days to years.")

print(
    "Age range:",
    round(cardio["age"].min(), 2),
    "-",
    round(cardio["age"].max(), 2)
)


# ============================================================
# 6. DEFINE FEATURES AND TARGET
# ============================================================

features = [
    "age",
    "gender",
    "height",
    "weight",
    "ap_hi",
    "ap_lo",
    "cholesterol",
    "gluc",
    "smoke",
    "alco",
    "active"
]

target = "cardio"


X = cardio[features].copy()

y = cardio[target].copy()


print("\nFeatures:")

for feature in features:
    print("-", feature)


print("\nTarget:")
print(target)


# ============================================================
# 7. CHECK TARGET DISTRIBUTION
# ============================================================

print("\nTarget distribution:")

print(
    y.value_counts()
)


print("\nTarget percentage:")

print(
    (y.value_counts(normalize=True) * 100).round(2)
)


# ============================================================
# 8. TRAIN / TEST SPLIT
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
# 9. STANDARDIZATION
# ============================================================

scaler = StandardScaler()


# IMPORTANT:
# Fit scaler ONLY on training data.

X_train = pd.DataFrame(

    scaler.fit_transform(X_train),

    columns=X_train.columns,

    index=X_train.index
)


# Test data uses parameters learned from training data.

X_test = pd.DataFrame(

    scaler.transform(X_test),

    columns=X_test.columns,

    index=X_test.index
)


# ============================================================
# 10. CHECK MISSING VALUES
# ============================================================

train_missing = X_train.isnull().sum().sum()

test_missing = X_test.isnull().sum().sum()


print("\nMissing values after preprocessing:")

print(
    "Train:",
    train_missing
)

print(
    "Test :",
    test_missing
)


# ============================================================
# 11. SAFETY CHECK
# ============================================================

if train_missing > 0:

    raise ValueError(
        "X_train still contains missing values!"
    )


if test_missing > 0:

    raise ValueError(
        "X_test still contains missing values!"
    )


# ============================================================
# 12. CREATE PROCESSED DIRECTORY
# ============================================================

processed_dir = "../../data/processed"


os.makedirs(

    processed_dir,

    exist_ok=True
)


# ============================================================
# 13. SAVE PROCESSED DATA
# ============================================================

X_train.to_csv(

    f"{processed_dir}/cardio_X_train.csv",

    index=False
)


X_test.to_csv(

    f"{processed_dir}/cardio_X_test.csv",

    index=False
)


y_train.to_csv(

    f"{processed_dir}/cardio_y_train.csv",

    index=False
)


y_test.to_csv(

    f"{processed_dir}/cardio_y_test.csv",

    index=False
)


# ============================================================
# 14. FINAL CHECK
# ============================================================

print("\n")
print("=" * 60)
print("PROCESSED DATA")
print("=" * 60)


print(
    "X_train:",
    X_train.shape
)

print(
    "X_test :",
    X_test.shape
)

print(
    "y_train:",
    y_train.shape
)

print(
    "y_test :",
    y_test.shape
)


print("\nFinal missing values:")

print(
    "Train:",
    X_train.isnull().sum().sum()
)

print(
    "Test :",
    X_test.isnull().sum().sum()
)


print("\nProcessed files saved successfully.")


print("\n")
print("=" * 60)
print("CARDIO PREPROCESSING COMPLETED SUCCESSFULLY")
print("=" * 60)