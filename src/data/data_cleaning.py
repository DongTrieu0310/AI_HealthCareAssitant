import pandas as pd

# ==========================================
# 1. LOAD RAW DATA
# ==========================================

diabetes = pd.read_csv("../../data/raw/diabetes.csv")

print("Original shape:", diabetes.shape)

# ==========================================
# 2. CHECK DUPLICATES
# ==========================================

duplicates = diabetes.duplicated().sum()

print("Duplicates:", duplicates)

if duplicates > 0:
    diabetes = diabetes.drop_duplicates()

# ==========================================
# 3. CHECK MISSING VALUES
# ==========================================

print("\nMissing values:")
print(diabetes.isnull().sum())

# ==========================================
# 4. CHECK DATA TYPES
# ==========================================

print("\nData types:")
print(diabetes.dtypes)

# ==========================================
# 5. FINAL SHAPE
# ==========================================

print("\nCleaned shape:", diabetes.shape)

output_path = "../../data/processed/diabetes_clean.csv"

diabetes.to_csv(output_path, index=False)

print("\nSaved cleaned data to:")
print(output_path)

