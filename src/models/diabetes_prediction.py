import os
import joblib
import pandas as pd


print("=" * 60)
print("DIABETES - PREDICTION")
print("=" * 60)


# ============================================================
# 1. LOAD FINAL MODEL
# ============================================================

model_path = "../../data/models/diabetes_random_forest.pkl"
threshold_path = "../../data/models/diabetes_threshold.pkl"
features_path = "../../data/models/diabetes_features.pkl"

model = joblib.load(model_path)
threshold = joblib.load(threshold_path)
features = joblib.load(features_path)


print("\nFinal model loaded successfully.")
print("Threshold:", threshold)

print("\nFeatures required by model:")
for feature in features:
    print("-", feature)


# ============================================================
# 2. LOAD TRAINING DATA
#    Used to obtain the Insulin imputation value
# ============================================================

X_train = pd.read_csv(
    "../../data/processed/diabetes_X_train.csv"
)


# Median Insulin from processed training data
insulin_median = X_train["Insulin"].median()


print("\nInsulin imputation median:", insulin_median)


# ============================================================
# 3. INPUT PATIENT DATA
# ============================================================

print("\n")
print("=" * 60)
print("ENTER PATIENT INFORMATION")
print("=" * 60)

pregnancies = float(
    input("Pregnancies: ")
)

glucose = float(
    input("Glucose: ")
)

blood_pressure = float(
    input("BloodPressure: ")
)

skin_thickness = float(
    input("SkinThickness: ")
)

insulin = float(
    input("Insulin: ")
)

bmi = float(
    input("BMI: ")
)

diabetes_pedigree = float(
    input("DiabetesPedigreeFunction: ")
)

age = float(
    input("Age: ")
)


# ============================================================
# 4. CREATE INPUT DATAFRAME
# ============================================================

patient = pd.DataFrame([
    {
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": diabetes_pedigree,
        "Age": age
    }
])


print("\n")
print("=" * 60)
print("RAW PATIENT DATA")
print("=" * 60)

print(patient.to_string(index=False))


# ============================================================
# 5. HANDLE SUSPICIOUS INSULIN VALUES
# ============================================================

suspicious_insulin_values = [
    102.5,
    169.5
]

if patient.loc[0, "Insulin"] in suspicious_insulin_values:

    print("\nSuspicious Insulin value detected.")

    patient.loc[0, "Insulin"] = insulin_median

    print(
        "Insulin replaced with median:",
        insulin_median
    )


# ============================================================
# 6. CHECK MISSING VALUES
# ============================================================

missing_values = patient.isnull().sum().sum()

print("\nMissing values:", missing_values)


if missing_values > 0:

    raise ValueError(
        "Patient data contains missing values."
    )


# ============================================================
# 7. ENSURE CORRECT FEATURE ORDER
# ============================================================

patient = patient[features]


# ============================================================
# 8. PREDICT PROBABILITY
# ============================================================

probability = model.predict_proba(
    patient
)[0, 1]


print("\n")
print("=" * 60)
print("PREDICTION PROBABILITY")
print("=" * 60)

print(
    f"Diabetes probability: {probability:.4f}"
)

print(
    f"Diabetes probability: {probability * 100:.2f}%"
)


# ============================================================
# 9. APPLY FINAL THRESHOLD
# ============================================================

prediction = int(
    probability >= threshold
)


print("\n")
print("=" * 60)
print("FINAL PREDICTION")
print("=" * 60)

print(
    f"Threshold: {threshold}"
)


if prediction == 1:

    print("Prediction: DIABETES RISK")

else:

    print("Prediction: NO DIABETES RISK")


# ============================================================
# 10. FINAL RESULT
# ============================================================

print("\n")
print("=" * 60)
print("PREDICTION COMPLETED")
print("=" * 60)