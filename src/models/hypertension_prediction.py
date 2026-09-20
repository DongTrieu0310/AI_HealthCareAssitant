import os
import pickle
import pandas as pd


print("=" * 60)
print("HYPERTENSION - PREDICTION")
print("=" * 60)


# ============================================================
# 1. LOAD FINAL MODEL
# ============================================================

model_path = "../../data/models/hypertension_random_forest.pkl"

threshold_path = "../../data/models/hypertension_threshold.pkl"

features_path = "../../data/models/hypertension_features.pkl"


with open(model_path, "rb") as file:
    model = pickle.load(file)


with open(threshold_path, "rb") as file:
    threshold = pickle.load(file)


with open(features_path, "rb") as file:
    features = pickle.load(file)


print("\nFinal model loaded successfully.")

print(
    f"Threshold: {threshold}"
)


# ============================================================
# 2. DISPLAY REQUIRED FEATURES
# ============================================================

print("\nFeatures required by model:")

for feature in features:

    print(
        f"- {feature}"
    )


# ============================================================
# 3. ENTER PATIENT INFORMATION
# ============================================================

print("\n")
print("=" * 60)
print("ENTER PATIENT INFORMATION")
print("=" * 60)


male = float(
    input("Male (0 = Female, 1 = Male): ")
)

age = float(
    input("Age: ")
)

currentSmoker = float(
    input("Current Smoker (0 = No, 1 = Yes): ")
)

cigsPerDay = float(
    input("Cigarettes per Day: ")
)

BPMeds = float(
    input("Blood Pressure Medication (0 = No, 1 = Yes): ")
)

diabetes = float(
    input("Diabetes (0 = No, 1 = Yes): ")
)

totChol = float(
    input("Total Cholesterol: ")
)

sysBP = float(
    input("Systolic Blood Pressure (sysBP): ")
)

diaBP = float(
    input("Diastolic Blood Pressure (diaBP): ")
)

BMI = float(
    input("BMI: ")
)

heartRate = float(
    input("Heart Rate: ")
)

glucose = float(
    input("Glucose: ")
)


# ============================================================
# 4. CREATE PATIENT DATAFRAME
# ============================================================

patient_data = pd.DataFrame({

    "male": [male],

    "age": [age],

    "currentSmoker": [currentSmoker],

    "cigsPerDay": [cigsPerDay],

    "BPMeds": [BPMeds],

    "diabetes": [diabetes],

    "totChol": [totChol],

    "sysBP": [sysBP],

    "diaBP": [diaBP],

    "BMI": [BMI],

    "heartRate": [heartRate],

    "glucose": [glucose]
})


# ============================================================
# 5. ENSURE FEATURE ORDER
# ============================================================

patient_data = patient_data[
    features
]


print("\n")
print("=" * 60)
print("PATIENT DATA")
print("=" * 60)

print(
    patient_data.to_string(index=False)
)


print(
    "\nMissing values:",
    patient_data.isnull().sum().sum()
)


# ============================================================
# 6. PREDICT PROBABILITY
# ============================================================

probability = model.predict_proba(
    patient_data
)[0][1]


print("\n")
print("=" * 60)
print("PREDICTION PROBABILITY")
print("=" * 60)

print(
    f"Hypertension probability: {probability:.4f}"
)

print(
    f"Hypertension probability: {probability * 100:.2f}%"
)


# ============================================================
# 7. APPLY THRESHOLD
# ============================================================

prediction = int(
    probability >= threshold
)


# ============================================================
# 8. FINAL PREDICTION
# ============================================================

print("\n")
print("=" * 60)
print("FINAL PREDICTION")
print("=" * 60)

print(
    f"Threshold: {threshold}"
)


if prediction == 1:

    print(
        "Prediction: HYPERTENSION RISK"
    )

else:

    print(
        "Prediction: NO HYPERTENSION RISK"
    )


# ============================================================
# 9. COMPLETED
# ============================================================

print("\n")
print("=" * 60)
print("PREDICTION COMPLETED")
print("=" * 60)