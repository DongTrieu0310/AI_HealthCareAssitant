import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier


print("=" * 60)
print("DIABETES - RANDOM FOREST FEATURE IMPORTANCE")
print("=" * 60)


# ============================================================
# 1. LOAD PROCESSED DATA
# ============================================================

X_train = pd.read_csv(
    "../../data/processed/diabetes_X_train.csv"
)

X_test = pd.read_csv(
    "../../data/processed/diabetes_X_test.csv"
)

y_train = pd.read_csv(
    "../../data/processed/diabetes_y_train.csv"
).squeeze()

y_test = pd.read_csv(
    "../../data/processed/diabetes_y_test.csv"
).squeeze()


print("\nData:")
print("X_train:", X_train.shape)
print("X_test :", X_test.shape)
print("y_train:", y_train.shape)
print("y_test :", y_test.shape)


# ============================================================
# 2. TRAIN RANDOM FOREST BASELINE
# ============================================================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)


print("\nRandom Forest trained successfully.")


# ============================================================
# 3. GET FEATURE IMPORTANCE
# ============================================================

importance = model.feature_importances_


feature_importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": importance
})


# Sort from highest to lowest
feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
).reset_index(drop=True)


# ============================================================
# 4. DISPLAY FEATURE IMPORTANCE
# ============================================================

print("\n")
print("=" * 60)
print("FEATURE IMPORTANCE")
print("=" * 60)

print(
    feature_importance.to_string(index=False)
)


# ============================================================
# 5. TOP FEATURE
# ============================================================

top_feature = feature_importance.iloc[0]

print("\n")
print("=" * 60)
print("MOST IMPORTANT FEATURE")
print("=" * 60)

print(
    f"Feature   : {top_feature['Feature']}"
)

print(
    f"Importance: {top_feature['Importance']:.4f}"
)


# ============================================================
# 6. VISUALIZATION
# ============================================================

plt.figure(figsize=(10, 6))

plt.barh(
    feature_importance["Feature"],
    feature_importance["Importance"]
)

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Random Forest Feature Importance")

plt.gca().invert_yaxis()

plt.tight_layout()

plt.show()


# ============================================================
# 7. COMPLETED
# ============================================================

print("\n")
print("=" * 60)
print("FEATURE IMPORTANCE ANALYSIS COMPLETED")
print("=" * 60)