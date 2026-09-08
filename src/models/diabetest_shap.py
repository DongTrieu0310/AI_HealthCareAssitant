import pandas as pd
import matplotlib.pyplot as plt
import shap

from sklearn.ensemble import RandomForestClassifier


print("=" * 60)
print("DIABETES - SHAP EXPLAINABILITY")
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
# 2. TRAIN RANDOM FOREST
# ============================================================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)


print("\nRandom Forest trained successfully.")


# ============================================================
# 3. CREATE SHAP EXPLAINER
# ============================================================

explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(X_test)


print("\nSHAP values calculated successfully.")


# ============================================================
# 4. SHAP SUMMARY PLOT
# ============================================================

print("\n" + "=" * 60)
print("SHAP SUMMARY PLOT")
print("=" * 60)

shap.summary_plot(
    shap_values[:, :, 1],
    X_test,
    show=True
)


# ============================================================
# 5. SHAP FEATURE IMPORTANCE
# ============================================================

shap_importance = pd.DataFrame({
    "Feature": X_test.columns,
    "Importance": abs(
        shap_values[:, :, 1]
    ).mean(axis=0)
})

shap_importance = shap_importance.sort_values(
    "Importance",
    ascending=False
)


print("\n" + "=" * 60)
print("SHAP FEATURE IMPORTANCE")
print("=" * 60)

print(
    shap_importance.to_string(index=False)
)


# ============================================================
# 6. MOST IMPORTANT FEATURE
# ============================================================

most_important = shap_importance.iloc[0]


print("\n" + "=" * 60)
print("MOST IMPORTANT FEATURE")
print("=" * 60)

print(
    f"Feature   : {most_important['Feature']}"
)

print(
    f"Importance: {most_important['Importance']:.4f}"
)


# ============================================================
# 7. BAR PLOT
# ============================================================

plt.figure()

shap_importance.sort_values(
    "Importance"
).plot(
    x="Feature",
    y="Importance",
    kind="barh",
    legend=False
)

plt.title("Diabetes - SHAP Feature Importance")

plt.xlabel("Mean |SHAP value|")

plt.ylabel("Feature")

plt.tight_layout()

plt.show()


print("\n" + "=" * 60)
print("SHAP ANALYSIS COMPLETED")
print("=" * 60)