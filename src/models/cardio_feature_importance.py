import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier


print("=" * 60)
print("CARDIO - FEATURE IMPORTANCE")
print("=" * 60)


# ============================================================
# 1. PATHS
# ============================================================

X_TRAIN_PATH = "../../data/processed/cardio_X_train.csv"
Y_TRAIN_PATH = "../../data/processed/cardio_y_train.csv"


# ============================================================
# 2. LOAD DATA
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
# 3. TRAIN RANDOM FOREST
# ============================================================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


model.fit(
    X_train,
    y_train
)


print("\nRandom Forest trained successfully.")


# ============================================================
# 4. GET FEATURE IMPORTANCE
# ============================================================

importance = model.feature_importances_


feature_importance = pd.DataFrame({

    "Feature": X_train.columns,

    "Importance": importance

})


# ============================================================
# 5. SORT FEATURES
# ============================================================

feature_importance = feature_importance.sort_values(

    by="Importance",

    ascending=False

).reset_index(drop=True)


# ============================================================
# 6. DISPLAY RESULTS
# ============================================================

print("\n")
print("=" * 60)
print("FEATURE IMPORTANCE")
print("=" * 60)


for index, row in feature_importance.iterrows():

    print(
        f"{index + 1:2d}. "
        f"{row['Feature']:<15} "
        f"{row['Importance']:.6f}"
    )


# ============================================================
# 7. FEATURE IMPORTANCE SUM CHECK
# ============================================================

importance_sum = feature_importance["Importance"].sum()


print("\n")
print("=" * 60)
print("IMPORTANCE CHECK")
print("=" * 60)


print(
    f"Total importance: {importance_sum:.6f}"
)


if abs(importance_sum - 1.0) > 0.000001:

    print(
        "WARNING: Feature importance does not sum to 1.0"
    )

else:

    print(
        "Feature importance sum check: PASSED"
    )


# ============================================================
# 8. PLOT
# ============================================================

plt.figure(
    figsize=(10, 6)
)


plt.barh(

    feature_importance["Feature"],

    feature_importance["Importance"]

)


plt.xlabel(
    "Importance"
)


plt.ylabel(
    "Feature"
)


plt.title(
    "CARDIO - Random Forest Feature Importance"
)


plt.gca().invert_yaxis()


plt.tight_layout()


plt.show()


# ============================================================
# 9. TOP FEATURES
# ============================================================

print("\n")
print("=" * 60)
print("TOP 5 FEATURES")
print("=" * 60)


top_features = feature_importance.head(5)


for index, row in top_features.iterrows():

    print(
        f"{index + 1}. "
        f"{row['Feature']} "
        f"({row['Importance']:.6f})"
    )


# ============================================================
# 10. COMPLETED
# ============================================================

print("\n")
print("=" * 60)
print("CARDIO FEATURE IMPORTANCE COMPLETED")
print("=" * 60)