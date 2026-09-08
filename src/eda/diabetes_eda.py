import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2_contingency
# ==========================================
# 1. LOAD CLEAN DATA
# ==========================================

diabetes = pd.read_csv("../../data/processed/diabetes_clean.csv")

print("Shape:", diabetes.shape)

# # ==========================================
# # 2. TARGET DISTRIBUTION
# # ==========================================
#
# print("\n" + "=" * 60)
# print("DIABETES - TARGET DISTRIBUTION")
# print("=" * 60)
#
# print(diabetes["Outcome"].value_counts())
#
# print("\nPercentage:")
# print(
#     diabetes["Outcome"]
#     .value_counts(normalize=True)
#     .mul(100)
#     .round(2)
# )
#
# # ==========================================
# # 3. VISUALIZE TARGET
# # ==========================================
#
# diabetes["Outcome"].value_counts().plot(
#     kind="bar"
# )
#
# plt.title("Diabetes Outcome Distribution")
# plt.xlabel("Outcome")
# plt.ylabel("Number of Patients")
#
# plt.tight_layout()
# plt.show()
#
# output_path = "../../data/processed/diabetes_clean.csv"
# diabetes.to_csv(output_path, index=False)
#
# print("\nSaved cleaned data to:")
# print(output_path)
#
# # ==========================================
# # 4. GLUCOSE VS DIABETES
# # ==========================================
#
# print("\n" + "=" * 60)
# print("DIABETES - GLUCOSE ANALYSIS")
# print("=" * 60)
#
# print("\nGlucose statistics by Outcome:")
#
# print(
#     diabetes.groupby("Outcome")["Glucose"]
#     .agg(["count", "mean", "median", "min", "max", "std"])
# )
#
# # ==========================================
# # 5. VISUALIZATION
# # ==========================================
#
# plt.figure(figsize=(8, 5))
#
# diabetes.boxplot(
#     column="Glucose",
#     by="Outcome"
# )
#
# plt.title("Glucose Distribution by Diabetes Outcome")
# plt.suptitle("")
# plt.xlabel("Outcome")
# plt.ylabel("Glucose")
#
# plt.tight_layout()
# plt.show()
#
# # ==========================================
# # 6. BMI VS DIABETES
# # ==========================================
#
# print("\n" + "=" * 60)
# print("DIABETES - BMI ANALYSIS")
# print("=" * 60)
#
# print("\nBMI statistics by Outcome:")
#
# print(
#     diabetes.groupby("Outcome")["BMI"]
#     .agg(["count", "mean", "median", "min", "max", "std"])
# )
#
# # ==========================================
# # 7. BMI BOXPLOT
# # ==========================================
#
# plt.figure(figsize=(8, 5))
#
# diabetes.boxplot(
#     column="BMI",
#     by="Outcome"
# )
#
# plt.title("BMI Distribution by Diabetes Outcome")
# plt.suptitle("")
# plt.xlabel("Outcome")
# plt.ylabel("BMI")
#
# plt.tight_layout()
# plt.show()
#
# # ==========================================
# # 8. AGE VS DIABETES
# # ==========================================
#
# print("\n" + "=" * 60)
# print("DIABETES - AGE ANALYSIS")
# print("=" * 60)
#
# print("\nAge statistics by Outcome:")
#
# print(
#     diabetes.groupby("Outcome")["Age"]
#     .agg(["count", "mean", "median", "min", "max", "std"])
# )
#
# # ==========================================
# # 9. AGE BOXPLOT
# # ==========================================
#
# plt.figure(figsize=(8, 5))
#
# diabetes.boxplot(
#     column="Age",
#     by="Outcome"
# )
#
# plt.title("Age Distribution by Diabetes Outcome")
# plt.suptitle("")
# plt.xlabel("Outcome")
# plt.ylabel("Age")
#
# plt.tight_layout()
# plt.show()
#
#
# print("\n" + "=" * 60)
# print("DIABETES - PREGNANCIES ANALYSIS")
# print("=" * 60)
#
# preg_stats = diabetes.groupby("Outcome")["Pregnancies"].agg(
#     ["count", "mean", "median", "min", "max", "std"]
# )
#
# print(preg_stats)
# plt.figure(figsize=(8, 6))
#
# diabetes.boxplot(
#     column="Pregnancies",
#     by="Outcome"
# )
#
# plt.title("Pregnancies Distribution by Diabetes Outcome")
# plt.suptitle("")
# plt.xlabel("Outcome")
# plt.ylabel("Pregnancies")
#
# plt.show()
#
# print("\n" + "=" * 60)
# print("DIABETES - BLOOD PRESSURE ANALYSIS")
# print("=" * 60)
#
# bp_stats = diabetes.groupby("Outcome")["BloodPressure"].agg(
#     ["count", "mean", "median", "min", "max", "std"]
# )
#
# print(bp_stats)
# plt.figure(figsize=(8, 6))
#
# diabetes.boxplot(
#     column="BloodPressure",
#     by="Outcome"
# )
#
# plt.title("Blood Pressure Distribution by Diabetes Outcome")
# plt.suptitle("")
# plt.xlabel("Outcome")
# plt.ylabel("Blood Pressure")
#
# plt.show()
#
# print("\n" + "=" * 60)
# print("DIABETES - INSULIN ANALYSIS")
# print("=" * 60)
#
# insulin_stats = diabetes.groupby("Outcome")["Insulin"].agg(
#     ["count", "mean", "median", "min", "max", "std"]
# )
#
# print(insulin_stats)
# plt.figure(figsize=(8, 6))
#
# diabetes.boxplot(
#     column="Insulin",
#     by="Outcome"
# )
#
# plt.title("Insulin Distribution by Diabetes Outcome")
# plt.suptitle("")
# plt.xlabel("Outcome")
# plt.ylabel("Insulin")
#
# plt.tight_layout()
# plt.show()
#
# print("\n" + "=" * 60)
# print("DIABETES - PEDIGREE FUNCTION ANALYSIS")
# print("=" * 60)
#
# print(
#     diabetes.groupby("Outcome")["DiabetesPedigreeFunction"]
#     .agg(["count", "mean", "median", "min", "max", "std"])
# )
#
# plt.figure(figsize=(8, 6))
#
# diabetes.boxplot(
#     column="DiabetesPedigreeFunction",
#     by="Outcome"
# )
#
# plt.title("Diabetes Pedigree Function Distribution by Diabetes Outcome")
# plt.suptitle("")
# plt.xlabel("Outcome")
# plt.ylabel("Diabetes Pedigree Function")
#
# plt.tight_layout()
# plt.show()
#
# print("\n" + "=" * 60)
# print("DIABETES - SKIN THICKNESS ANALYSIS")
# print("=" * 60)
#
# print(
#     diabetes.groupby("Outcome")["SkinThickness"]
#     .agg(["count", "mean", "median", "min", "max", "std"])
# )
#
# plt.figure(figsize=(8, 6))
#
# diabetes.boxplot(
#     column="SkinThickness",
#     by="Outcome"
# )
#
# plt.title("Skin Thickness Distribution by Diabetes Outcome")
# plt.suptitle("")
# plt.xlabel("Outcome")
# plt.ylabel("Skin Thickness")
#
# print("\n" + "=" * 60)
# print("DIABETES - FEATURE CORRELATION HEATMAP")
# print("=" * 60)
#
# correlation = diabetes.drop(columns=["Outcome"]).corr()
#
# print(correlation.round(3))
#
# plt.figure(figsize=(10, 8))
#
# plt.imshow(correlation, cmap="coolwarm", aspect="auto")
#
# plt.colorbar(label="Correlation")
#
# plt.xticks(
#     range(len(correlation.columns)),
#     correlation.columns,
#     rotation=45,
#     ha="right"
# )
#
# plt.yticks(
#     range(len(correlation.columns)),
#     correlation.columns
# )
#
# plt.title("Diabetes Feature Correlation")
#
# plt.tight_layout()
# plt.show()
#
# print("\n" + "=" * 60)
# print("DIABETES - GLUCOSE vs BMI")
# print("=" * 60)
#
# plt.figure(figsize=(8, 6))
#
# for outcome in [0, 1]:
#
#     subset = diabetes[diabetes["Outcome"] == outcome]
#
#     plt.scatter(
#         subset["Glucose"],
#         subset["BMI"],
#         label=f"Outcome {outcome}",
#         alpha=0.5
#     )
#
# plt.xlabel("Glucose")
# plt.ylabel("BMI")
# plt.title("Glucose vs BMI by Diabetes Outcome")
# plt.legend()
#
# plt.tight_layout()
# plt.show()
#
# plt.tight_layout()
# plt.show()
#
# print("\n" + "=" * 60)
# print("DIABETES - FEATURE MEAN DIFFERENCE")
# print("=" * 60)
#
# features = [
#     "Pregnancies",
#     "Glucose",
#     "BloodPressure",
#     "SkinThickness",
#     "Insulin",
#     "BMI",
#     "DiabetesPedigreeFunction",
#     "Age"
# ]
#
# for feature in features:
#
#     mean_0 = diabetes.loc[
#         diabetes["Outcome"] == 0, feature
#     ].mean()
#
#     mean_1 = diabetes.loc[
#         diabetes["Outcome"] == 1, feature
#     ].mean()
#
#     difference = mean_1 - mean_0
#
#     print(
#         f"{feature:30s}"
#         f"Outcome 0 = {mean_0:8.2f} | "
#         f"Outcome 1 = {mean_1:8.2f} | "
#         f"Difference = {difference:8.2f}"
#     )
#
# print("\n" + "=" * 60)
# print("DIABETES - STANDARDIZED MEAN DIFFERENCE")
# print("=" * 60)
#
# for feature in features:
#
#     group_0 = diabetes.loc[
#         diabetes["Outcome"] == 0, feature
#     ]
#
#     group_1 = diabetes.loc[
#         diabetes["Outcome"] == 1, feature
#     ]
#
#     mean_0 = group_0.mean()
#     mean_1 = group_1.mean()
#
#     std_0 = group_0.std()
#     std_1 = group_1.std()
#
#     pooled_std = np.sqrt(
#         (std_0 ** 2 + std_1 ** 2) / 2
#     )
#
#     effect_size = (mean_1 - mean_0) / pooled_std
#
#     print(
#         f"{feature:30s}"
#         f"Effect Size = {effect_size:.3f}"
#     )
#
#
# print("\n" + "=" * 60)
# print("DIABETES - OUTCOME RATE BY GLUCOSE")
# print("=" * 60)
#
# glucose_bins = [0, 100, 125, 140, 160, 200]
#
# diabetes["GlucoseGroup"] = pd.cut(
#     diabetes["Glucose"],
#     bins=glucose_bins
# )
#
# glucose_risk = diabetes.groupby(
#     "GlucoseGroup",
#     observed=True
# )["Outcome"].mean()
#
# print(glucose_risk)
#
# print("\n" + "=" * 60)
# print("DIABETES - OUTCOME RATE BY BMI")
# print("=" * 60)
#
# bmi_bins = [0, 18.5, 25, 30, 35, 40, 100]
#
# diabetes["BMIGroup"] = pd.cut(
#     diabetes["BMI"],
#     bins=bmi_bins
# )
#
# bmi_risk = diabetes.groupby(
#     "BMIGroup",
#     observed=True
# )["Outcome"].mean()
#
# print(bmi_risk)
#
# print("\n" + "=" * 60)
# print("DIABETES - OUTCOME RATE BY AGE")
# print("=" * 60)
#
# age_bins = [0, 25, 35, 45, 55, 65, 100]
#
# diabetes["AgeGroup"] = pd.cut(
#     diabetes["Age"],
#     bins=age_bins
# )
#
# age_risk = diabetes.groupby(
#     "AgeGroup",
#     observed=True
# )["Outcome"].mean()
#
# print(age_risk)
#
# print("\n" + "=" * 60)
# print("DIABETES - GROUP SAMPLE SIZE")
# print("=" * 60)
#
# print("\nGlucose groups:")
# print(diabetes["GlucoseGroup"].value_counts().sort_index())
#
# print("\nBMI groups:")
# print(diabetes["BMIGroup"].value_counts().sort_index())
#
# print("\nAge groups:")
# print(diabetes["AgeGroup"].value_counts().sort_index())
#
# print("\n" + "=" * 60)
# print("DIABETES - PREGNANCIES RISK ANALYSIS")
# print("=" * 60)
#
# pregnancy_bins = [-1, 0, 2, 5, 10, 20]
#
# diabetes["PregnancyGroup"] = pd.cut(
#     diabetes["Pregnancies"],
#     bins=pregnancy_bins
# )
#
# pregnancy_risk = diabetes.groupby(
#     "PregnancyGroup",
#     observed=True
# )["Outcome"].mean()
#
# print(pregnancy_risk)
#
# print("\n" + "=" * 60)
# print("DIABETES - FEATURE IMPORTANCE BY CORRELATION")
# print("=" * 60)
#
# # Chỉ lấy các feature dạng số
# numeric_features = diabetes.select_dtypes(include=np.number)
#
# # Bỏ target
# numeric_features = numeric_features.drop(
#     columns=["Outcome"],
#     errors="ignore"
# )
#
# correlation = (
#     numeric_features
#     .corrwith(diabetes["Outcome"])
#     .abs()
#     .sort_values(ascending=False)
# )
#
# print(correlation)
#
#
# features = [
#     "Glucose",
#     "BMI",
#     "Age",
#     "Insulin",
#     "Pregnancies"
# ]
#
# for feature in features:
#
#     plt.figure(figsize=(8, 5))
#
#     diabetes[diabetes["Outcome"] == 0][feature].hist(
#         alpha=0.5,
#         label="No Diabetes"
#     )
#
#     diabetes[diabetes["Outcome"] == 1][feature].hist(
#         alpha=0.5,
#         label="Diabetes"
#     )
#
#     plt.xlabel(feature)
#     plt.ylabel("Frequency")
#     plt.title(f"{feature} Distribution by Diabetes Outcome")
#     plt.legend()
#     plt.tight_layout()
#
#     plt.show()

print("\n" + "=" * 60)
print("DIABETES - GLUCOSE + BMI RISK ANALYSIS")
print("=" * 60)

# Create risk groups
diabetes["GlucoseRisk"] = pd.cut(
    diabetes["Glucose"],
    bins=[0, 100, 125, 140, 160, 200],
    labels=[
        "Low",
        "Moderate",
        "Elevated",
        "High",
        "Very High"
    ]
)

diabetes["BMIRisk"] = pd.cut(
    diabetes["BMI"],
    bins=[0, 25, 30, 35, 40, 100],
    labels=[
        "Normal",
        "Overweight",
        "Obese I",
        "Obese II",
        "Obese III"
    ]
)

risk_table = pd.crosstab(
    diabetes["GlucoseRisk"],
    diabetes["BMIRisk"],
    values=diabetes["Outcome"],
    aggfunc="mean"
)

print("\nDiabetes rate by Glucose + BMI:")
print((risk_table * 100).round(2))

print("\n" + "=" * 60)
print("DIABETES - GLUCOSE + BMI SAMPLE SIZE")
print("=" * 60)

sample_size = pd.crosstab(
    diabetes["GlucoseRisk"],
    diabetes["BMIRisk"]
)

print(sample_size)

risk_count = pd.crosstab(
    diabetes["GlucoseRisk"],
    diabetes["BMIRisk"]
)

risk_rate = pd.crosstab(
    diabetes["GlucoseRisk"],
    diabetes["BMIRisk"],
    values=diabetes["Outcome"],
    aggfunc="mean"
)

print("\nRisk Rate:")
print((risk_rate * 100).round(2))

print("\nSample Size:")
print(risk_count)

print("\n" + "=" * 60)
print("DIABETES - GLUCOSE + BMI + AGE RISK ANALYSIS")
print("=" * 60)

# Create age groups
diabetes["AgeRisk"] = pd.cut(
    diabetes["Age"],
    bins=[0, 25, 35, 45, 55, 100],
    labels=[
        "Young",
        "Adult",
        "Middle-aged",
        "Older",
        "Senior"
    ]
)

# Calculate diabetes rate
risk_3d = diabetes.groupby(
    ["GlucoseRisk", "BMIRisk", "AgeRisk"],
    observed=True
)["Outcome"].agg(
    ["mean", "count"]
)

risk_3d["rate"] = risk_3d["mean"] * 100

print("\nGlucose + BMI + Age risk:")
print(risk_3d)
print("\n" + "=" * 60)
print("DIABETES - RELIABLE 3-FEATURE RISK GROUPS")
print("=" * 60)

reliable_groups = risk_3d[
    risk_3d["count"] >= 10
].sort_values(
    "rate",
    ascending=False
)

print(reliable_groups)

print("\n" + "=" * 60)
print("DIABETES - 3-FEATURE RISK GROUP RELIABILITY")
print("=" * 60)

risk_3 = (
    diabetes
    .groupby(["GlucoseRisk", "BMIRisk", "AgeRisk"])["Outcome"]
    .agg(["mean", "count"])
    .reset_index()
)

risk_3["rate"] = risk_3["mean"] * 100

# Classify reliability based on sample size
def reliability(count):
    if count >= 30:
        return "Strong"
    elif count >= 20:
        return "Moderate"
    elif count >= 10:
        return "Exploratory"
    else:
        return "Too Small"

risk_3["Reliability"] = risk_3["count"].apply(reliability)

# Keep groups with at least 10 samples
reliable_risk = (
    risk_3[risk_3["count"] >= 10]
    .sort_values("rate", ascending=False)
)

print(reliable_risk.to_string(index=False))

print("\n" + "=" * 60)
print("DIABETES - ODDS RATIO ANALYSIS")
print("=" * 60)



features = [
    "Glucose",
    "BMI",
    "Age",
    "Insulin",
    "Pregnancies"
]

for feature in features:

    median_value = diabetes[feature].median()

    diabetes[f"{feature}_High"] = (
        diabetes[feature] > median_value
    ).astype(int)

    table = pd.crosstab(
        diabetes[f"{feature}_High"],
        diabetes["Outcome"]
    )

    print(f"\n{feature}")
    print("-" * 40)
    print("Median:", median_value)
    print(table)

    if table.shape == (2, 2):

        a = table.loc[0, 0]
        b = table.loc[0, 1]
        c = table.loc[1, 0]
        d = table.loc[1, 1]

        odds_ratio = (d * a) / (b * c)

        print("Odds Ratio:", round(odds_ratio, 3))

print("\n" + "=" * 60)
print("DIABETES - ODDS RATIO + CHI-SQUARE")
print("=" * 60)

features = [
    "Glucose",
    "BMI",
    "Age",
    "Insulin",
    "Pregnancies"
]

for feature in features:

    median_value = diabetes[feature].median()

    high_feature = (
        diabetes[feature] > median_value
    ).astype(int)

    table = pd.crosstab(
        high_feature,
        diabetes["Outcome"]
    )

    chi2, p_value, dof, expected = chi2_contingency(table)

    a = table.loc[0, 0]
    b = table.loc[0, 1]
    c = table.loc[1, 0]
    d = table.loc[1, 1]

    odds_ratio = (d * a) / (b * c)

    print(f"\n{feature}")
    print("-" * 40)
    print("Odds Ratio :", round(odds_ratio, 3))
    print("Chi-square :", round(chi2, 3))
    print("p-value    :", round(p_value, 8))

    if p_value < 0.05:
        print("Significant: YES")
    else:
        print("Significant: NO")

print("\n" + "=" * 60)
print("DIABETES - INSULIN ZERO ANALYSIS")
print("=" * 60)

for outcome in [0, 1]:

    subset = diabetes[diabetes["Outcome"] == outcome]

    zero_count = (subset["Insulin"] == 0).sum()

    total = len(subset)

    percentage = zero_count / total * 100

    print(
        f"Outcome {outcome}: "
        f"Zero Insulin = {zero_count} "
        f"({percentage:.2f}%)"
    )

print("\n" + "=" * 60)
print("DIABETES - INSULIN DISTRIBUTION")
print("=" * 60)

print(
    diabetes.groupby("Outcome")["Insulin"]
    .describe()
)

print("\n" + "=" * 60)
print("DIABETES - INSULIN NON-ZERO ANALYSIS")
print("=" * 60)

insulin_nonzero = diabetes[diabetes["Insulin"] > 0]

print(
    insulin_nonzero.groupby("Outcome")["Insulin"]
    .describe()
)

print("\nNon-zero Insulin sample size:")

print(
    insulin_nonzero["Outcome"]
    .value_counts()
    .sort_index()
)

print("\n" + "=" * 60)
print("DIABETES - INSULIN VALUE BY OUTCOME")
print("=" * 60)

for outcome in [0, 1]:

    print(f"\nOutcome = {outcome}")

    insulin_count = (
        diabetes[diabetes["Outcome"] == outcome]["Insulin"]
        .value_counts()
        .head(15)
    )

    print(insulin_count)

print("\n" + "=" * 60)
print("DIABETES - INSULIN SUSPICIOUS VALUE PROFILE")
print("=" * 60)

for value in [102.5, 169.5]:

    subset = diabetes[diabetes["Insulin"] == value]

    print("\n" + "-" * 50)
    print(f"Insulin = {value}")
    print("-" * 50)

    print("Count:", len(subset))

    print("\nOutcome:")
    print(subset["Outcome"].value_counts())

    print("\nOther feature statistics:")

    print(
        subset[
            [
                "Pregnancies",
                "Glucose",
                "BloodPressure",
                "SkinThickness",
                "BMI",
                "DiabetesPedigreeFunction",
                "Age"
            ]
        ].describe()
    )


print("=" * 60)
print("ZERO VALUE CHECK")
print("=" * 60)

columns = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

for col in columns:
    print(f"{col:25s}: {(diabetes[col] == 0).sum()}")