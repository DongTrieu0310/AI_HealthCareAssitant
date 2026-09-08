import pandas as pd


# ============================================================
# 1. LOAD DATA
# ============================================================

cardio = pd.read_csv("../../data/raw/cardio.csv")

diabetes = pd.read_csv("../../data/raw/diabetes.csv")

hypertension = pd.read_csv("../../data/raw/hypertension.csv")


# ============================================================
# 2. DATASETS
# ============================================================

datasets = {
    "CARDIO": cardio,
    "DIABETES": diabetes,
    "HYPERTENSION": hypertension
}


# ============================================================
# 3. BASIC DATA AUDIT
# ============================================================

for name, df in datasets.items():

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    print("\nShape:")
    print(df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nDuplicate rows:")
    print(df.duplicated().sum())

    print("\nFirst 5 rows:")
    print(df.head())


# ============================================================
# 4. TARGET ANALYSIS
# ============================================================

targets = {
    "CARDIO": "cardio",
    "DIABETES": "Outcome",
    "HYPERTENSION": "Risk"
}


for name, target in targets.items():

    df = datasets[name]

    print("\n" + "=" * 60)
    print(name, "- TARGET ANALYSIS")
    print("=" * 60)

    print("\nTarget:")
    print(target)

    print("\nTarget values:")
    print(df[target].value_counts())

    print("\nTarget percentage:")
    print(
        df[target].value_counts(normalize=True) * 100
    )


# ============================================================
# 5. HYPERTENSION RISK ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("HYPERTENSION - RISK ANALYSIS")
print("=" * 60)


# ------------------------------------------------------------
# 5.1 Risk vs Systolic Blood Pressure
# ------------------------------------------------------------

print("\nRisk vs Systolic Blood Pressure:")

print(
    pd.crosstab(
        pd.cut(
            hypertension["sysBP"],
            bins=[0, 120, 130, 140, 160, 300]
        ),
        hypertension["Risk"],
        normalize="index"
    )
)


# ------------------------------------------------------------
# 5.2 Risk vs Diastolic Blood Pressure
# ------------------------------------------------------------

print("\nRisk vs Diastolic Blood Pressure:")

print(
    pd.crosstab(
        pd.cut(
            hypertension["diaBP"],
            bins=[0, 80, 90, 100, 130]
        ),
        hypertension["Risk"],
        normalize="index"
    )
)
print("\n" + "=" * 60)
print("RISK vs SBP EXACT ANALYSIS")
print("=" * 60)

print(
    hypertension.groupby("Risk")["sysBP"].agg(
        ["min", "max", "mean", "median", "std"]
    )
)


print("\n" + "=" * 60)
print("RISK vs DBP EXACT ANALYSIS")
print("=" * 60)

print(
    hypertension.groupby("Risk")["diaBP"].agg(
        ["min", "max", "mean", "median", "std"]
    )
)
print("\n" + "=" * 60)
print("RISK DISTRIBUTION BY SBP")
print("=" * 60)

print(
    hypertension.groupby("sysBP")["Risk"]
    .agg(["count", "mean"])
    .head(30)
)
print("\n" + "=" * 60)
print("POTENTIAL LABEL RULE")
print("=" * 60)

for threshold in [120, 130, 140, 150, 160]:

    predicted = (hypertension["sysBP"] >= threshold).astype(int)

    accuracy = (predicted == hypertension["Risk"]).mean()

    print(
        f"SBP >= {threshold}: "
        f"match = {accuracy:.4f}"
    )