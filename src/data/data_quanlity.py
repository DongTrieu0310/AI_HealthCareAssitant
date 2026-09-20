import pandas as pd


# ============================================================
# LOAD DATA
# ============================================================

cardio = pd.read_csv("../../data/raw/cardio.csv")
diabetes = pd.read_csv("../../data/raw/diabetes.csv")
hypertension = pd.read_csv("../../data/raw/hypertension.csv")

print("\n" + "=" * 60)
print("DIABETES - ZERO VALUE ANALYSIS")
print("=" * 60)

print(
    (diabetes == 0).sum()
)

# ============================================================
# DIABETES - RANGE ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("DIABETES - RANGE ANALYSIS")
print("=" * 60)

columns_to_check = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age"
]

print(
    diabetes[columns_to_check].describe().T
)

# ============================================================
# DIABETES - OUTLIER ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("DIABETES - OUTLIER ANALYSIS")
print("=" * 60)

for column in columns_to_check:

    Q1 = diabetes[column].quantile(0.25)
    Q3 = diabetes[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = diabetes[
        (diabetes[column] < lower_bound) |
        (diabetes[column] > upper_bound)
    ]

    print(f"\n{column}")

    print(f"Q1          : {Q1:.2f}")
    print(f"Q3          : {Q3:.2f}")
    print(f"IQR         : {IQR:.2f}")
    print(f"Lower bound : {lower_bound:.2f}")
    print(f"Upper bound : {upper_bound:.2f}")
    print(f"Outliers    : {len(outliers)}")

# ============================================================
# INVESTIGATE IMPORTANT OUTLIERS
# ============================================================

print("\n" + "=" * 60)
print("DIABETES - IMPORTANT OUTLIERS")
print("=" * 60)

for column in ["SkinThickness", "Insulin", "BloodPressure", "BMI"]:

    Q1 = diabetes[column].quantile(0.25)
    Q3 = diabetes[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = diabetes[
        (diabetes[column] < lower_bound) |
        (diabetes[column] > upper_bound)
    ]

    print(f"\n{'-' * 50}")
    print(f"Feature: {column}")
    print(f"Number of outliers: {len(outliers)}")

    display_columns = [
        column,
        "Glucose",
        "BMI",
        "Age",
        "Outcome"
    ]

    display_columns = list(dict.fromkeys(display_columns))

    print(
        outliers[
            display_columns
        ].sort_values(
            by=column,
            ascending=False
        ).head(10)
    )
    print("\n" + "=" * 60)
    print("DIABETES - ZERO VALUE PERCENTAGE")
    print("=" * 60)

    zero_columns = [
        "Pregnancies",
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
        "DiabetesPedigreeFunction",
        "Age"
    ]

    for column in zero_columns:
        zero_count = (diabetes[column] == 0).sum()
        zero_percent = zero_count / len(diabetes) * 100

        print(
            f"{column:30s} "
            f"Count = {zero_count:4d} "
            f"Percentage = {zero_percent:6.2f}%"
        )
print("\n" + "=" * 60)
print("DIABETES - PREGNANCIES OUTLIERS")
print("=" * 60)

preg_outliers = diabetes[
    diabetes["Pregnancies"] > 13.5
]

print(preg_outliers)
preg_outliers = diabetes[
    diabetes["Pregnancies"] > 13.5
]

print(preg_outliers)
print("\n" + "=" * 60)
print("DIABETES - CORRELATION ANALYSIS")
print("=" * 60)

correlation = diabetes.corr(numeric_only=True)

print(
    correlation["Outcome"]
    .sort_values(ascending=False)
)

print("\n" + "=" * 60)
print("DIABETES - FEATURE CORRELATION")
print("=" * 60)

feature_corr = diabetes.drop(columns=["Outcome"]).corr()

print(feature_corr.round(3))
print("\n" + "=" * 60)
print("DIABETES - HIGH FEATURE CORRELATION")
print("=" * 60)

feature_corr = diabetes.drop(columns=["Outcome"]).corr()

threshold = 0.5

for i in range(len(feature_corr.columns)):
    for j in range(i + 1, len(feature_corr.columns)):

        corr_value = feature_corr.iloc[i, j]

        if abs(corr_value) >= threshold:

            print(
                f"{feature_corr.columns[i]:30}"
                f"{feature_corr.columns[j]:30}"
                f"{corr_value:.3f}"
            )
print("\n" + "=" * 60)
print("DIABETES - LOGICAL VALUE CHECK")
print("=" * 60)

checks = {
    "Pregnancies < 0": (diabetes["Pregnancies"] < 0).sum(),
    "Glucose <= 0": (diabetes["Glucose"] <= 0).sum(),
    "BloodPressure <= 0": (diabetes["BloodPressure"] <= 0).sum(),
    "SkinThickness < 0": (diabetes["SkinThickness"] < 0).sum(),
    "Insulin < 0": (diabetes["Insulin"] < 0).sum(),
    "BMI <= 0": (diabetes["BMI"] <= 0).sum(),
    "DiabetesPedigreeFunction < 0": (
        diabetes["DiabetesPedigreeFunction"] < 0
    ).sum(),
    "Age <= 0": (diabetes["Age"] <= 0).sum(),
}

for check, count in checks.items():
    print(f"{check:35} : {count}")

print("\n" + "=" * 60)
print("DIABETES - RAW INSULIN QUALITY CHECK")
print("=" * 60)

print("\nInsulin == 0:")
print(diabetes["Insulin"].eq(0).sum())

print("\n" + "=" * 60)
print("DIABETES - INSULIN MEDIAN VALUE CHECK")
print("=" * 60)

median_value = diabetes["Insulin"].median()

print("Median:", median_value)

print(
    "Number of values equal to median:",
    (diabetes["Insulin"] == median_value).sum()
)

print(
    "Percentage:",
    round(
        (diabetes["Insulin"] == median_value).mean() * 100,
        2
    ),
    "%"
)

print("\n" + "=" * 60)
print("DIABETES - SUSPICIOUS VALUE CHECK")
print("=" * 60)

for column in [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]:

    print(f"\n{column}")

    print(
        "Median:",
        diabetes[column].median()
    )

    print(
        "Values equal median:",
        (diabetes[column] == diabetes[column].median()).sum()
    )

print("\n" + "=" * 60)
print("DIABETES - INSULIN MEDIAN VALUE BY OUTCOME")
print("=" * 60)

median_value = diabetes["Insulin"].median()

median_insulin = diabetes[
    diabetes["Insulin"] == median_value
]

print(
    median_insulin["Outcome"].value_counts()
)

print("\nPercentage by Outcome:")

print(
    pd.crosstab(
        median_insulin["Outcome"],
        columns="count"
    )
)

print("\nInsulin <= 0:")
print(diabetes["Insulin"].le(0).sum())

print("\nInsulin missing:")
print(diabetes["Insulin"].isna().sum())

print("\nInsulin distribution:")
print(diabetes["Insulin"].describe())

print("\n" + "=" * 60)
print("DIABETES - INSULIN 102.5 ANALYSIS")
print("=" * 60)

insulin_102 = diabetes[
    diabetes["Insulin"] == 102.5
]

print("\nNumber of records:", len(insulin_102))

print("\nOutcome distribution:")
print(
    insulin_102["Outcome"].value_counts()
)

print("\nInsulin 102.5 statistics of other features:")

print(
    insulin_102[
        [
            "Glucose",
            "BloodPressure",
            "BMI",
            "Age",
            "Outcome"
        ]
    ].describe()
)

print("\n" + "=" * 60)
print("DIABETES - INSULIN VALUE FREQUENCY")
print("=" * 60)

print(
    diabetes["Insulin"]
    .value_counts()
    .head(20)
)
print("\n" + "=" * 60)
print("DIABETES - KEY INSULIN VALUES")
print("=" * 60)

for value in [102.5, 169.5]:

    subset = diabetes[
        diabetes["Insulin"] == value
    ]

    print(f"\nInsulin = {value}")
    print("Count:", len(subset))

    print("\nOutcome:")
    print(
        subset["Outcome"].value_counts()
    )
diabetes["Insulin"].value_counts().head(20)
for value in [102.5, 169.5]:
    subset = diabetes[diabetes["Insulin"] == value]

    print(f"\nInsulin = {value}")
    print("Count:", len(subset))
    print(subset["Outcome"].value_counts())

print("\n" + "=" * 60)
print("DIABETES - INSULIN DUPLICATION CHECK")
print("=" * 60)

insulin_counts = diabetes["Insulin"].value_counts()

print("\nTop Insulin values:")
print(insulin_counts.head(20))

print("\n" + "=" * 60)
print("DIABETES - INSULIN vs OUTCOME")
print("=" * 60)

insulin_outcome = (
    diabetes
    .groupby("Insulin")["Outcome"]
    .agg(["count", "mean", "min", "max"])
    .sort_values("count", ascending=False)
)

print(insulin_outcome.head(20))

print("\n" + "=" * 60)
print("DIABETES - SUSPICIOUS INSULIN VALUES")
print("=" * 60)

suspicious = diabetes[
    diabetes["Insulin"].isin([102.5, 169.5])
]

print(suspicious[
    [
        "Insulin",
        "Glucose",
        "BloodPressure",
        "BMI",
        "DiabetesPedigreeFunction",
        "Age",
        "Outcome"
    ]
].to_string(index=False))

print("=" * 60)
print("INSULIN RAW VALUE INVESTIGATION")
print("=" * 60)

print("\nUnique Insulin values:")
print(diabetes["Insulin"].nunique())

print("\nMost frequent Insulin values:")
print(diabetes["Insulin"].value_counts().head(20))

print("\nCheck 102.5:")
print((diabetes["Insulin"] == 102.5).sum())

print("\nCheck 169.5:")
print((diabetes["Insulin"] == 169.5).sum())

print("\nDecimal Insulin values:")
print(
    diabetes[
        diabetes["Insulin"].apply(
            lambda x: isinstance(x, float) and x % 1 != 0
        )
    ]["Insulin"].value_counts().head(20)
)

print(diabetes["Insulin"].head(20))
print(diabetes["Insulin"].describe())
print(diabetes["Insulin"].isna().sum())
print(diabetes["Insulin"].value_counts().head(10))

print("INSULIN RAW FILE CHECK")
print("=" * 60)

print("\nShape:")
print(diabetes.shape)

print("\nFirst 10 Insulin values:")
print(diabetes["Insulin"].head(10))

print("\nNumber of unique Insulin values:")
print(diabetes["Insulin"].nunique())

print("\nTop 20 Insulin values:")
print(diabetes["Insulin"].value_counts().head(20))

print("\nCheck 102.5:")
print((diabetes["Insulin"] == 102.5).sum())

print("\nCheck 169.5:")
print((diabetes["Insulin"] == 169.5).sum())

print("\nZero Insulin:")
print((diabetes["Insulin"] == 0).sum())

print("\nNaN Insulin:")
print(diabetes["Insulin"].isna().sum())

print("INSULIN SUSPICIOUS VALUE POSITION")
print("=" * 60)

print("\nPositions of 102.5:")
print(diabetes.index[diabetes["Insulin"] == 102.5].tolist()[:50])

print("\nPositions of 169.5:")
print(diabetes.index[diabetes["Insulin"] == 169.5].tolist()[:50])

print("\nINSULIN vs OUTCOME")
print("=" * 60)

print(
    diabetes.groupby("Insulin")["Outcome"]
    .agg(["count", "mean", "min", "max"])
    .loc[[102.5, 169.5]]
)

print("\nSUSPICIOUS INSULIN GROUP PROFILE")
print("=" * 60)

for value in [102.5, 169.5]:

    group = diabetes[diabetes["Insulin"] == value]

    print(f"\nInsulin = {value}")
    print("-" * 40)

    print("Count:", len(group))

    print("\nOutcome:")
    print(group["Outcome"].value_counts())

    print("\nFeature means:")
    print(
        group[
            [
                "Pregnancies",
                "Glucose",
                "BloodPressure",
                "SkinThickness",
                "BMI",
                "DiabetesPedigreeFunction",
                "Age"
            ]
        ].mean()
    )

print("INSULIN MEDIAN BY OUTCOME")
print("=" * 60)

print(
    diabetes.groupby("Outcome")["Insulin"]
    .agg(["count", "mean", "median", "min", "max"])
)

print("\nOVERALL MEDIAN:")
print(diabetes["Insulin"].median())

print("\nOUTCOME = 0 MEDIAN:")
print(diabetes.loc[diabetes["Outcome"] == 0, "Insulin"].median())

print("\nOUTCOME = 1 MEDIAN:")
print(diabetes.loc[diabetes["Outcome"] == 1, "Insulin"].median())