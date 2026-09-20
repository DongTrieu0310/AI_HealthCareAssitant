import pandas as pd
from pathlib import Path


DATA_DIR = Path("../../data/raw")


files = list(DATA_DIR.glob("*.csv"))

print("Number of datasets:", len(files))

for file in files:

    print("\n" + "=" * 60)
    print("FILE:", file.name)
    print("=" * 60)

    df = pd.read_csv(file)

    print("Shape:", df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nFirst 5 rows:")
    print(df.head())