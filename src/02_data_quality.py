
import pandas as pd
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

df = pd.read_csv(DATA_PATH)

# 1. Check missing values
print("\n===== MISSING VALUES =====")
print(df.isnull().sum().sort_values(ascending=False))

# 2. Check empty or whitespace-only values
print("\n===== EMPTY / WHITESPACE VALUES =====")
for col in df.select_dtypes(include=["object", "str"]).columns:
    count = df[col].astype(str).str.strip().eq("").sum()
    if count > 0:
        print(f"{col}: {count}")

# 3. Check duplicate records
print("\n===== DUPLICATE RECORDS =====")
print("Duplicate rows:", df.duplicated().sum())
print("Duplicate customer IDs:", df["customerID"].duplicated().sum())

# 4. Investigate TotalCharges
print("\n===== TOTAL CHARGES CHECK =====")
total_charges = pd.to_numeric(df["TotalCharges"], errors="coerce")

print("Original data type:", df["TotalCharges"].dtype)
print("Values that cannot be converted:", total_charges.isna().sum())
print("Minimum valid value:", total_charges.min())
print("Maximum valid value:", total_charges.max())

# 5. Check numeric ranges
print("\n===== NUMERIC SUMMARY =====")
print(df[["tenure", "MonthlyCharges"]].describe())

# 6. Inspect categorical values
print("\n===== CATEGORICAL VALUE COUNTS =====")
for col in ["gender", "Contract", "InternetService",
            "PaymentMethod", "Churn"]:
    print(f"\n{col}:")
    print(df[col].value_counts(dropna=False))

# 7. Check target consistency
print("\n===== CHURN LABELS =====")
print(df["Churn"].unique())


# ////////////////////

# 8. Investigate blank TotalCharges records
print("\n===== BLANK TOTAL CHARGES RECORDS =====")

blank_mask = df["TotalCharges"].astype(str).str.strip().eq("")

print(
    df.loc[
        blank_mask,
        ["customerID", "tenure", "MonthlyCharges",
         "TotalCharges", "Churn"]
    ].to_string(index=False)
)

print("\n===== ZERO TENURE ANALYSIS =====")
print("Customers with tenure = 0:", (df["tenure"] == 0).sum())
print(
    "Blank TotalCharges with tenure = 0:",
    (blank_mask & (df["tenure"] == 0)).sum()
)

print("\n===== TOTAL CHARGES CONVERSION CHECK =====")
df["TotalCharges_numeric"] = pd.to_numeric(
    df["TotalCharges"].str.strip(),
    errors="coerce"
)

print(
    df.loc[
        df["TotalCharges_numeric"].isna(),
        ["tenure", "MonthlyCharges", "TotalCharges", "Churn"]
    ].to_string(index=False)
)

# ////////////////////

# (.venv) PS C:\Users\DELL\Desktop\SEM 7\ML Devops Lab\telecom-churn-mlops> python src/02_data_quality.py

# ===== MISSING VALUES =====
# customerID          0
# gender              0
# SeniorCitizen       0
# Partner             0
# Dependents          0
# tenure              0
# PhoneService        0
# MultipleLines       0
# InternetService     0
# OnlineSecurity      0
# OnlineBackup        0
# DeviceProtection    0
# TechSupport         0
# StreamingTV         0
# StreamingMovies     0
# Contract            0
# PaperlessBilling    0
# PaymentMethod       0
# MonthlyCharges      0
# TotalCharges        0
# Churn               0
# dtype: int64

# ===== EMPTY / WHITESPACE VALUES =====
# TotalCharges: 11

# ===== DUPLICATE RECORDS =====
# Duplicate rows: 0
# Duplicate customer IDs: 0

# ===== TOTAL CHARGES CHECK =====
# Original data type: str
# Values that cannot be converted: 11
# Minimum valid value: 18.8
# Maximum valid value: 8684.8

# ===== NUMERIC SUMMARY =====
#             tenure  MonthlyCharges
# count  7043.000000     7043.000000
# mean     32.371149       64.761692
# std      24.559481       30.090047
# min       0.000000       18.250000
# 25%       9.000000       35.500000
# 50%      29.000000       70.350000
# 75%      55.000000       89.850000
# max      72.000000      118.750000

# ===== CATEGORICAL VALUE COUNTS =====

# gender:
# gender
# Male      3555
# Female    3488
# Name: count, dtype: int64

# Contract:
# Contract
# Month-to-month    3875
# Two year          1695
# One year          1473
# Name: count, dtype: int64

# InternetService:
# InternetService
# Fiber optic    3096
# DSL            2421
# No             1526
# Name: count, dtype: int64

# PaymentMethod:
# PaymentMethod
# Electronic check             2365
# Mailed check                 1612
# Bank transfer (automatic)    1544
# Credit card (automatic)      1522
# Name: count, dtype: int64

# Churn:
# Churn
# No     5174
# Yes    1869
# Name: count, dtype: int64

# ===== CHURN LABELS =====
# <StringArray>
# ['No', 'Yes']
# Length: 2, dtype: str
