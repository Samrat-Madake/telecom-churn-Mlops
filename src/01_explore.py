
import pandas as pd
from pathlib import Path

# Locate the project directory and dataset
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

# Load dataset
df = pd.read_csv(DATA_PATH)

# Basic inspection
print("\n===== DATASET SHAPE =====")
print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")

print("\n===== FIRST 5 ROWS =====")
print(df.head())

print("\n===== COLUMN NAMES =====")
print(df.columns.tolist())

print("\n===== DATA TYPES =====")
print(df.dtypes)

print("\n===== DATASET INFORMATION =====")
df.info()

print("\n===== TARGET DISTRIBUTION =====")
print(df["Churn"].value_counts(dropna=False))

print("\n===== TARGET DISTRIBUTION (%) =====")
print(
    (df["Churn"].value_counts(normalize=True, dropna=False) * 100)
    .round(2)
)
###

# (.venv) PS C:\Users\DELL\Desktop\SEM 7\ML Devops Lab\telecom-churn-mlops> python src/01_explore.py

# ===== DATASET SHAPE =====
# Rows: 7043
# Columns: 21

# ===== FIRST 5 ROWS =====
#    customerID  gender  ... TotalCharges Churn
# 0  7590-VHVEG  Female  ...        29.85    No
# 1  5575-GNVDE    Male  ...       1889.5    No
# 2  3668-QPYBK    Male  ...       108.15   Yes
# 3  7795-CFOCW    Male  ...      1840.75    No
# 4  9237-HQITU  Female  ...       151.65   Yes

# [5 rows x 21 columns]

# ===== COLUMN NAMES =====
# ['customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 
# 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 
# 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 
# 'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges', 'TotalCharges', 'Churn']

# ===== DATA TYPES =====
# customerID              str
# gender                  str
# SeniorCitizen         int64
# Partner                 str
# Dependents              str
# tenure                int64
# PhoneService            str
# MultipleLines           str
# InternetService         str
# OnlineSecurity          str
# OnlineBackup            str
# DeviceProtection        str
# TechSupport             str
# StreamingTV             str
# StreamingMovies         str
# Contract                str
# PaperlessBilling        str
# PaymentMethod           str
# MonthlyCharges      float64
# TotalCharges            str
# Churn                   str
# dtype: object

# ===== DATASET INFORMATION =====
# <class 'pandas.DataFrame'>
# RangeIndex: 7043 entries, 0 to 7042
# Data columns (total 21 columns):
#  #   Column            Non-Null Count  Dtype  
# ---  ------            --------------  -----  
#  0   customerID        7043 non-null   str    
#  1   gender            7043 non-null   str    
#  2   SeniorCitizen     7043 non-null   int64  
#  3   Partner           7043 non-null   str    
#  4   Dependents        7043 non-null   str    
#  5   tenure            7043 non-null   int64  
#  6   PhoneService      7043 non-null   str    
#  7   MultipleLines     7043 non-null   str    
#  8   InternetService   7043 non-null   str    
#  9   OnlineSecurity    7043 non-null   str    
#  10  OnlineBackup      7043 non-null   str    
#  11  DeviceProtection  7043 non-null   str    
#  12  TechSupport       7043 non-null   str    
#  13  StreamingTV       7043 non-null   str    
#  14  StreamingMovies   7043 non-null   str    
#  15  Contract          7043 non-null   str    
#  16  PaperlessBilling  7043 non-null   str    
#  17  PaymentMethod     7043 non-null   str    
#  18  MonthlyCharges    7043 non-null   float64
#  19  TotalCharges      7043 non-null   str    
#  20  Churn             7043 non-null   str    
# dtypes: float64(1), int64(2), str(18)
# memory usage: 1.1 MB

# ===== TARGET DISTRIBUTION =====
# Churn
# No     5174
# Yes    1869
# Name: count, dtype: int64

# ===== TARGET DISTRIBUTION (%) =====
# Churn
# No     73.46
# Yes    26.54
# Name: proportion, dtype: float64
