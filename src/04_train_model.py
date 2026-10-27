
import os
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------
df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

print("Dataset shape:", df.shape)

# --------------------------------------------------
# 2. Clean TotalCharges
# --------------------------------------------------
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"].astype(str).str.strip(),
    errors="coerce"
)

print("Missing TotalCharges:", df["TotalCharges"].isna().sum())

# --------------------------------------------------
# 3. Separate features and target
# --------------------------------------------------
# customerID is an identifier, not a useful predictive feature.
X = df.drop(columns=["customerID", "Churn"])
y = df["Churn"].map({"No": 0, "Yes": 1})

if y.isna().any():
    raise ValueError("Unexpected values found in the Churn target.")

# Identify numerical and categorical columns.
numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

print("Numerical features:", numeric_features)
print("Categorical features:", categorical_features)

# --------------------------------------------------
# 4. Split before fitting preprocessing
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# --------------------------------------------------
# 5. Build preprocessing pipelines
# --------------------------------------------------
numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features),
    ]
)

# --------------------------------------------------
# 6. Build and train Logistic Regression
# --------------------------------------------------
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=2000)),
    ]
)

print("\nTraining Logistic Regression...")
model.fit(X_train, y_train)

# --------------------------------------------------
# 7. Generate predictions
# --------------------------------------------------
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# --------------------------------------------------
# 8. Evaluate model
# --------------------------------------------------
metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred, zero_division=0),
    "recall": recall_score(y_test, y_pred, zero_division=0),
    "f1_score": f1_score(y_test, y_pred, zero_division=0),
    "roc_auc": roc_auc_score(y_test, y_prob),
}

print("\n===== MODEL EVALUATION =====")
for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")

print("\n===== CLASSIFICATION REPORT =====")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["No Churn", "Churn"],
        zero_division=0,
    )
)

print("\n===== CONFUSION MATRIX =====")
print(confusion_matrix(y_test, y_pred))

# --------------------------------------------------
# 9. Save model and evaluation results
# --------------------------------------------------
os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)

joblib.dump(model, "models/churn_model.joblib")

with open("reports/model_metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=4)

print("\nModel saved to: models/churn_model.joblib")
print("Metrics saved to: reports/model_metrics.json")
print("\nModel training completed successfully!")



# ////////////////
# -mlops> python src/04_train_model.py
# Dataset shape: (7043, 21)
# Missing TotalCharges: 11
# Numerical features: ['SeniorCitizen', 'tenure', 'MonthlyCharges','TotalCharges']
# Categorical features: ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']
# Training samples: 5634
# Testing samples: 1409

# Training Logistic Regression...

# ===== MODEL EVALUATION =====
# accuracy: 0.8055
# precision: 0.6572
# recall: 0.5588
# f1_score: 0.6040
# roc_auc: 0.8419

# ===== CLASSIFICATION REPORT =====
#               precision    recall  f1-score   support

#     No Churn       0.85      0.89      0.87      1035
#        Churn       0.66      0.56      0.60       374

#     accuracy                           0.81      1409
#    macro avg       0.75      0.73      0.74      1409
# weighted avg       0.80      0.81      0.80      1409


# ===== CONFUSION MATRIX =====
# [[926 109]
#  [165 209]]

# Model saved to: models/churn_model.joblib
# Metrics saved to: reports/model_metrics.json

# Model training completed successfully!