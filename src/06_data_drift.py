
import os
import json
import numpy as np
import pandas as pd
import mlflow

from evidently import Report
from evidently.presets import DataDriftPreset


# --------------------------------------------------
# 1. Load reference dataset
# --------------------------------------------------
DATA_PATH = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"

df = pd.read_csv(DATA_PATH)

# Exclude customer ID and target from feature drift analysis.
feature_columns = [
    col for col in df.columns
    if col not in ["customerID", "Churn"]
]

reference_data = df[feature_columns].copy()

# Convert TotalCharges to numeric.
reference_data["TotalCharges"] = pd.to_numeric(
    reference_data["TotalCharges"].astype(str).str.strip(),
    errors="coerce"
)


# --------------------------------------------------
# 2. Create simulated current data
# --------------------------------------------------
# Fixed seeds make the experiment reproducible.
rng = np.random.default_rng(42)

current_data = reference_data.sample(
    n=len(reference_data),
    replace=True,
    random_state=42
).reset_index(drop=True)

# Simulate a change in customer tenure.
current_data["tenure"] = np.clip(
    current_data["tenure"] + rng.integers(
        -12, 13, size=len(current_data)
    ),
    0,
    72
)

# Simulate a 20% increase in monthly charges.
current_data["MonthlyCharges"] = (
    current_data["MonthlyCharges"] * 1.20
).clip(lower=0)


# --------------------------------------------------
# 3. Configure MLflow
# --------------------------------------------------
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Telecom-Customer-Churn")


# --------------------------------------------------
# 4. Generate data drift report
# --------------------------------------------------
os.makedirs("reports", exist_ok=True)

report = Report([
    DataDriftPreset()
])

result = report.run(
    reference_data=reference_data,
    current_data=current_data
)

# Save interactive HTML report.
report_path = "reports/data_drift_report.html"
result.save_html(report_path)

# Convert Evidently results to a dictionary.
result_dict = result.dict()

# Save machine-readable JSON report.
json_path = "reports/data_drift_results.json"

with open(json_path, "w", encoding="utf-8") as file:
    json.dump(result_dict, file, indent=2, default=str)


# --------------------------------------------------
# 5. Extract drift metrics dynamically
# --------------------------------------------------
drift_summary = next(
    (
        metric
        for metric in result_dict["metrics"]
        if metric["metric_name"].startswith("DriftedColumnsCount")
    ),
    None
)

if drift_summary is None:
    raise RuntimeError(
        "Could not find DriftedColumnsCount in Evidently results."
    )

drifted_columns_count = float(
    drift_summary["value"]["count"]
)

drifted_columns_share = float(
    drift_summary["value"]["share"]
)

# Extract per-feature drift scores.
feature_drift_scores = {}

for metric in result_dict["metrics"]:
    metric_name = metric.get("metric_name", "")

    if metric_name.startswith("ValueDrift("):
        config = metric.get("config", {})
        column_name = config.get("column")
        value = metric.get("value")

        if column_name and isinstance(value, (int, float)):
            # Use valid MLflow metric names.
            safe_column_name = (
                column_name.lower()
                .replace(" ", "_")
                .replace("-", "_")
            )

            feature_drift_scores[safe_column_name] = float(value)


# --------------------------------------------------
# 6. Print drift summary
# --------------------------------------------------
print("\n===== DATA DRIFT ANALYSIS =====")
print("Reference records:", len(reference_data))
print("Current records:", len(current_data))
print("Features analysed:", len(feature_columns))
print("Report saved to:", report_path)

print("\n===== EXTRACTED DRIFT METRICS =====")
print("Drifted columns:", int(drifted_columns_count))
print(f"Share of drifted columns: {drifted_columns_share:.4f}")

print("\n===== FEATURE DRIFT SCORES =====")

for feature, score in feature_drift_scores.items():
    print(f"{feature}: {score:.6f}")


# --------------------------------------------------
# 7. Log drift metrics and artifacts to MLflow
# --------------------------------------------------
with mlflow.start_run(run_name="Data-Drift-Detection"):

    # Log parameters.
    mlflow.log_param(
        "drift_method",
        "Evidently DataDriftPreset"
    )
    mlflow.log_param("reference_rows", len(reference_data))
    mlflow.log_param("current_rows", len(current_data))
    mlflow.log_param("features_analysed", len(feature_columns))
    mlflow.log_param("simulated_tenure_shift", True)
    mlflow.log_param("simulated_monthly_charges_shift", True)

    # Log dataset-level metrics.
    mlflow.log_metric(
        "drifted_columns_count",
        drifted_columns_count
    )
    mlflow.log_metric(
        "drifted_columns_share",
        drifted_columns_share
    )

    # Log each feature's drift score.
    for feature, score in feature_drift_scores.items():
        mlflow.log_metric(
            f"drift_score_{feature}",
            score
        )

    # Log report artifacts.
    mlflow.log_artifact(report_path)
    mlflow.log_artifact(json_path)

    run_id = mlflow.active_run().info.run_id

    print("\nMLflow Run ID:", run_id)


# --------------------------------------------------
# 8. Completion summary
# --------------------------------------------------
print("\n===== COMPLETION SUMMARY =====")
print("HTML report:", report_path)
print("JSON report:", json_path)
print("MLflow experiment: Telecom-Customer-Churn")
print("Results logged to MLflow successfully!")
