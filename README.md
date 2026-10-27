
# Telecom Customer Churn Prediction — ML DevOps

An end-to-end Machine Learning and MLOps project that analyses telecom customer churn, trains a classification model, tracks experiments using MLflow, and detects feature-level data drift using Evidently.

## Project Objectives

- Explore and validate the telecom customer churn dataset.
- Perform exploratory data analysis (EDA).
- Build a machine learning pipeline to predict customer churn.
- Evaluate the model using classification metrics.
- Track model parameters, metrics, and artifacts using MLflow.
- Generate data drift reports using Evidently.
- Maintain a reproducible project structure.

## Tech Stack

- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib and Seaborn
- MLflow
- Evidently
- Joblib

## Project Structure

```text
telecom-churn-mlops/
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── models/
│   └── churn_model.joblib
├── notebooks/
├── reports/
│   ├── churn_distribution.png
│   ├── churn_by_contract.png
│   ├── churn_by_tenure.png
│   ├── model_metrics.json
│   ├── data_drift_report.html
│   └── data_drift_results.json
├── src/
│   ├── 01_explore.py
│   ├── 02_data_quality.py
│   ├── 03_eda.py
│   ├── 04_train_model.py
│   ├── 05_mlflow_tracking.py
│   └── 06_data_drift.py
├── .gitignore
├── requirements.txt
└── README.md
```

## Dataset

The project uses the IBM Telco Customer Churn dataset.

- Records: 7,043
- Features: 19 input features after excluding the customer ID and target column
- Target: `Churn`
- Classes: `Yes` and `No`

The dataset contains customer account information, subscribed services, contract details, and billing information.

Place the CSV file at:

`data/WA_Fn-UseC_-Telco-Customer-Churn.csv`

## Installation

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd telecom-churn-mlops
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Running the Project

Run the scripts from the project root in the following order:

```bash
python src/01_explore.py
python src/02_data_quality.py
python src/03_eda.py
python src/04_train_model.py
python src/05_mlflow_tracking.py
python src/06_data_drift.py
```

Ensure the dataset is present before running the scripts. Each script may generate or update files under `reports/`, `models/`, or the MLflow tracking store.

## Model Performance

The baseline Logistic Regression model was evaluated using a stratified 80/20 train-test split with `random_state=42`.

| Metric | Score |
|---|---:|
| Accuracy | 0.8055 |
| Precision | 0.6572 |
| Recall | 0.5588 |
| F1-score | 0.6040 |
| ROC-AUC | 0.8419 |

The model achieved an ROC-AUC of approximately 0.842. Churn recall is approximately 0.559, meaning some customers who churned were not identified by the baseline model.

## MLflow Experiment Tracking

MLflow is used to record model parameters, evaluation metrics, and model artifacts.

The experiment name is:

`Telecom-Customer-Churn`

The project uses a local SQLite tracking database. Start the MLflow UI from the project root:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Open `http://127.0.0.1:5000` in your browser.

## Data Drift Detection

Evidently compares a reference dataset with a reproducibly generated current dataset.

The experiment modifies `MonthlyCharges` and `tenure` to test the drift detection workflow. In the observed run:

- Reference records: 7,043
- Current records: 7,043
- Features analysed: 19
- Drifted features: 1
- Drifted feature share: approximately 5.26%
- `MonthlyCharges` drift score: approximately 0.429

The dataset-level drift threshold is based on the share of drifted features. A feature can exhibit drift without the entire dataset being classified as drifted.

**Important:** The current dataset is simulated for testing. These results demonstrate the drift detection pipeline and do not establish that actual production customer data has drifted.

Generated outputs:

- `reports/data_drift_report.html`
- `reports/data_drift_results.json`

## Limitations and Future Improvements

- Evaluate additional classification algorithms.
- Tune the classification threshold to improve churn recall.
- Use real incoming data for production drift monitoring.
- Add automated testing and a CI/CD pipeline.
- Introduce model versioning and a deployment workflow.
- Monitor model performance when actual labels become available.

## Disclaimer

This project is intended for educational and demonstration purposes. Model results depend on the dataset and experimental setup. Data drift alone does not necessarily mean that model performance has degraded.
