import requests
import pandas as pd
import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from feast import FeatureStore
from dotenv import load_dotenv

load_dotenv()

KSERVE_URL = os.environ.get("KSERVE_URL", "http://localhost:7070/v1/models/employee_attrition_prediction:predict")
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI_EXTERNAL", "http://localhost:5000")
MLFLOW_EXPERIMENT_NAME = os.environ.get("MLFLOW_EXPERIMENT_NAME", "employee-attrition-v1")
FEAST_REPO_PATH = os.environ.get("FEAST_REPO_PATH", "./_feast/feature_repo")

# Initialize Feast store once at startup
store = FeatureStore(repo_path="./_feast/feature_repo")
feature_service = store.get_feature_service("employee_attrition_features")
TITLE_TO_SNAKE = {
    "Years at Company": "years_at_company",
    "Performance Rating": "performance_rating",
    "Number of Promotions": "number_of_promotions",
    "Overtime": "overtime",
    "Education Level": "education_level",
    "Number of Dependents": "number_of_dependents",
    "Job Level": "job_level",
    "Company Size": "company_size",
    "Company Tenure": "company_tenure",
    "Remote Work": "remote_work",
    "Company Reputation": "company_reputation",
    "OverallSatisfaction": "overall_satisfaction",
    "Opportunities": "opportunities",
    "AnnualIncome": "annual_income",
    "AgeGroup": "age_group",
    "RoleStagnationRatio": "role_stagnation_ratio",
    "TenureGap": "tenure_gap",
    "EarlyCompanyTenureRisk": "early_company_tenure_risk",
    "LongTenureLowRoleRisk": "long_tenure_low_role_risk",
    "Employee ID": "employee_id",
    "Attrition": "attrition"
}


# features schema data (snake_case to match training pipeline)
features = [
    "years_at_company",
    "performance_rating",
    "number_of_promotions",
    "overtime",
    "education_level",
    "number_of_dependents",
    "job_level",
    "company_size",
    "company_tenure",
    "remote_work",
    "company_reputation",
    "overall_satisfaction",
    "opportunities",
    "annual_income",
    "age_group",
    "role_stagnation_ratio",
    "tenure_gap",
    "early_company_tenure_risk",
    "long_tenure_low_role_risk",
]

app = Flask(__name__)
CORS(app)


@app.route('/')
def index():
    return render_template('index.html')



@app.route("/features/<employeeId>", methods=["GET"])
def get_features(employeeId):
    print(employeeId)

    feast_response = store.get_online_features(
        features=feature_service,
        entity_rows=[{"employee_id": employeeId}]
    ).to_dict()

    print('feast-response: ', feast_response)

<<<<<<< Updated upstream:frontend/app.py
    feast_features = {
        k: v[0] for k, v in feast_response.items()
        if k != "employee_id"
    }
=======
    # Check if features are found (Feast returns None/null if key not in online store)
    is_found = any(v[0] is not None for k, v in feast_response.items() if k != "employee_id")

    if not is_found:
        return jsonify({"error": "Employee not found in feature store"}), 404

    feast_features = {}
    for k, v in feast_response.items():
        if k != "employee_id":
            snake_k = TITLE_TO_SNAKE.get(k, k.lower().replace(" ", "_"))
            feast_features[snake_k] = v[0]

>>>>>>> Stashed changes:workdir/kubeflow-training-pipeline/frontend/app.py
    return jsonify({
        "employee_id": employeeId,
        "features": feast_features  # optional: for UI display
    })

    
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    print('incoming-data: ', data)

    # Compute engineered features (snake_case to match training pipeline)
    years_at_company = data["years_at_company"]
    company_tenure = data["company_tenure"]
    job_level = data["job_level"]

    data["role_stagnation_ratio"] = round(years_at_company / (company_tenure + 1), 3)
    data["tenure_gap"] = round(company_tenure - years_at_company, 2)
    data["early_company_tenure_risk"] = 1 if years_at_company <= 2 else 0
    data["long_tenure_low_role_risk"] = 1 if (company_tenure > 5 and job_level <= 2) else 0

    try:
        missing = set(features) - set(data.keys())
        if missing:
            raise ValueError(f"Missing features: {missing}")

<<<<<<< Updated upstream:frontend/app.py
        df_input = pd.Dataframe([data]).reindex(columns=feature_columns)
=======
        cols = features + ["employee_id"]
        df_input = pd.DataFrame([data]).reindex(columns=cols).astype(float)
>>>>>>> Stashed changes:workdir/kubeflow-training-pipeline/frontend/app.py
        print('df-input: ', df_input.to_dict(orient="records"))


        response = requests.post(
            KSERVE_URL, 
            json={"instances": [df_input.to_dict(orient="records")]}
        )
        print('results: ', response.json())

        prediction_result = response.json()
        # { predictions: [1] }
        
        payload = { 
            "prediction": prediction_result['predictions'][0], 
        }

        return payload
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=4000, debug=True)