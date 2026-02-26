import requests
import pandas as pd
import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

KSERVE_URL = os.environ.get("KSERVE_URL", "http://localhost:7070/v1/models/employee_attrition_prediction:predict")

# features schema data from mlflow
features = [
    "Years at Company",
    "Performance Rating",
    "Number of Promotions",
    "Overtime",
    "Education Level",
    "Number of Dependents",
    "Job Level",
    "Company Size",
    "Company Tenure",
    "Remote Work",
    "Company Reputation",
    "OverallSatisfaction",
    "Opportunities",
    "AnnualIncome",
    "AgeGroup",
    "RoleStagnationRatio",
    "TenureGap",
    "EarlyCompanyTenureRisk",
    "LongTenureLowRoleRisk"
]

app = Flask(__name__)
CORS(app)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    print('incoming-data: ', data)

    # Compute engineered features
    years_at_company = data["Years at Company"]
    company_tenure = data["Company Tenure"]
    job_level = data["Job Level"]

    data["RoleStagnationRatio"] = round(years_at_company / (company_tenure + 1), 3)
    data["TenureGap"] = round(company_tenure - years_at_company, 2)
    data["EarlyCompanyTenureRisk"] = 1 if years_at_company <= 2 else 0
    data["LongTenureLowRoleRisk"] = 1 if (company_tenure > 5 and job_level <= 2) else 0


    try:
        missing = set(features) - set(data.keys())
        if missing:
            raise ValueError(f"Missing features: {missing}")

        df_input = pd.DataFrame([[data.get(f) for f in features]], columns=features)

        response = requests.post(
            KSERVE_URL, 
            json={"instances": df_input.to_dict(orient="records")}
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