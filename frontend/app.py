import requests
import pandas as pd
import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

KSERVE_URL = os.environ.get("KSERVE_URL", "http://localhost:7070/v1/models/employee_attrition_prediction:predict")

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