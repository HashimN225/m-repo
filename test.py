import pandas as pd
import joblib
import os
import mlflow
from mlflow import MlflowClient
import json
from dotenv import load_dotenv

load_dotenv()

MLFLOW_TRACKING_URI = os.environ['MLFLOW_TRACKING_URI']
MODEL_NAME = "register-employee-attrition-model"
MODEL_VERSION = 11

def test(input_data: dict):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()

    # Load artifacts
    try:
        model_version_details = client.get_model_version(name=MODEL_NAME, version=MODEL_VERSION)
        run_id = model_version_details.run_id

        model_uri = f"models:/{MODEL_NAME}/{model_version_details.version}"
        model = mlflow.sklearn.load_model(model_uri)

        features_dict = client.download_artifacts(run_id, "features_schema.json", "/tmp")
        with open(features_dict, 'r') as f:
            features = json.load(f)['raw_features']

    except Exception as e:
        print(f"❌ Error loading artifacts: {e}")
        return

    # Convert input to DataFrame with correct column order
    df_input = input_data[features]
    print('df head: \n', df_input.head())

    # Predict probability
    prediction = model.predict(df_input)[0]
    print('prediciton value: ', prediction)
    print("😢 Leave" if prediction == 1 else "😃 Stay")

    probability = model.predict_proba(df_input)[0]
    p_stay = probability[0]
    p_leave = probability[1]

    if p_leave >= 0.65:
        print(f'⏰ Risk: High ({p_leave})')
    elif p_leave >= 0.45:
        print(f'⏰ Risk: Medium ({p_leave})')
    elif p_leave >= 0.25:
        print(f'⏰ Risk: Low ({p_leave})')
    else:
        print(f'⏰ Risk: Very Low ({p_leave})')

    return True


if __name__ == "__main__":
    print("=" * 60)
    print("👔 Employee Attrition Prediction App")
    print("=" * 60)

    # Collect inputs
    print("Load test dataset")
    input_df = pd.read_csv("datasets/test.csv")
    print(f"input-df: {input_df}")

    # Derived features
    input_df["RoleStagnationRatio"] = (input_df["Years at Company"] / (input_df["Company Tenure"] + 1)).round(3)
    input_df["TenureGap"] = (input_df["Company Tenure"] - input_df["Years at Company"]).round(2)
    input_df['EarlyCompanyTenureRisk'] = input_df['Years at Company'].apply(lambda x: 1 if x <=2 else 0)
    input_df['LongTenureLowRoleRisk'] = ((input_df['Company Tenure'] > 5) & (input_df['Job Level'] <= 2)).astype(int)

    print(f"final input-df: {input_df}")  
    # Run test
    test(input_df)