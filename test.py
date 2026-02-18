import pandas as pd
import joblib
import os
import mlflow
from mlflow import MlflowClient
import json
from dotenv import load_dotenv

load_dotenv()

MLFLOW_TRACKING_URI = os.environ["MLFLOW_TRACKING_URI"]
MODEL_NAME = "register-employee-attrition-model"
MODEL_VERSION = 11
# Feast repo path (same as in training); set FEAST_REPO_PATH in .env to override
FEAST_REPO_PATH = os.environ.get("FEAST_REPO_PATH", "_feast/feature_repo")


def get_features_from_feast(
    entity_df: pd.DataFrame,
    feast_repo_path: str = FEAST_REPO_PATH,
    use_online: bool = False,
) -> pd.DataFrame:

    from feast import FeatureStore

    store = FeatureStore(repo_path=feast_repo_path)
    entity_df = entity_df[["employee_id", "event_timestamp"]].copy()
    entity_df["event_timestamp"] = pd.to_datetime(entity_df["event_timestamp"])

    if use_online:
        # Low-latency: fetch latest features from Redis (one entity per row)
        feature_service = store.get_feature_service("employee_attrition_features")
        result = store.get_online_features(
            features=feature_service,
            entity_rows=[{"employee_id": eid} for eid in entity_df["employee_id"]],
        ).to_df()
    else:
        # Batch: fetch historical features (same as training pipeline)
        result = store.get_historical_features(
            entity_df=entity_df,
            features=store.get_feature_service("employee_attrition_features"),
        ).to_df()

    # Drop keys and target so only model features remain
    cols_to_drop = ["employee_id", "event_timestamp", "attrition"]
    for c in cols_to_drop:
        if c in result.columns:
            result = result.drop(columns=[c])
    return result


def test(input_data: pd.DataFrame):
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
    print("df head:\n", df_input.head())

    # Predict (supports single row or batch)
    predictions = model.predict(df_input)
    probabilities = model.predict_proba(df_input)

    for i in range(len(df_input)):
        prediction = predictions[i]
        p_stay, p_leave = probabilities[i][0], probabilities[i][1]
        print(f"  Row {i + 1}: pred={prediction} | P(Stay)={p_stay:.3f} P(Leave)={p_leave:.3f} ", end="")
        print("😢 Leave" if prediction == 1 else "😃 Stay")
        if p_leave >= 0.65:
            print(f"    ⏰ Risk: High ({p_leave:.2f})")
        elif p_leave >= 0.45:
            print(f"    ⏰ Risk: Medium ({p_leave:.2f})")
        elif p_leave >= 0.25:
            print(f"    ⏰ Risk: Low ({p_leave:.2f})")
        else:
            print(f"    ⏰ Risk: Very Low ({p_leave:.2f})")

    return True


def test_with_feast(
    entity_df: pd.DataFrame,
    feast_repo_path: str = FEAST_REPO_PATH,
    use_online: bool = False,
):

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()

    try:
        model_version_details = client.get_model_version(name=MODEL_NAME, version=MODEL_VERSION)
        run_id = model_version_details.run_id
        model_uri = f"models:/{MODEL_NAME}/{model_version_details.version}"
        model = mlflow.sklearn.load_model(model_uri)
        features_path = client.download_artifacts(run_id, "features_schema.json", "/tmp")
        with open(features_path, "r") as f:
            features = json.load(f)["raw_features"]
            
    except Exception as e:
        print(f"❌ Error loading artifacts: {e}")
        return

    print("Fetching features from Feast...")
    df_features = get_features_from_feast(
        entity_df=entity_df,
        feast_repo_path=feast_repo_path,
        use_online=use_online,
    )
    # Align columns with model's expected feature order
    df_input = df_features[[c for c in features if c in df_features.columns]]
    if set(df_input.columns) != set(features):
        missing = set(features) - set(df_input.columns)
        print(f"⚠️ Missing features for model: {missing}")
    print("Feast features shape:", df_input.shape)

    return test(df_input)


if __name__ == "__main__":
    print("=" * 60)
    print("👔 Employee Attrition Prediction App")
    print("=" * 60)

    print("Load test dataset")
    input_df = pd.read_csv("datasets/test.csv")
    print("input-df shape:", input_df.shape)

    # Use Feast when test data has employee_id + event_timestamp
    has_entity_cols = "employee_id" in input_df.columns and "event_timestamp" in input_df.columns
    use_online = os.environ.get("FEAST_ONLINE", "0").strip().lower() in ("1", "true", "yes")

    if has_entity_cols:
        print("Using Feast for features (employee_id + event_timestamp found)")
        test_with_feast(
            entity_df=input_df,
            feast_repo_path=FEAST_REPO_PATH,
            use_online=use_online,
        )
    else:
        # Fallback: CSV must already have columns matching MLflow features_schema (snake_case)
        # If your CSV has "Years at Company" etc., add derived columns and rename to match schema
        test(input_df)