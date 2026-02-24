
from kfp.dsl import component, Input, InputPath, Dataset


@component(
    base_image="sandy345/kubeflow-employee-attrition:v1"
)
def evaluation_component(
    test_data: Input[Dataset],
    tracking_uri: str,
    experiment_name: str,
    artifact_name: str,
    mlflow_metadata: str,
    minio_endpoint: str,
    minio_access_key: str,
    minio_secret_key: str,
):
    import os
    from src.model_pipeline._09_evaluation import evaluate_data

    test_path = os.path.join(test_data.path, "test.csv")

    # Must be set BEFORE any feast/pyarrow/boto3 imports
    os.environ["AWS_ACCESS_KEY_ID"] = minio_access_key
    os.environ["AWS_SECRET_ACCESS_KEY"] = minio_secret_key
    os.environ["AWS_DEFAULT_REGION"] = "us-east-1"          # dummy, but required
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = minio_endpoint

    metrics = evaluate_data(
        test_path=test_path, 
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
        artifact_name=artifact_name,
        mlflow_run_id=mlflow_metadata
    )

    print(f"Evaluation is completed. Got accuracy: {metrics['recall']}")

