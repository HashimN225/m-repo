from kfp.dsl import component, InputPath

@component(
    base_image="<docker-repo:tag>"
)
def register_model_component(
    registry_name: str, 
    recall_threshold: float,
    tracking_uri: str,
    experiment_name: str,
    artifact_name: str,
    mlflow_run_id: str,
    minio_endpoint: str,
    minio_access_key: str,
    minio_secret_key: str,
):
    import os
    from src.model_pipeline._10_registry import register_model_to_mlflow, promote_to_production

    # Must be set BEFORE any feast/pyarrow/boto3 imports
    os.environ["AWS_ACCESS_KEY_ID"] = minio_access_key
    os.environ["AWS_SECRET_ACCESS_KEY"] = minio_secret_key
    os.environ["AWS_DEFAULT_REGION"] = "us-east-1"          # dummy, but required
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = minio_endpoint

    registered_model, metrics = register_model_to_mlflow(    
        registry_name=registry_name, 
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
        artifact_name=artifact_name,
        mlflow_run_id=mlflow_run_id
    )

    promote_to_production(
        metric=metrics['recall'],
        model_name=registered_model.name,
        version=registered_model.version,
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
        recall_threshold=recall_threshold,
    )

    print("Model registration completed.")