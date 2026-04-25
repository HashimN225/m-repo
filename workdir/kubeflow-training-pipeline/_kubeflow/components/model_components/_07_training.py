from kfp import dsl
from kfp.dsl import Input, Artifact, OutputPath, Dataset
from _kubeflow.config import BASE_IMAGE


@dsl.component(
    base_image="hashimn/kubeflow_pipeline:test",
)
def trainer_model_component_v2(
    feast_repo_path: str,
    train_path: Input[Dataset],
    preprocessor_model: Input[Artifact],
    best_parameters: Input[Artifact],
    mlflow_run_id: str,
    job_output: OutputPath(str),
    tracking_uri: str,
    experiment_name: str,
    artifact_name: str,
    minio_endpoint: str,
    minio_access_key: str,
    minio_secret_key: str,
):
    """Bypasses TrainJob and runs training directly as a standard component"""

    import os
    from src.model_development._08_training import training_data

    # Set required environment variables for the training script
    os.environ['MINIO_ENDPOINT'] = minio_endpoint.strip()
    os.environ['AWS_ACCESS_KEY_ID'] = minio_access_key.strip()
    os.environ['AWS_SECRET_ACCESS_KEY'] = minio_secret_key.strip()
    os.environ['MLFLOW_TRACKING_URI'] = tracking_uri.strip()
    os.environ['MLFLOW_EXPERIMENT_NAME'] = experiment_name.strip()
    os.environ['MLFLOW_MODEL_NAME'] = artifact_name.strip()

    # Call the training function directly
    # Note: We pass the paths directly as they are already local in the KFP executor
    success = training_data(
        train_path=train_path.path, 
        preprocessor_path=preprocessor_model.path, 
        best_params_path=best_parameters.path,
        mlflow_run_id=mlflow_run_id,
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
        artifact_name=artifact_name,
        feast_repo_path=feast_repo_path
    )

    if success:
        print("✅ Training completed successfully!")
        with open(job_output, "w") as f:
            f.write("direct-training-success")
    else:
        raise Exception("❌ Training failed!")