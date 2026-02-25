from kfp import dsl
from kfp.dsl import Input, Artifact, OutputPath


@dsl.component(
    base_image="sandy345/kubeflow-employee-attrition:latest"
)
def trainer_model_component(
    # job_name: str,
    cpu: str,
    memory: str,
    train_path: Input[Artifact],
    preprocessor_model: Input[Artifact],
    tuning_metadata: Input[Artifact],
    mlflow_metadata: str,
    job_output: OutputPath(str),
    tracking_uri: str,
    experiment_name: str,
    artifact_name: str,
    minio_endpoint: str,
    minio_access_key: str,
    minio_secret_key: str,
):

    from kubeflow.trainer import TrainerClient, CustomTrainer

    def run_training(
        train_uri: str,
        preproc_uri: str,
        best_params_uri: str,
        mlflow_run_id: str,
    ):
        import os
        import subprocess

        subprocess.run([
            "python", "-m", "src.model_pipeline._08_training",
            "--train_path",        train_uri,
            "--preprocessor_path", preproc_uri,
            "--best_params_path",  best_params_uri,
            "--mlflow_run_id",     mlflow_run_id,
        ], check=True)

    # ----------------------------------------------------------------
    # Create and submit the TrainJob via SDK
    # ----------------------------------------------------------------
    client = TrainerClient()

    job_id = client.train(
        runtime="torch-distributed",
        trainer=CustomTrainer(
            image="sandy345/kubeflow-employee-attrition:latest",
            func=run_training,
            func_args={
                "train_uri":        train_path.uri,
                "preproc_uri":      preprocessor_model.uri,
                "best_params_uri":  tuning_metadata.uri,
                "mlflow_run_id":    mlflow_metadata,
                "tracking_uri":     tracking_uri,
                "experiment_name":  experiment_name,
                "artifact_name":    artifact_name,
            },
            num_nodes=1,
            resources_per_node={
                "cpu":    cpu,
                "memory": memory,
            },
            env={
                "MLFLOW_TRACKING_URI":   tracking_uri,
                "MLFLOW_EXPERIMENT_NAME": experiment_name,
                "MLFLOW_MODEL_NAME":     artifact_name,
                "MINIO_ENDPOINT":        minio_endpoint,
                "AWS_ACCESS_KEY_ID":     minio_access_key,
                "AWS_SECRET_ACCESS_KEY": minio_secret_key,
            }
        ),
    )

    print(f"Created TrainJob: {job_id}")

    for line in client.get_job_logs(job_id):
        print(line)

    print(f"TrainJob {job_id} completed!")  
    
    # client.wait_for_job_status(job_id)

    with open(job_output, "w") as f:
        f.write(job_id)