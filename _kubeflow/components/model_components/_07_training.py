from kfp import dsl
from kfp.dsl import Input, Artifact, OutputPath


@dsl.component(
    base_image="sandy345/kubeflow-employee-attrition:v2"
)
def trainer_model_component(
    job_name: str,
    namespace: str,
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
    """Launches a Kubeflow TrainJob using the Trainer SDK."""
    
    import os
    from kubeflow.trainer import TrainerClient, CustomTrainer

    # ----------------------------------------------------------------
    # Capture all args so the inner function can close over them
    # ----------------------------------------------------------------
    _train_uri       = train_path.uri
    _preproc_uri     = preprocessor_model.uri
    _best_params_uri = tuning_metadata.uri
    _mlflow_run_id   = mlflow_metadata
    _tracking_uri    = tracking_uri
    _experiment_name = experiment_name
    _artifact_name   = artifact_name

    # ----------------------------------------------------------------
    # Read MinIO secrets from env (injected into THIS pod via KFP)
    # ----------------------------------------------------------------
    _access_key = minio_access_key
    _secret_key = minio_secret_key
    _minio_endpoint = minio_endpoint

    # ----------------------------------------------------------------
    # Training function — runs inside the TrainJob pod
    # ----------------------------------------------------------------
    def run_training(
        train_uri: str,
        preproc_uri: str,
        best_params_uri: str,
        mlflow_run_id: str,
        tracking_uri: str,
        experiment_name: str,
        artifact_name: str,
        minio_endpoint,
        access_key: str,
        secret_key: str,
    ):
        import os
        import subprocess

        # Set env vars inside the training pod
        os.environ["MLFLOW_TRACKING_URI"]   = tracking_uri
        os.environ["MLFLOW_EXPERIMENT_NAME"] = experiment_name
        os.environ["MLFLOW_MODEL_NAME"]      = artifact_name
        os.environ["MINIO_ENDPOINT"]         = minio_endpoint
        os.environ["AWS_ACCESS_KEY_ID"]      = access_key
        os.environ["AWS_SECRET_ACCESS_KEY"]  = secret_key

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
    client = TrainerClient(namespace=namespace)

    job_id = client.train(
        name=job_name,
        trainer=CustomTrainer(
            func=run_training,
            func_args={
                "train_uri":        _train_uri,
                "preproc_uri":      _preproc_uri,
                "best_params_uri":  _best_params_uri,
                "mlflow_run_id":    _mlflow_run_id,
                "tracking_uri":     _tracking_uri,
                "experiment_name":  _experiment_name,
                "artifact_name":    _artifact_name,
                "minio_endpoint":   _minio_endpoint,
                "access_key":       _access_key,
                "secret_key":       _secret_key,
            },
            num_nodes=1,
            resources_per_node={
                "cpu":    cpu,
                "memory": memory,
            },
        ),
    )

    print(f"Created TrainJob: {job_id}")
    
    # Write job name to output
    with open(job_output, "w") as f:
        f.write(job_name)

    # # ----------------------------------------------------------------
    # # Wait for completion (unlike your original code which didn't wait)
    # # ----------------------------------------------------------------
    # client.wait_for_job_status(job_id)
    # print(f"TrainJob {job_id} completed!")

    # with open(job_output, "w") as f:
    #     f.write(job_id)