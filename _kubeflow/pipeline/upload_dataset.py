from kfp.dsl import component, pipeline
from kfp import compiler
from kfp import Client

@component(
    base_image="python:3.10",
    packages_to_install=["GitPython", "boto3"]
)
def save_dataset_to_s3():
    import os
    from git import Repo
    import boto3

    repo_url = "https://github.com/mlops-hub/kubeflow-training-pipeline.git"
    clone_dir = "/tmp/repo"
    Repo.clone_from(repo_url, clone_dir)

    src_file = os.path.join(clone_dir, "datasets", "employee_attrition.csv")
    if not os.path.exists(src_file):
        raise FileNotFoundError(f"File not found: {src_file}")

    s3 = boto3.client(
        "s3",
        endpoint_url=os.environ.get("MLFLOW_S3_ENDPOINT_URL"),
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        region_name=os.environ.get("AWS_REGION", "us-east-1"),
        config=__import__("botocore").config.Config(s3={"addressing_style": "path"})  # force path style
    )

    bucket = os.environ.get("S3_BUCKET", "mlpipeline")
    key = "datasets/employee_attrition.csv"

    try:
        s3.head_bucket(Bucket=bucket)
    except Exception:
        s3.create_bucket(Bucket=bucket)

    s3.upload_file(src_file, bucket, key)
    print(f"Uploaded to s3://{bucket}/{key}")


@pipeline(
    name="git-s3-upload-with-env-pipeline"
)
def dataset_pipeline():
    task = save_dataset_to_s3()

    # Your component env vars
    task.set_env_variable("AWS_ACCESS_KEY_ID", "minio")
    task.set_env_variable("AWS_SECRET_ACCESS_KEY", "minio123")
    task.set_env_variable("AWS_REGION", "us-east-1")
    task.set_env_variable("AWS_DEFAULT_REGION", "us-east-1")
    task.set_env_variable("MLFLOW_S3_ENDPOINT_URL", "http://minio-service.kubeflow:9000")
    task.set_env_variable("MLFLOW_S3_IGNORE_TLS", "true")
    task.set_env_variable("S3_BUCKET", "mlpipeline")

    # ✅ These env vars are picked up by the KFP launcher inside the same pod
    # and force it to use MinIO instead of AWS S3
    task.set_env_variable("AWS_S3_FORCE_PATH_STYLE", "true")
    task.set_env_variable("S3_FORCE_PATH_STYLE", "true")
    task.set_env_variable("S3_USE_HTTPS", "0")
    task.set_env_variable("S3_VERIFY_SSL", "0")
    task.set_env_variable("S3_ENDPOINT", "minio-service.kubeflow:9000")  # host:port, no http://


if __name__ == "__main__":
    pipeline_file = "pipeline_save_to_s3_with_env.yaml"

    compiler.Compiler().compile(
        pipeline_func=dataset_pipeline,
        package_path=pipeline_file
    )

    print(f"Compiled {pipeline_file}")

    KFP_HOST = "http://143.244.129.206:31621"
    client = Client(host=KFP_HOST, namespace="kubeflow")

    client.create_run_from_pipeline_package(
        pipeline_file,
        arguments={},
        experiment_name="dataset-upload"
    )