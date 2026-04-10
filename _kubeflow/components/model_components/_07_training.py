from kfp import dsl
from kfp.dsl import Input, Artifact, OutputPath, Dataset
from _kubeflow.config import BASE_IMAGE


@dsl.component(
    base_image="hashimn/kubeflow_pipeline:test",
    packages_to_install=['kubernetes']
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
    """Creates a Kubeflow TrainJob for model training"""

    import uuid
    job_name = f"train-{mlflow_run_id[:5]}-{uuid.uuid4().hex[:4]}"
    namespace = "kubeflow"

    from kubernetes import client, config

    # ✅ Fix newline issue (IMPORTANT)
    tracking_uri = tracking_uri.strip()
    minio_endpoint = minio_endpoint.strip()

    # Load cluster config
    config.load_incluster_config()

    api = client.CustomObjectsApi()

    # ✅ TrainJob Spec
    train_job = {
        "apiVersion": "trainer.kubeflow.org/v1alpha1",
        "kind": "TrainJob",
        "metadata": {
            "name": job_name,
            "namespace": namespace
        },
        "spec": {
            "suspend": False,
            "runtimeRef": {
                "apiGroup": "trainer.kubeflow.org",
                "kind": "ClusterTrainingRuntime",
                "name": "torch-distributed"
            },
            "trainer": {
                "image": "hashimn/kubeflow_pipeline:v15",
                "command": ["python", "-m", "src.model_development._08_training"],
                "args": [
                    "--feast_repo_path", feast_repo_path,
                    "--train_path", train_path.uri,
                    "--preprocessor_path", preprocessor_model.uri,
                    "--best_params_path", best_parameters.uri,
                    "--mlflow_run_id", mlflow_run_id,
                ],
                "numNodes": 1,
                "resourcesPerNode": {
                    "requests": {
                        "cpu": "500m",
                        "memory": "512Mi",
                    },
                    "limits": {
                        "cpu": "1",
                        "memory": "1Gi",
                    }
                },
                "env": [
                    # ✅ MLflow
                    {"name": "MLFLOW_TRACKING_URI", "value": tracking_uri},
                    {"name": "MLFLOW_EXPERIMENT_NAME", "value": experiment_name},
                    {"name": "MLFLOW_MODEL_NAME", "value": artifact_name},

                    # ✅ MinIO
                    {"name": "MINIO_ENDPOINT", "value": minio_endpoint},
                    {"name": "S3_ENDPOINT_URL", "value": minio_endpoint},
                    {"name": "MLFLOW_S3_ENDPOINT_URL", "value": minio_endpoint},

                    {
                        "name": "AWS_ACCESS_KEY_ID",
                        "value": minio_access_key
                    },
                    {
                        "name": "AWS_SECRET_ACCESS_KEY",
                        "value": minio_secret_key
                    },

                    # ✅ Feast
                    {
                        "name": "FEAST_REGISTRY_URL",
                        "value": "postgresql+psycopg://feast:feast@postgres.feast.svc.cluster.local:5432/feast"
                    },
                    {
                        "name": "FEAST_REDIS_URL",
                        "value": "redis.feast.svc.cluster.local:6379,password=changeMeVeryStrong"
                    },
                    {
                        "name": "FEAST_S3_ENDPOINT_URL",
                        "value": minio_endpoint
                    },
                ]
            }
        }
    }

    # ✅ Create TrainJob
    api.create_namespaced_custom_object(
        group="trainer.kubeflow.org",
        version="v1alpha1",
        namespace=namespace,
        plural="trainjobs",
        body=train_job
    )

    print(f"✅ Created TrainJob: {job_name}")

    # Output
    with open(job_output, "w") as f:
        f.write(job_name)