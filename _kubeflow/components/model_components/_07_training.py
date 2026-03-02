from kfp import dsl
from kfp.dsl import Input, Artifact, OutputPath


@dsl.component(
    base_image="<docker-repo:tag>"
)
def trainer_model_component(
    job_name: str,
    namespace: str,
    image: str,
    train_path: Input[Artifact],
    preprocessor_model: Input[Artifact],
    best_parameters: Input[Artifact],
    mlflow_run_id: str,
    job_output: OutputPath(str),
    tracking_uri: str,
    experiment_name: str,
    artifact_name: str,
):
    from kubernetes import client, config
    
    # Load in-cluster config
    config.load_incluster_config()
    
    # Create custom objects API
    api = client.CustomObjectsApi()
    
    # TrainJob specification with MinIO credentials
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
                "image": image,
                "command": ["python", "-m", "src.model_pipeline._08_training"],
                "args": [
                    "--train_path", train_path.uri,
                    "--preprocessor_path", preprocessor_model.uri,
                    "--best_params_path", best_parameters.uri,
                    "--mlflow_run_id", mlflow_run_id,
                ],
                "numNodes": 1,
                "resourcesPerNode": {
                    "requests": {
                        "cpu": "500m",
                        "memory": "256Mi"
                    },
                    "limits": {
                        "cpu": "500m",
                        "memory": "512Mi"
                    }
                },
                "env": [
                    # MLflow settings
                    {"name": "MLFLOW_TRACKING_URI", "value": str(tracking_uri)},
                    {"name": "MLFLOW_EXPERIMENT_NAME", "value": str(experiment_name)},
                    {"name": "MLFLOW_MODEL_NAME", "value": str(artifact_name)},
                    # MinIO settings
                    {"name": "MLFLOW_S3_ENDPOINT_URL", "value": "http://minio-service.kubeflow:9000"},
                    # MinIO credentials from secret
                    {
                        "name": "AWS_ACCESS_KEY_ID",
                        "valueFrom": {
                            "secretKeyRef": {
                                "name": "mlpipeline-minio-artifact",
                                "key": "accesskey"
                            }
                        }
                    },
                    {
                        "name": "AWS_SECRET_ACCESS_KEY",
                        "valueFrom": {
                            "secretKeyRef": {
                                "name": "mlpipeline-minio-artifact",
                                "key": "secretkey"
                            }
                        }
                    }
                ]
            }
        }
    }
    
    # Create the TrainJob
    api.create_namespaced_custom_object(
        group="trainer.kubeflow.org",
        version="v1alpha1",
        namespace=namespace,
        plural="trainjobs",
        body=train_job
    )
    
    print(f"Created TrainJob: {job_name}")
    
    # Write job name to output
    with open(job_output, "w") as f:
        f.write(job_name)