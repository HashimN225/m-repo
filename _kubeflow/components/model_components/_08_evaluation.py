
from kfp.dsl import component, Input, InputPath, Dataset


@component(
    base_image="python:3.10",
    packages_to_install=['pandas', 'mlflow', 'scikit-learn', "git+https://github.com/mlops-hub/kubeflow-training-pipeline.git@main"]
)
def evaluation_component(
    test_data: Input[Dataset],
    tracking_uri: str,
    experiment_name: str,
    artifact_name: str,
    mlflow_metadata: str,
):
    import os
    from src.model_pipeline._09_evaluation import evaluate_data

    test_path = os.path.join(test_data.path, "test.csv")

    metrics = evaluate_data(
        test_path=test_path, 
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
        artifact_name=artifact_name,
        mlflow_run_id=mlflow_metadata
    )

    print(f"Evaluation is completed. Got accuracy: {metrics['recall']}")

