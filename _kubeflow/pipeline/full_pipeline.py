from kfp import dsl, kubernetes
from dotenv import load_dotenv
from pathlib import Path
import os

# Load env
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(dotenv_path=BASE_DIR.parent.parent / ".env")

# components
from _kubeflow.components.data_components._01_ingestion import ingestion_component
from _kubeflow.components.data_components._02_validation import validation_component
from _kubeflow.components.data_components._03_cleaning import cleaned_component
from _kubeflow.components.data_components._04_feature_engg import feature_engg_component
from _kubeflow.components.data_components._05_preprocessing import preprocessed_component

from _kubeflow.components.model_components.feast_sync import feast_sync_component
from _kubeflow.components.model_components._07_training import trainer_model_component_v2
from _kubeflow.components.model_components._08_evaluation import evaluation_component
from _kubeflow.components.model_components._06_tuning import tuning_component
from _kubeflow.components.model_components._09_register import register_model_component


@dsl.pipeline(
    name="Employee Attrition Full Pipeline",
    description="Data → Feast → Training → Evaluation → MLflow Registry"
)
def full_pipeline(
    tracking_uri: str = (os.getenv("MLFLOW_TRACKING_INTERNAL_URI") or "").strip(),
    experiment_name: str = (os.getenv("MLFLOW_EXPERIMENT_NAME") or "employee-attrition-v1").strip(),
    artifact_name: str = (os.getenv("MLFLOW_MODEL_NAME") or "model-name").strip(),
    registry_name: str = (os.getenv("MLFLOW_REGISTER_MODEL_NAME") or "reg-model-name").strip(),
    recall_threshold: float = 0.65,
    feast_repo_path: str = "_feast/feature_repo",

    minio_endpoint: str = (os.getenv("MINIO_ENDPOINT") or "").strip(),
    minio_access_key: str = (os.getenv("AWS_ACCESS_KEY_ID") or "").strip(),
    minio_secret_key: str = (os.getenv("AWS_SECRET_ACCESS_KEY") or "").strip(),
):
    # -------------------------
    # Data Pipeline
    # -------------------------
    ingest = ingestion_component()

    validate = validation_component(
        input_data=ingest.outputs['output_data']
    )

    cleaned = cleaned_component(
        input_data=validate.outputs['output_data']
    )

    feature_engg = feature_engg_component(
        input_data=cleaned.outputs['output_data']
    )

    preprocess = preprocessed_component(
        input_data=feature_engg.outputs['output_data']
    )

    # -------------------------
    # Feast Sync
    # -------------------------
    feast_sync = feast_sync_component(
        feast_data=preprocess.outputs['feast_data'],
        feast_repo_path=feast_repo_path,
        minio_endpoint=minio_endpoint,
        minio_access_key=minio_access_key,
        minio_secret_key=minio_secret_key,
    )

    # -------------------------
    # Tuning
    # -------------------------
    tuning = tuning_component(
        feast_repo_path=feast_repo_path,
        train_data=preprocess.outputs['train_data'],
        test_data=preprocess.outputs['test_data'],
        preprocessor_model=preprocess.outputs['preprocessor_model'],
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
        minio_endpoint=minio_endpoint,
        minio_access_key=minio_access_key,
        minio_secret_key=minio_secret_key,
    ).after(feast_sync)

    # -------------------------
    # Training (NO TrainJob now)
    # -------------------------
    train = trainer_model_component_v2(
        feast_repo_path=feast_repo_path,
        train_path=preprocess.outputs['train_data'],
        preprocessor_model=preprocess.outputs['preprocessor_model'],
        best_parameters=tuning.outputs['best_parameters'],
        mlflow_run_id=tuning.outputs["mlflow_run_id"],
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
        artifact_name=artifact_name,
        minio_endpoint=minio_endpoint,
        minio_access_key=minio_access_key,
        minio_secret_key=minio_secret_key,
    ).after(tuning)



    # -------------------------
    # Evaluation (directly after training)
    # -------------------------
    eval = evaluation_component(
        feast_repo_path=feast_repo_path,
        test_data=preprocess.outputs['test_data'],
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
        artifact_name=artifact_name,
        mlflow_run_id=tuning.outputs["mlflow_run_id"],
        minio_endpoint=minio_endpoint,
        minio_access_key=minio_access_key,
        minio_secret_key=minio_secret_key,
    ).after(train)



    # -------------------------
    # Register
    # -------------------------
    reg = register_model_component(
        registry_name=registry_name,
        recall_threshold=recall_threshold,
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
        artifact_name=artifact_name,
        mlflow_run_id=tuning.outputs["mlflow_run_id"],
        minio_endpoint=minio_endpoint,
        minio_access_key=minio_access_key,
        minio_secret_key=minio_secret_key,
    ).after(eval)
    for task in [ingest, validate, cleaned, feature_engg, preprocess, feast_sync, tuning, train, eval, reg]:
        kubernetes.set_image_pull_policy(task, "Always")