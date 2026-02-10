import json
import mlflow
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient


class MLflowRegistry:
    def __init__(self, tracking_uri: str, experiment_name: str):
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name

        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)
        self.client = MlflowClient()

    # start run
    def start_run(self, run_name: str = None, run_id: str = None):
        return mlflow.start_run(run_name=run_name, run_id=run_id)

    # get run_id
    def get_run_id(self):
        run_id = mlflow.active_run().info.run_id
        return run_id


    def log_model(
        self, 
        model, 
        X_train, 
        parameters: dict, 
        artifact_name: str,
        stage: str = "training"
    ):
        signature = infer_signature(X_train, model.predict(X_train))

        mlflow.log_params(parameters)

        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=artifact_name,
            signature=signature,
        )

        mlflow_metadata = {
            "run_id": mlflow.active_run().info.run_id,
            "model_uri": model_info.model_uri,
            "artifact_name": artifact_name,
        }
        mlflow.log_dict(mlflow_metadata, "mlflow_metadata.json")
        
        features = list(X_train.columns)
        features_dict = {"raw_features": features}
        mlflow.log_param("num_features", len(features))
        mlflow.log_dict(features_dict, "features_schema.json")

        mlflow.set_tag("artifact_name", artifact_name)
        mlflow.set_tag("stage", stage)

        # to check model is logged?
        run_id = mlflow.active_run().info.run_id
        artifacts = self.client.list_artifacts(run_id, artifact_name)
        print(f"Artifacts in '{artifact_name}':")
        for artifact in artifacts:
            print(f"  ✓ {artifact.path}")
        
        if not artifacts:
            raise ValueError(f"No artifacts found in {artifact_name}!")
        
        print("✓ Model logged to MLflow")
        return True
    

    def log_metrics_mlflow(self, metrics: dict, stage: str):
        mlflow.log_metrics(metrics)
        mlflow.set_tag("stage", stage)
        return True


    def log_params_mlflow(self, params: dict, stage: str):
        mlflow.log_params(params)
        mlflow.set_tag("stage", stage)
        return True
    

    def load_model(self, run_id: str, artifact_name: str):
        model_uri = f"runs:/{run_id}/{artifact_name}"
        print(f"Loading model from: {model_uri}")
        return mlflow.sklearn.load_model(model_uri)


    def register_model(self, run_id: str, artifact_name: str, registry_name: str):
        # model_uri = f"runs:/{metadata['run_id']}/{metadata['artifact_name']}"
        model_uri = f"runs:/{run_id}/{artifact_name}"
        print(f'Model URI: {model_uri}')

        register_model = mlflow.register_model(
            model_uri=model_uri,
            name=registry_name
        )
        version = register_model.version

        # set tags
        self.client.set_model_version_tag(
            name=register_model.name,
            version=version,
            key="source_run_id",
            value=run_id
        )

        self.client.set_model_version_tag(
            name=register_model.name,
            version=version,
            key="production_ready",
            value="pending_approval",
        )

        # Transition to stage
        self.client.transition_model_version_stage(
            name=register_model.name,
            version=version,
            stage='Staging',
        )

        return register_model


    def promote_model(self, model_name: str, version: int, metric_value: float, threshold: float):
        stage = "Production" if metric_value >= threshold else "Staging"
        self.client.transition_model_version_stage( 
            name=model_name, 
            version=version, 
            stage=stage, 
        ) 
        self.client.set_model_version_tag( 
            name=model_name, 
            version=version, 
            key="promotion_decision", 
            value=f"{stage} (metric={metric_value:.3f}, threshold={threshold})", 
        ) 
        return stage


    def get_metric_from_mlfow(self, run_id: str):
        run = self.client.get_run(run_id)
        metrics = run.data.metrics
        return metrics


    # can remove
    def load_features_from_mlflow(self):
        runs = mlflow.search_runs(
            filter_string="tags.artifact_name = 'employee-attrition-model'",
            order_by=["start_time DESC"],
            max_results=1,
        )
        run_id = runs.iloc[0].run_id

        local_path = mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path="features_schema.json"
        )

        with open(local_path) as f:
            features = json.load(f)
        
        return features['raw_features']


    def load_features(self, run_id: str):
        features_path = self.client.download_artifacts(
            run_id=run_id,
            path="feature_schema.json",
        )

        with open(features_path, 'r') as f:
            features_list = json.load(f)
        
        return features_list['raw_features']
    

    def load_registered_model(self, model_name: str, version: int):
        model_uri = f"models://{model_name}/{version}"
        return mlflow.sklearn.load_model(model_uri)
