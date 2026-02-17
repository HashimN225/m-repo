from kfp import dsl
from kfp.dsl import component, Input, Dataset

@component(
    base_image="sandy345/kubeflow-employee-attrition:latest"
)
def feast_sync_component(
    feast_data: Input[Dataset],
    feast_repo_path: str = "_feast/feature_repo"
):
    import os
    from src.model_development.feast_sync import sync_to_feast
    
    # feast_data.path is the directory
    # We need the actual parquet file inside it
    parquet_file_path = os.path.join(feast_data.path, "preprocessed_data.parquet")
    
    # If feast_data is stored in S3/MinIO, feast_data.path might be /tmp/... 
    # but Feast needs the S3 URL (s3://...) to read it from other pods.
    # Check if we should use .uri instead of .path
    print('data-parquet-path: ', parquet_file_path)
    actual_path = feast_data.uri if feast_data.uri.startswith("s3://") else parquet_file_path
    print(f"Actual Feast Path: ", actual_path)


    sync_to_feast(
        parquet_path=actual_path,
        feast_repo_path=feast_repo_path
    )