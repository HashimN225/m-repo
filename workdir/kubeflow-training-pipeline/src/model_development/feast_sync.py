import os
from datetime import datetime


def sync_to_feast(parquet_path: str, feast_repo_path: str,
                  minio_endpoint: str = None,
                  minio_access_key: str = None,
                  minio_secret_key: str = None):

    _endpoint = minio_endpoint or "http://minio-service.kubeflow:9000"

    # ✅ boto3 / pyarrow env vars
    os.environ["AWS_ACCESS_KEY_ID"]       = minio_access_key or ""
    os.environ["AWS_SECRET_ACCESS_KEY"]   = minio_secret_key or ""
    os.environ["AWS_ENDPOINT_URL"]        = _endpoint
    os.environ["AWS_S3_ENDPOINT_URL"]     = _endpoint
    os.environ["AWS_S3_ENDPOINT"]         = _endpoint
    os.environ["FEAST_S3_ENDPOINT_URL"]   = _endpoint
    os.environ["AWS_S3_FORCE_PATH_STYLE"] = "true"
    os.environ["AWS_DEFAULT_REGION"]      = "us-east-1"
    os.environ["AWS_VIRTUAL_HOSTING"]     = "false"
    os.environ["FEAST_PARQUET_PATH"]      = str(parquet_path)

    # ✅ s3fs / aiobotocore env vars (used by materialize_incremental)
    os.environ["FSSPEC_S3_ENDPOINT_URL"]  = _endpoint
    os.environ["FSSPEC_S3_KEY"]           = minio_access_key or ""
    os.environ["FSSPEC_S3_SECRET"]        = minio_secret_key or ""
    os.environ["FSSPEC_S3_CLIENT_KWARGS"] = f'{{"endpoint_url": "{_endpoint}"}}'

    print(f'Parquet Path:   {parquet_path}')
    print(f'MinIO Endpoint: {_endpoint}')

    # ✅ Deferred imports AFTER env vars
    from feast import FeatureStore
    from _feast.feature_repo.feature_definitions import (
        employee, employee_features_fv, employee_features_fs
    )

    store = FeatureStore(repo_path=feast_repo_path)

    print("Updating Feast Registry...")
    store.apply([employee, employee_features_fv, employee_features_fs])

    print("\nRegistered Entities:")
    for entity in store.list_entities():
        print(f"   - {entity.name}")

    print("Materializing data from Offline store (MinIO) to Online store (Redis)...")
    store.materialize_incremental(end_date=datetime.now())

    print("✓ Feast Sync Complete")


if __name__ == "__main__":
    from pathlib import Path

    BASE_DIR       = Path(__file__).resolve().parents[2]
    FEAST_DATA_DIR = BASE_DIR / "_feast" / "feature_repo"
    os.makedirs(FEAST_DATA_DIR, exist_ok=True)
    PARQUET_PATH   = FEAST_DATA_DIR / "data" / "preprocessed_data.parquet"

    sync_to_feast(
        parquet_path=str(PARQUET_PATH),
        feast_repo_path=str(FEAST_DATA_DIR)
    )