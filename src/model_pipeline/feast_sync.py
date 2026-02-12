import pandas as pd
from feast import FeatureStore, FileSource
from datetime import datetime
import os
from _feast.feature_repo.feature_definitions import employee, employee_features_fv, employee_attrition_fs, employee_preprocessed_source


def sync_to_feast(parquet_path: str, feast_repo_path: str):
    """
    parquet_path: Path to the file created in preprocessing
    repo_path: Path where your feature_store.yaml lives
    """
    store = FeatureStore(repo_path=feast_repo_path)

    # UPDATE THE SOURCE
    # You are telling the source: "Don't look at the old path, look at this 
    # specific MinIO path for this Kubeflow run."
    employee_features_fv.batch_source = FileSource(
        path=str(parquet_path),
        timestamp_field=employee_preprocessed_source.timestamp_field,
        # copy other relevant fields if they exist, like created_timestamp_column
    )

    # 1. Update the Registry (schema definitions)
    # This is equivalent to 'feast apply'
    print("Updating Feast Registry...")
    store.apply([
        employee,
        employee_features_fv,
        employee_attrition_fs,
    ])
    # Or store.apply_capsule()

    # 2. Materialize to Online Store
    # This pushes data from your parquet file into your Online DB (SQLite/Redis)
    print(f"Materializing data from {parquet_path} to Online Store (Redis)...")
    store.materialize_incremental(end_date=datetime.now())
    
    print("✓ Feast Sync Complete")



if __name__ == "__main__":
    from pathlib import Path

    BASE_DIR = Path(__file__).resolve().parents[2]  
    FEAST_DATA_DIR = BASE_DIR / "_feast" / "feature_repo"  
    os.makedirs(FEAST_DATA_DIR, exist_ok=True)
    PARQUET_PATH = FEAST_DATA_DIR / "data" / "preprocessed_data.parquet"

    sync_to_feast(
        parquet_path=PARQUET_PATH, 
        feast_repo_path=FEAST_DATA_DIR
    )