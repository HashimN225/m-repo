import pandas as pd
from feast import FeatureStore, FileSource
from datetime import datetime
import os
from _feast.feature_repo.feature_definitions import employee, employee_features_fv, employee_features_fs, employee_preprocessed_source


def sync_to_feast(parquet_path: str, feast_repo_path: str):
    store = FeatureStore(repo_path=feast_repo_path)

    # UPDATE THE SOURCE
    # You are telling the source: "Don't look at the old path, look at this 
    # specific MinIO path for this Kubeflow run."
    employee_features_fv.batch_source = FileSource(
        path=str(parquet_path),
        timestamp_field=employee_preprocessed_source.timestamp_field,
    )
    print('emp-source: ', employee_features_fv)

    # 1. Update the Registry (schema definitions)
    # This is equivalent to 'feast apply'
    print("Updating Feast Registry...")
    store.apply([
        employee,
        employee_features_fv,
        employee_features_fs
    ])
    print("\n Registered Entities:")
    for entity in store.list_entities():
        print(f"   - {entity.name}")

    # 2. Materialize to Online Store
    # This pushes data from your parquet file into your Online DB (SQLite/Redis)
    print(f"Materializing data from Offline store (minIO) to Online store (Redis)...")
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