import pandas as pd
import os


def ingestion(dataset_path) -> pd.DataFrame:
    
    df = pd.read_csv(dataset_path)
    print(df.head(5))
    print("------")

    print(f"Shape: {df.shape}")
    print("------")

    print(f"Information: {df.info()}")
    print("------")
    return df


if __name__ == "__main__":
    from pathlib import Path

    BASE_DIR = Path(__file__).resolve().parents[2]
    DATASET_PATH = BASE_DIR / "datasets"
    RAW_DATA_PATH = DATASET_PATH / "employee_attrition.csv"
    INGESTION_PATH = DATASET_PATH / "data-pipeline" / "01_ingestion.csv"
    os.makedirs(INGESTION_PATH.parent, exist_ok=True)


    df = ingestion(RAW_DATA_PATH)

    df.to_csv(INGESTION_PATH, index=False)
