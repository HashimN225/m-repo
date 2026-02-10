import pandas as pd
import os
from pathlib import Path
import boto3
from io import BytesIO

BASE_DIR = Path(__file__).resolve().parents[2]
DATASET_PATH = BASE_DIR / "datasets"
RAW_DATA_PATH = DATASET_PATH / "employee_attrition.csv"
INGESTION_PATH = DATASET_PATH / "data-pipeline" / "01_ingestion.csv"
os.makedirs(INGESTION_PATH.parent, exist_ok=True)

def ingestion() -> pd.DataFrame:
    s3 = boto3.client('s3')

    bucket = "ml-basics"
    key = "employee-attrition/raw_dataset.csv"

    response = s3.get_object(Bucket=bucket, Key=key)
    raw_df = BytesIO(response['Body'].read())
 
    df = pd.read_csv(raw_df)
    print(df.head(5))
    print("------")

    print(f"Shape: {df.shape}")
    print("------")

    print(f"Information: {df.info()}")
    print("------")
    return df


if __name__ == "__main__":

    df = ingestion()

    df.to_csv(INGESTION_PATH, index=False)
