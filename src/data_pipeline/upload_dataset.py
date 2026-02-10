import pandas as pd
import boto3
import os

def upload_df() -> pd.DataFrame:
    s3 = boto3.client(
        's3',
        endpoint_url=os.environ["MINIO_ENDPOINT"],
        aws_access_key_id=os.environ["MINIO_AWS_ACCESS_KEY"],
        aws_secret_access_key=os.environ["MINIO_AWS_SECRET_KEY"]
    )

    bucket = "mlpipeline"
    key = "datasets/employee_attrition.csv"

    s3.upload_file(
        Filename="/tmp/employee_attirition.csv",
        Bucket=bucket,
        Key=key
    )