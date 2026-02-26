from kfp.dsl import component, Output, Dataset

@component(
    base_image="sandy345/kubeflow-employee-attrition:latest"
)
def ingestion_component(
    output_data: Output[Dataset]
):
    import os
    import io
    import boto3
    from src.data_pipeline._01_ingestion import ingestion

    s3 = boto3.client(
        "s3",
        endpoint_url="http://minio-service.kubeflow:9000",
        aws_access_key_id="minio",
        aws_secret_access_key="minio123",
    )

    obj = s3.get_object(Bucket="mlpipeline", Key="datasets/employee_attrition.csv")
    data_bytes = obj["Body"].read()
    print(data_bytes[:100])

    # Pass a file-like buffer to ingestion so pd.read_csv works
    df = ingestion(io.BytesIO(data_bytes))

    print(df.head(3))

    # KFP v2: output_data.path is a DIRECTORY, not a file
    os.makedirs(output_data.path, exist_ok=True)
    output_path = os.path.join(output_data.path, "ingestion.csv")

    df.to_csv(output_path, index=False)

    print(f"Ingestion completed. Saved to {output_path}")

