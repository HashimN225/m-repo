
from kfp.dsl import component, Input, Output, Dataset

@component(
    base_image="python:3.10",
    packages_to_install=['pandas', 'pandera', "git+https://github.com/mlops-hub/kubeflow-training-pipeline.git@main"]
)
def validation_component(
    input_data: Input[Dataset], 
    output_data: Output[Dataset]
):
    import os
    # import boto3
    from data_pipeline._02_validation import validate_data

    input_path = os.path.join(input_data.path, "ingestion.csv")

    validated_df = validate_data(input_path)

    os.makedirs(output_data.path, exist_ok=True)
    output_path = os.path.join(output_data.path, "validation.csv")
    validated_df.to_csv(output_path, index=False)

    # save in s3
    # s3 = boto3.client('s3')
    # bucket = "ml-basics"
    # key = "employee-attrition/validation"

    # s3.upload_file(output_path, bucket, f"{key}/validation.csv")

    print(f"Validation completed. Saved to {output_path}")

