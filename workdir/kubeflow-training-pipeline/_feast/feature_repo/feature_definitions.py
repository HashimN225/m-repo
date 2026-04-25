import os
from feast import Entity, FeatureView, Field, FileSource, ValueType, FeatureService
from feast.types import Int64, Float32
from datetime import timedelta

# ✅ Read from env vars set by feast_sync.py before this file is imported
MINIO_ENDPOINT = os.environ.get("AWS_ENDPOINT_URL", "http://minio-service.kubeflow:9000")
PARQUET_PATH   = os.environ.get("FEAST_PARQUET_PATH", "data/preprocessed_data.parquet")

# 1. Entity
employee = Entity(
    name="employee_id",
    value_type=ValueType.INT64,
    description="Employee ID as primary key",
)

# 2. FileSource — NO schema here, schema belongs in FeatureView only
employee_preprocessed_source = FileSource(
    path=PARQUET_PATH,                      # ✅ from env var
    timestamp_field="event_timestamp",      # ✅ explicit — no inference
    s3_endpoint_override=MINIO_ENDPOINT,    # ✅ MinIO endpoint
)

# 3. FeatureView — schema stays here, this is correct
employee_features_fv = FeatureView(
    name="employee_features",
    entities=[employee],
    ttl=timedelta(days=365),
    schema=[
        Field(name="years_at_company",             dtype=Float32),
        Field(name="performance_rating",           dtype=Int64),
        Field(name="number_of_promotions",         dtype=Int64),
        Field(name="overtime",                     dtype=Int64),
        Field(name="education_level",              dtype=Int64),
        Field(name="number_of_dependents",         dtype=Int64),
        Field(name="job_level",                    dtype=Int64),
        Field(name="company_size",                 dtype=Int64),
        Field(name="company_tenure",               dtype=Float32),
        Field(name="remote_work",                  dtype=Int64),
        Field(name="company_reputation",           dtype=Int64),
        Field(name="attrition",                    dtype=Int64),
        Field(name="overall_satisfaction",         dtype=Int64),
        Field(name="opportunities",                dtype=Int64),
        Field(name="annual_income",                dtype=Int64),
        Field(name="age_group",                    dtype=Int64),
        Field(name="role_stagnation_ratio",        dtype=Float32),
        Field(name="tenure_gap",                   dtype=Float32),
        Field(name="early_company_tenure_risk",    dtype=Float32),
        Field(name="long_tenure_low_role_risk",    dtype=Float32),
    ],
    online=True,
    source=employee_preprocessed_source,
    tags={"team": "hr_analytics"}
)

# 4. FeatureService
employee_features_fs = FeatureService(
    name="employee_attrition_features",
    features=[
        employee_features_fv[[
            "age_group", "annual_income", "company_reputation",
            "company_size", "company_tenure", "education_level",
            "early_company_tenure_risk", "job_level", "long_tenure_low_role_risk",
            "number_of_dependents", "number_of_promotions", "opportunities",
            "overtime", "overall_satisfaction", "performance_rating",
            "remote_work", "role_stagnation_ratio", "tenure_gap",
            "years_at_company", "attrition",
        ]]
    ]
)

feature_count = len(employee_features_fv.schema)
print("total features in feast:", feature_count)