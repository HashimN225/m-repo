from feast import Entity, FeatureView, Field, FileSource, ValueType, FeatureService
from feast.types import Int64, Float32
from datetime import timedelta


# 1. Define the entity (the primary key)
employee = Entity(
    name="Employee ID", 
    value_type=ValueType.INT64, 
    description="Employee ID for attrition prediction"
)

# 2. Define the Data Source
# This points to where your preprocessed data is stored (Parquet is recommended for Feast)
employee_preprocessed_source = FileSource(
    path="data/preprocessed_data.parquet", # Path to offline store
    event_timestamp_column="event_timestamp",   # Feast requires a timestamp column
)

# 3. Define the Feature View
employee_features_fv = FeatureView(
    name="employee_preprocessed_features",
    entities=[employee],
    ttl=timedelta(days=365), # A long TTL as these are mostly static features
    schema=[
        Field(name="AgeGroup", dtype=Int64),
        Field(name="AnnualIncome", dtype=Int64),
        Field(name="Company Reputation", dtype=Int64),
        Field(name="Company Size", dtype=Int64),
        Field(name="Company Tenure", dtype=Float32),
        Field(name="Education Level", dtype=Int64),
        Field(name="EarlyCompanyTenureRisk", dtype=Float32),
        Field(name="Job Level", dtype=Int64),
        Field(name="LongTenureLowRoleRisk", dtype=Float32),
        Field(name="Number of Dependents", dtype=Int64),
        Field(name="Number of Promotions", dtype=Int64),
        Field(name="Opportunities", dtype=Int64), 
        Field(name="Overtime", dtype=Int64),
        Field(name="OverallSatisfaction", dtype=Int64),
        Field(name="Performance Rating", dtype=Int64),
        Field(name="Remote Work", dtype=Int64),
        Field(name="RoleStagnationRatio", dtype=Float32),
        Field(name="TenureGap", dtype=Float32),
        Field(name="Years at Company", dtype=Float32),
        Field(name="Attrition", dtype=Int64),  # The target label for training
    ],
    online=True,
    source=employee_preprocessed_source,
    tags={"team": "hr_analytics"}
)


# Define a FeatureService for your employee attrition model
employee_attrition_fs = FeatureService(
    name="employee_attrition_features",
    features=[
        employee_features_fv[[
            "AgeGroup", 
            "AnnualIncome", 
            "Company Reputation", 
            "Company Size", 
            "Company Tenure",
            "Education Level", 
            "EarlyCompanyTenureRisk", 
            "Job Level", 
            "LongTenureLowRoleRisk", 
            "Number of Dependents", 
            "Number of Promotions", 
            "Opportunities", 
            "Overtime", 
            "OverallSatisfaction", 
            "Performance Rating", 
            "Remote Work", 
            "RoleStagnationRatio", 
            "TenureGap",
            "Years at Company"]]
    ]
    # Note: 'attrition_label' is not included here as it's typically the target, not a feature for inference
)

#  testing or logging the feature values
feature_count = len(employee_features_fv.schema)
print('total features in feast: ', feature_count)

# 18 feature namesclear