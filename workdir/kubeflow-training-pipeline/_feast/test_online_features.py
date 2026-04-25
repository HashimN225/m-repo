from feast import FeatureStore
import pandas as pd
import redis


store = FeatureStore(repo_path="./feature_repo", fs_yaml_file="./feature_repo/feature_store.local.yaml")

try:
    print("Attempting to ping Redis...")
    r = redis.Redis(
        host="68.183.87.245",
        port=30379,
        username="default",
        password="changeMeVeryStrong",
        db=0
    )
    r.ping()
    print("✅ Online Store Connection Successful!")
except Exception as e:
    print(f"❌ Connection Failed: {e}")


# Online Fetch
feature_service = store.get_feature_service("employee_attrition_features")

features = store.get_online_features(
    features=feature_service,
    entity_rows=[{"Employee ID": 9063}]
).to_dict()

print("Online features: ", features)


# Offline Fetch
entity_df = pd.DataFrame({
    "Employee ID": [30257, 9063],
    "event_timestamp": [pd.Timestamp.now(), pd.Timestamp.now()]
})
print('ent-df: ', entity_df)

training_df = store.get_historical_features(
    entity_df=entity_df,
    features=store.get_feature_service("employee_attrition_features")
).to_df()

print("Training: ", training_df)


# Output:

# Training:     Employee ID                  event_timestamp  AgeGroup  ...  RoleStagnationRatio  TenureGap  Years at Company
# 0        30257 2026-02-12 03:01:38.392035+00:00         1  ...                0.116       5.34              0.83
# 1         9063 2026-02-12 03:01:38.392035+00:00         2  ...                0.120       2.67              0.50

# [2 rows x 21 columns]
# Online features:  {'Employee ID': [9063], 'Company Tenure': [3.1700000762939453], 'RoleStagnationRatio': [0.11999999731779099], 'Number of Dependents': [0], 'Performance Rating': [1], 'EarlyCompanyTenureRisk': [1.0], 'Number of Promotions': [2], 'Opportunities': [0], 'AgeGroup': [2], 'Education Level': [3], 'Remote Work': [0], 'TenureGap': [2.6700000762939453], 'Company Reputation': [1], 'OverallSatisfaction': [2], 'Job Level': [2], 'Years at Company': [0.5], 'Company Size': [2], 'LongTenureLowRoleRisk': [0.0], 'AnnualIncome': [0], 'Overtime': [0]}