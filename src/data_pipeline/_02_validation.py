import pandera.pandas as pa
from pandera import Column, Check
import pandas as pd

# define schema
employee_schema = pa.DataFrameSchema(
    {
        "employee_id": Column(int, Check.ge(1), unique=True, nullable=False),
        "attrition": Column(str, Check.isin(["Stayed", "Left"]), nullable=False),
        "age": Column(int, Check.ge(18), nullable=False),
        "gender": Column(str, Check.isin(["Male", "Female"]), nullable=False),
        "years_at_company": Column(int, Check.ge(0), nullable=False),
        "job_role": Column(str, nullable=False),
        "monthly_income": Column(int, Check.ge(0), nullable=False),
        "job_level": Column(str, Check.isin(["Entry", "Mid", "Senior", "Executive"]), nullable=False),
        "work_life_balance": Column(str, Check.isin(["Poor", "Fair", "Good", "Excellent"]), nullable=False),
        "job_satisfaction": Column(str, Check.isin(["Low", "Medium", "High", "Very High"]), nullable=False),
        "performance_rating": Column(str, Check.isin(["Low", "Below Average", "Average", "High"]), nullable=False),
        "number_of_promotions": Column(int, Check.ge(0), nullable=False),
        "company_size": Column(str, Check.isin(["Small", "Medium", "Large"]), nullable=False),
        "company_tenure": Column(int, Check.ge(0), nullable=False),
        "remote_work": Column(str, Check.isin(["Yes", "No"]), nullable=False),
        "leadership_opportunities": Column(str, Check.isin(["Yes", "No"]), nullable=False),
        "innovation_opportunities": Column(str, Check.isin(["Yes", "No"]), nullable=False),
        "company_reputation": Column(str, Check.isin(["Poor", "Fair", "Good", "Excellent"]), nullable=False),
        "employee_recognition": Column(str, Check.isin(["Low", "Medium", "High", "Very High"]), nullable=False),
    },
    strict=False,
    coerce=True,
)


def validate_data(df_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(df_path)

        validate_df = employee_schema(df, lazy=True)
        print("Data validation successful.")
        return validate_df
    except pa.errors.SchemaErrors as e:
        print("Data validation errors found:")
        print(e.failure_cases)
        return None
    

if __name__ == "__main__":
    from pathlib import Path

    BASE_DIR = Path(__file__).resolve().parents[2]
    DATASET_PATH = BASE_DIR / "datasets" / "data-pipeline"
    INGESTION_PATH = DATASET_PATH / "01_ingestion.csv"
    VALIDATION_PATH = DATASET_PATH / "02_validation.csv"

    
    validated_df = validate_data(INGESTION_PATH)

    validated_df.to_csv(VALIDATION_PATH, index=False)

