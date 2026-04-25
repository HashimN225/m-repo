import os
import pandas as pd
from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from dotenv import load_dotenv

load_dotenv()

def preprocess_data(df_path: str, parquet_output_path: str):
    print(f"Loading data from {df_path}...")
    full_df = pd.read_csv(df_path)

    # Convert attrition to numeric if it's string
    if full_df['attrition'].dtype == object:
        full_df['attrition'] = full_df['attrition'].map({'Stayed': 0, 'Left': 1})

    if 'event_timestamp' not in full_df.columns:
        full_df['event_timestamp'] = pd.Timestamp.now()

    # --- FEAST LOGIC ---
    # We only store CURRENT employees (attrition == 0) in the online store.
    # This ensures "Left" employees show as "Not Found" in the UI.
    feast_df = full_df[full_df['attrition'] == 0]
    
    print(f"Saving {len(feast_df)} active employees to Feast parquet...")
    os.makedirs(os.path.dirname(parquet_output_path), exist_ok=True)
    feast_df.to_parquet(parquet_output_path, index=False)
    # --- END FEAST LOGIC ---
     
    # Separate features and target
    X = full_df.drop(columns=["attrition"])
    y = full_df["attrition"]

    NUMERIC_COLS = [
        "years_at_company",
        "company_tenure",
        "annual_income",
        "number_of_promotions",
        "number_of_dependents",
    ]

    BINARY_COLS = ["overtime", "remote_work"]

    CATEGORICAL_COLS = [
        "education_level",
        "job_level",
        "company_size",
        "performance_rating",
        "age_group",
        "overall_satisfaction",
        "opportunities",
        "company_reputation",
    ]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_COLS),
            ("cat", OneHotEncoder(handle_unknown='ignore'), CATEGORICAL_COLS),
            ("bin", "passthrough", BINARY_COLS),
        ],
        remainder="passthrough"
    )

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    print("Preprocessing completed.")
    return train_df, test_df, preprocessor


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parents[2]    
    DATASET_PATH = BASE_DIR / "datasets" / "data-pipeline"
    FEATURED_PATH = DATASET_PATH / "05_feature_engg_df.csv"
    
    OUTPUT_DIR = DATASET_PATH / "data-pipeline"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    TRAIN_PATH = OUTPUT_DIR / "06_preprocess_train_df.csv"
    TEST_PATH = OUTPUT_DIR / "06_preprocess_test_df.csv"

    ARTIFACTS_PATH = BASE_DIR / "artifacts"
    os.makedirs(ARTIFACTS_PATH, exist_ok=True)
    PREPROCESSOR_PATH = ARTIFACTS_PATH / "preprocessor.pkl"

    FEAST_DATA_DIR = BASE_DIR / "_feast" / "feature_repo" / "data" 
    PARQUET_PATH = FEAST_DATA_DIR / "preprocessed_data.parquet"

    train_df, test_df, preprocessor = preprocess_data(
        df_path=str(FEATURED_PATH),
        parquet_output_path=str(PARQUET_PATH),
    )

    # Save preprocessed data and preprocessor
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    train_df.to_csv(TRAIN_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)
    print(f"Data saved to {OUTPUT_DIR}")