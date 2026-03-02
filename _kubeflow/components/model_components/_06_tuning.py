from kfp import dsl
from kfp.dsl import Input, Output, OutputPath, Model, Dataset

@dsl.component(
    base_image="<docker-repo:tag>"
)
def tuning_component(
    train_data: Input[Dataset],
    test_data: Input[Dataset],
    preprocessor_model: Input[Model],
    best_parameters: Output[Dataset],
    tracking_uri: str,
    experiment_name: str,
    mlflow_run_id: OutputPath(str),
):
    import os
    import json
    from src.model_pipeline._07_tuning import tuning_data

    train_path = os.path.join(train_data.path, "train.csv")
    test_path = os.path.join(test_data.path, "test.csv")
    preprocessor_output = os.path.join(preprocessor_model.path, "preprocessor.pkl")

    run_id, best_parameters_output = tuning_data(
        train_path=train_path, 
        test_path=test_path, 
        preprocess_path=preprocessor_output,
        tracking_uri=tracking_uri,
        experiment_name=experiment_name
    )

    os.makedirs(best_parameters.path, exist_ok=True)
    tuning_file = os.path.join(best_parameters.path, "best_parameters.json")
    with open(tuning_file, 'w') as f:
        json.dump(best_parameters_output, f)


    with open(mlflow_run_id, "w") as f:
        f.write(run_id)
            

    print("Tuning completed successfully.")

