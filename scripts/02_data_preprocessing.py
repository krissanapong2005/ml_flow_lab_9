import os
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import mlflow


def preprocess_data(test_size=0.25, random_state=42):
    """
    Loads raw data, splits into train/test, and logs processed data to MLflow.
    """
    mlflow.set_experiment("iris Quality - Data Preprocessing")

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"Starting data preprocessing run with run_id: {run_id}")
        mlflow.set_tag("ml.step", "data_preprocessing")

        # 1. Load data
        iris_data = load_iris(as_frame=True)
        df = iris_data.frame

        # 2. Split data
        X = df.drop("target", axis=1)
        y = df["target"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # 3. Save processed data
        processed_data_dir = "processed_data"
        os.makedirs(processed_data_dir, exist_ok=True)

        pd.concat([X_train, y_train], axis=1).to_csv(
            os.path.join(processed_data_dir, "train.csv"), index=False
        )
        pd.concat([X_test, y_test], axis=1).to_csv(
            os.path.join(processed_data_dir, "test.csv"), index=False
        )
        print(f"Saved processed data to '{processed_data_dir}' directory.")

        # 4. Log params & metrics
        mlflow.log_param("test_size", test_size)
        mlflow.log_metric("training_set_rows", len(X_train))
        mlflow.log_metric("test_set_rows", len(X_test))

        # 5. Log artifacts
        mlflow.log_artifacts(processed_data_dir, artifact_path="processed_data")
        print("Logged processed data as artifacts in MLflow.")
        print("-" * 50)
        print(f"Preprocessing Run ID: {run_id}")
        print("-" * 50)

        # Write run_id to GITHUB_OUTPUT for other steps in the workflow to use
        if "GITHUB_OUTPUT" in os.environ:
            with open(os.environ["GITHUB_OUTPUT"], "a") as f:
                print(f"run_id={run_id}", file=f)


if __name__ == "__main__":
    # This line correctly calls the function to ensure the script runs
    preprocess_data()

