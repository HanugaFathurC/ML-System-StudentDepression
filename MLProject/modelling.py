import os
import pandas as pd
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
from dotenv import load_dotenv

import warnings


def setup_mlflow():
    # Load .env file only in local development
    if os.getenv("ENV") != "production":
        load_dotenv()

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "https://dagshub.com/wibugarxurang/student-depression.mlflow")
    username = os.getenv("MLFLOW_TRACKING_USERNAME")
    password = os.getenv("MLFLOW_TRACKING_PASSWORD")

    # Use local fallback if not production and DagsHub is unreachable
    if os.getenv("ENV") != "production" and not username:
        print("No username detected. Falling back to local MLflow tracking URI.")

    # Final safety check
    if not tracking_uri:
        raise ValueError("MLFLOW_TRACKING_URI is not set!")


    # Set MLflow URI and experiment
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("Student Depression Modelling")

    # Set credentials (for DagsHub or remote servers)
    os.environ["MLFLOW_TRACKING_USERNAME"] = username or ""
    os.environ["MLFLOW_TRACKING_PASSWORD"] = password or ""
    print(f"MLflow tracking URI set to: {tracking_uri}")


def main():
    # Suppress warnings
    warnings.filterwarnings("ignore")

    # Suppress urllib3
    warnings.filterwarnings("ignore", category=UserWarning, module="urllib3")

    # Set up ml flow
    setup_mlflow()

    # === Load preprocessed data ===
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    PREPROCESS_DIR = os.path.join(BASE_DIR, "preprocessing", "output")

    X_train = pd.read_csv(os.path.join(PREPROCESS_DIR, "X_train.csv"))
    y_train = pd.read_csv(os.path.join(PREPROCESS_DIR, "y_train.csv"))
    X_test = pd.read_csv(os.path.join(PREPROCESS_DIR, "X_test.csv"))
    y_test = pd.read_csv(os.path.join(PREPROCESS_DIR, "y_test.csv"))

    input_example = pd.DataFrame(X_train).head(5)

    # === Parameters from modelling tuning ===
    n_estimators = 505
    max_depth = 25

    # === Start MLflow run ===
    if mlflow.active_run() is None:
        mlflow.start_run(run_name="Student Depression Modelling")
    
    # Log hyperparameters manually
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)

    # Train model
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)

    # Evaluate and log metrics
    accuracy = model.score(X_test, y_test)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    # Log metrics manually
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision_weighted", precision)
    mlflow.log_metric("recall_weighted", recall)
    mlflow.log_metric("f1_weighted", f1)

    # Log model manually
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="modelling",
        input_example=input_example
    )

    print(f"Model trained and logged to MLflow with accuracy: {accuracy:.4f}")
    print(f"accuracy={accuracy:.4f}, precision={precision:.4f}, recall={recall:.4f}, f1={f1:.4f}")


if __name__ == "__main__":
    main()