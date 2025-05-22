import os
import pandas as pd
import mlflow
from sklearn.ensemble import RandomForestClassifier


# === Set tracking URI to local MLflow ===
mlflow.set_tracking_uri("http://127.0.0.1:5000/")

# === Set experiment name ===
mlflow.set_experiment("Student Depression Modelling")

# === Load preprocessed data ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREPROCESS_DIR = os.path.join(BASE_DIR, "preprocessing", "output")

# === Load train/test splits ===
X_train = pd.read_csv(os.path.join(PREPROCESS_DIR, "X_train.csv"))
y_train = pd.read_csv(os.path.join(PREPROCESS_DIR, "y_train.csv"))
X_test = pd.read_csv(os.path.join(PREPROCESS_DIR, "X_test.csv"))
y_test = pd.read_csv(os.path.join(PREPROCESS_DIR, "y_test.csv"))

# === Save input example ===
input_example = pd.DataFrame(X_train).head(5)

# === Start MLflow run ===
with mlflow.start_run():
    # Enable autologging
    mlflow.autolog()

    # Define and train the modelling
    n_estimators = 100
    max_depth = 10
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)

    # Log the modelling explicitly
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="modelling",
        input_example=input_example
    )

    # Train the modelling
    model.fit(X_train, y_train)

    # === Evaluate the modelling ===
    accuracy = model.score(X_test, y_test)
    mlflow.log_metric("accuracy", accuracy)

    print("✅ Model trained and logged to MLflow.")




