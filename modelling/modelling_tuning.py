import os
import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# === Setup MLflow local or remote ===
mlflow.set_tracking_uri("http://127.0.0.1:5000/")
mlflow.set_experiment("Student Depression Modelling")

# === Load data ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREPROCESS_DIR = os.path.join(BASE_DIR, "preprocessing", "output")

X_train = pd.read_csv(os.path.join(PREPROCESS_DIR, "X_train.csv"))
y_train = pd.read_csv(os.path.join(PREPROCESS_DIR, "y_train.csv"))
X_test = pd.read_csv(os.path.join(PREPROCESS_DIR, "X_test.csv"))
y_test = pd.read_csv(os.path.join(PREPROCESS_DIR, "y_test.csv"))

input_example = X_train.head(5)

# === Hyperparameter Tuning ===
# Define the parameter grid
n_estimators_range = np.linspace(10, 1000, 5, dtype=int)  # 5 evenly spaced values
max_depth_range = np.linspace(1, 50, 5, dtype=int)  # 5 evenly spaced values

best_accuracy = 0
best_params = {}

# === Grid Search Loop with Manual Logging ===
for n_estimators in n_estimators_range:
    for max_depth in max_depth_range:
        with mlflow.start_run():
            # Log params
            mlflow.log_param("n_estimators", n_estimators)
            mlflow.log_param("max_depth", max_depth)

            # Train modelling
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Calculate metrics
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average="weighted")
            rec = recall_score(y_test, y_pred, average="weighted")
            f1 = f1_score(y_test, y_pred, average="weighted")

            # Log metrics
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("precision", prec)
            mlflow.log_metric("recall", rec)
            mlflow.log_metric("f1_score", f1)

            # Log modelling
            mlflow.sklearn.log_model(model, artifact_path="modelling", input_example=input_example)

            # Log data
            mlflow.log_artifact(os.path.join(PREPROCESS_DIR, "student_depression_processed.csv"),
                                artifact_path="data")

            # Track best modelling
            if acc > best_accuracy:
                best_accuracy = acc
                best_params = {
                    "n_estimators": n_estimators,
                    "max_depth": max_depth
                }

            print(f"Logged modelling with n_estimators={n_estimators}, max_depth={max_depth}, accuracy={acc:.4f}")

print("Best accuracy:", best_accuracy)
print("🏆 Best parameters:",
      {k: int(v) for k, v in best_params.items()})
