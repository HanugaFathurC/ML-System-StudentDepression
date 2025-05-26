import redis
import json
import os
import time
import joblib
import requests
from dotenv import load_dotenv
import pandas as pd
import psutil # For monitoring system
from flask import Flask, Response
import psutil
import threading
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST



# === Load environment variables ===

load_dotenv()
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
MODEL_API_URL = os.getenv("API_URL")

if not MODEL_API_URL:
    raise ValueError("API_URL not set in .env file")


# === Load pre-trained scaler and label encoders ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PREPROCESS_DIR = os.path.join(BASE_DIR, "preprocessing", "output")
if not os.path.exists(PREPROCESS_DIR):
    raise FileNotFoundError(f"Preprocessing output directory {PREPROCESS_DIR} does not exist.")
scaler_path = os.path.join(PREPROCESS_DIR, "scaler.joblib")
if not os.path.exists(scaler_path):
    raise FileNotFoundError(f"Scaler file {scaler_path} does not exist.")
scaler = joblib.load(scaler_path)

label_encoders_path = os.path.join(PREPROCESS_DIR, "label_encoders.joblib")
if not os.path.exists(label_encoders_path):
    raise FileNotFoundError(f"Label encoders file {label_encoders_path} does not exist.")
label_encoders = joblib.load(label_encoders_path)

# === Init Redis ===
redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)

# === Prometheus Metrics ===
INFERENCE_REQUESTS = Counter("http_requests_total", "Total inference requests processed")
INFERENCE_LATENCY = Histogram("http_duration_seconds", "Time spent on inference")
INFERENCE_ERRORS = Counter("http_errors_total", "Total inference errors")
THROUGHPUT = Counter("http_throughput_total", "Total inference throughput")

# System metrics
CPU_USAGE = Gauge('system_cpu_usage', 'CPU Usage Percentage')
RAM_USAGE = Gauge('system_ram_usage', 'RAM Usage Percentage')

# === Flask App for Prometheus /metrics ===
app = Flask(__name__)

# === Prometheus Metrics Endpoint ===
@app.route("/metrics")
def metrics():
    """
    Endpoint to expose Prometheus metrics.
    :return:
    """

    # Update system metrics
    CPU_USAGE.set(psutil.cpu_percent(interval=1)) # Get CPU usage percentage
    RAM_USAGE.set(psutil.virtual_memory().percent) # Get RAM usage percentage

    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)

def preprocess_input(input_dict):
    """
    Preprocess the input data using the loaded scaler and label encoders.
    """

    df = pd.DataFrame([input_dict])  # convert to single-row DataFrame

    # Format specific columns
    if "Sleep Duration" in df.columns:
        df["Sleep Duration"] = df["Sleep Duration"].apply(lambda x: f"'{x}'")

    # Encode categorical features
    for col, encoder in label_encoders.items():
        if col in df.columns:
            try:
                df[col] = encoder.transform(df[col])
            except ValueError:
                raise ValueError(f"Invalid value for column '{col}': {df[col].values[0]}")

    # Scale numerical features
    numerical_cols = scaler.feature_names_in_
    df[numerical_cols] = scaler.transform(df[numerical_cols])

    return df

def process_task():
    """
    Worker function to process tasks from Redis queue.
    """
    while True:
        request_data = redis_client.rpop("inference_queue")
        if request_data:
            start_time = time.time()
            INFERENCE_REQUESTS.inc()
            THROUGHPUT.inc()
            try:
                task = json.loads(request_data)
                job_id = task["id"]
                input_data = task["input"]

                # Convert input data as dictionary
                input_dict = dict(zip(input_data["columns"], input_data["data"][0]))

                # Preprocess input data
                input_data = preprocess_input(input_dict)

                # Format payload
                payload = {
                    "dataframe_split": {
                        "columns": input_data.columns.tolist(),
                        "data": input_data.values.tolist()
                    }
                }

                # Headers
                headers = {"Content-Type": "application/json"}

                response = requests.post(MODEL_API_URL, json=payload, headers=headers)  # Send request to model
                duration = time.time() - start_time
                INFERENCE_LATENCY.observe(duration)  # Record inference duration

                # Send request to model
                if response.status_code == 200:
                    result = response.json()
                    redis_client.set(f"result:{job_id}", json.dumps(result)) # Set result in Redis
                else:
                    INFERENCE_ERRORS.inc()
                    text = response.text
                    print(f" Error job {job_id}: {response.status_code} - {text}")
            except Exception as e:
                INFERENCE_ERRORS.inc()
                print(f"There was an error processing the task: {e}")

        time.sleep(1)  # Sleep for 1 second before processing the next task for reducing CPU usage

def run_metrics_server():
    app.run(host="127.0.0.1", port=8002)

if __name__ == "__main__":
    print("Worker started and listening for inference jobs...")
    threading.Thread(target=run_metrics_server, daemon=True).start()
    process_task()
