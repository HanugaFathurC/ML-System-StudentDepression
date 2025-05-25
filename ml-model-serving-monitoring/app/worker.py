import redis
import json
import os
import time

import requests
from dotenv import load_dotenv


# Load environment
load_dotenv()
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
MODEL_API_URL = os.getenv("API_URL")

if not MODEL_API_URL:
    raise ValueError("API_URL not set in .env file")

# Init Redis
redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)

def process_task():
    """
    Worker function to process tasks from Redis queue.
    """
    while True:
        request_data = redis_client.rpop("inference_queue")
        if request_data:
            try:
                task = json.loads(request_data)
                job_id = task["id"]
                input_data = task["input"]

                # Format payload
                payload = {
                    "dataframe_split": {
                        "columns": input_data["columns"],
                        "data": input_data["data"]
                    }
                }

                # Headers
                headers = {"Content-Type": "application/json"}

                # Send request to model
                response = requests.post(MODEL_API_URL, json=payload, headers=headers)

                # Send request to model
                if response.status_code == 200:
                    result = response.json()
                    redis_client.set(f"result:{job_id}", json.dumps(result)) # Set result in Redis
                    print(f"Processed Request ID: {job_id} - Result: {result.json()}")
                else:
                    text = response.text
                    print(f" Error job {job_id}: {response.status_code} - {text}")
            except Exception as e:
                print(f"There was an error processing the task: {e}")

        time.sleep(1)  # Sleep for 1 second before processing the next task for reducing CPU usage


if __name__ == "__main__":
    # Start the worker
    print("Worker started and listening for inference jobs...")
    process_task()
