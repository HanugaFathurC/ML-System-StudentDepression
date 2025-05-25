import requests
import pandas as pd
import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

API_URL = os.getenv("API_URL")

def predict_from_input(df: pd.DataFrame):
    if not API_URL:
        raise ValueError("API_URL not set in .env file")

    payload = {
        "dataframe_split": {
            "columns": list(df.columns),
            "data": df.values.tolist()
        }
    }

    headers = {"Content-Type": "application/json"}
    response = requests.post(API_URL, json=payload, headers=headers)

    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Prediction failed: {response.status_code} — {response.text}")
