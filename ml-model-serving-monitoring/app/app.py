import streamlit as st
import pandas as pd
import time
import requests
import os
from dotenv import load_dotenv

# Load .env
load_dotenv()
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8001")

st.set_page_config(page_title="Student Depression Detector")
st.title("🧠 Student Depression Detector")

columns = [
    "Gender", "Age", "City", "Profession", "Academic Pressure",
    "Work Pressure", "CGPA", "Study Satisfaction", "Job Satisfaction",
    "Sleep Duration", "Dietary Habits", "Degree",
    "Have you ever had suicidal thoughts ?", "Work/Study Hours",
    "Financial Stress", "Family History of Mental Illness"
]

# Input user
user_input = {}
for col in columns:
    user_input[col] = st.number_input(col, value=0.0)

if st.button("Predict"):
    input_df = pd.DataFrame([user_input])

    # Send to FastAPI /predict_queue
    try:
        payload = {
            "input": {
                "columns": input_df.columns.tolist(),
                "data": input_df.values.tolist()
            }
        }

        response = requests.post(f"{API_BASE_URL}/predict_queue", json=payload)
        response.raise_for_status()
        job_id = response.json()["job_id"]

        st.info(f"Prediction is being processed... Job ID: {job_id}")

        # Get result from Redis via FastAPI
        with st.spinner("Waiting for prediction..."):
            for _ in range(60):  # Max 60 attempts
                result_resp = requests.get(f"{API_BASE_URL}/result/{job_id}")
                result_data = result_resp.json()
                if result_data.get("status") == "done":
                    raw_result = result_data["result"]
                    result_value = raw_result[0] if isinstance(raw_result, list) else raw_result
                    label = "Depressed" if result_value == 1 else "Not Depressed"
                    st.success(f"Prediction: {label} ({result_value})")
                    break
                time.sleep(1)
            else:
                st.error("Timeout: prediction is still pending.")

    except Exception as e:
        st.error(f"Error occurred: {str(e)}")
