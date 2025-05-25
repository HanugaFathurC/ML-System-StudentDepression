import streamlit as st
import pandas as pd
import time
import requests
import os
from dotenv import load_dotenv

# Load .env
load_dotenv()
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8001")

columns = [
    "Gender", "Age", "Academic Pressure",
    "Work Pressure", "CGPA", "Study Satisfaction",
    "Job Satisfaction", "Sleep Duration",
    "Dietary Habits", "Have you ever had suicidal thoughts ?",
    "Work/Study Hours", "Financial Stress",
    "Family History of Mental Illness"
]

st.set_page_config(page_title="Student Depression Detector")
st.title("🧠 Student Depression Detector")


# Input for categorical features
gender = st.selectbox("Gender", ["Male", "Female"])
sleep_duration = st.selectbox("Sleep Duration",
                              ["Less than 5 hours", "5-6 hours", "7-8 hours",
                               "More than 8 hours", "Others"])
suicidal_thoughts = st.selectbox("Have you ever had suicidal thoughts?", [
    "Yes", "No"
])
dietary_habits = st.selectbox("Dietary Habits", [
    "Healthy", "Moderate", "Unhealthy", "Others"
])
financial_stress_level = st.selectbox("Financial Stress Level 1.0 (Low) - 5.0 (High)", [
    "1.0", "2.0", "3.0", "4.0", "5.0"
])
family_history = st.selectbox("Family History of Mental Illness", [
    "Yes", "No"
])

# Input for numerical features
age = st.number_input("Age", min_value=18, max_value=100, step=1, value=20)
academic_pressure = st.slider("Academic Pressure (1-5)", 1, 5, 3)
work_pressure = st.slider("Work Pressure (1-5)", 1, 5, 3)
cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, value=3.0, step=0.1)
study_satisfaction = st.slider("Study Satisfaction (1-5)", 1, 5, 3)
job_satisfaction = st.slider("Job Satisfaction (1-5)", 1, 5, 3)
work_study_hours = st.number_input("Work/Study Hours per Day", min_value=0, max_value=16, step=1)

# Save  to DataFrame
data = pd.DataFrame([[gender, age, academic_pressure,
                      work_pressure, cgpa, study_satisfaction, job_satisfaction, sleep_duration,
                      dietary_habits, suicidal_thoughts, work_study_hours,
                      financial_stress_level, family_history
                      ]], columns=columns)

if st.button("Predict"):
    st.write("Input Data:")
    st.dataframe(data)


    # Send to FastAPI /predict_queue
    try:
        payload = {
            "input": {
                "columns": data.columns.tolist(),
                "data": data.values.tolist()
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
                if result_data.get("status") == "completed":
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
