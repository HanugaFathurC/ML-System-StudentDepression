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

st.set_page_config(page_title="Student Depression Detector", layout="wide")
st.title("🧠 Student Depression Detector")


# === Create input grid ===
st.markdown("### 📝 Personal & Academic Info")
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.number_input("Age", min_value=18, max_value=100, step=1, value=20)
    cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, value=3.0, step=0.1)
    sleep_duration = st.selectbox("Sleep Duration", [
        "Less than 5 hours", "5-6 hours", "7-8 hours", "More than 8 hours", "Others"
    ])
    suicidal_thoughts = st.selectbox("Ever had suicidal thoughts?", ["Yes", "No"])
    family_history = st.selectbox("Family History of Mental Illness", ["Yes", "No"])
    dietary_habits = st.selectbox("Dietary Habits", ["Healthy", "Moderate", "Unhealthy", "Others"])

with col2:
    work_study_hours = st.number_input("Work/Study Hours per Day", min_value=0, max_value=16, step=1)
    financial_stress_level = st.selectbox("Financial Stress (1 Low - 5 High)", ["1.0", "2.0", "3.0", "4.0", "5.0"])
    academic_pressure = st.slider("Academic Pressure (1 Low - 5 High)", 1, 5, 3)
    work_pressure = st.slider("Work Pressure (1 Low - 5 High)", 1, 5, 3)
    study_satisfaction = st.slider("Study Satisfaction (1 Low - 5 High)", 1, 5, 3)
    job_satisfaction = st.slider("Job Satisfaction (1 Low -5 High)", 1, 5, 3)


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

        progress_text = "Sending request to server for prediction. Please wait..."
        my_bar = st.progress(0, text=progress_text)

        for percent_complete in range(100):
            time.sleep(0.01)
            my_bar.progress(percent_complete + 1, text=progress_text)
        time.sleep(3)
        my_bar.empty()


        if response.status_code != 200:
            st.error(f"Error: {response.status_code} - {response.text}")

        else:
            job_id = response.json()["job_id"]

            # Get result from Redis via FastAPI
            result_resp = requests.get(f"{API_BASE_URL}/get_result/{job_id}")

            result_data = result_resp.json()
            with st.spinner("Processing your request..."):
                time.sleep(3)
            if result_data.get("status") == "completed":
                raw_result = result_data["result"]

                # Extract prediction
                if isinstance(raw_result, dict) and "predictions" in raw_result:
                    result_value = raw_result["predictions"][0]
                elif isinstance(raw_result, list):
                    result_value = raw_result[0]
                else:
                    result_value = raw_result

                # Determine label
                if result_value == 1:
                    label = "🧠 Depressed"
                else:
                    label = "🙂 Not Depressed"

                # Display result
                st.subheader("🧾 Prediction Result")
                st.success(f"**Prediction:** {label}")

            else:
                st.warning(f"Prediction is still pending. Job ID: {job_id}")


    except Exception as e:
        st.error(f"Error occurred: {str(e)}")
