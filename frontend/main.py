import os

import requests
from pathlib import Path
import streamlit as st


if __name__ == "__main__":
    st.title("COVID-19 Abnormality Detection and Diagnosis API")
    image_file = st.file_uploader("Upload an Image", type=["png"])
    global show_report
    show_report = False

    if image_file is not None:
        st.header("Uploaded X-Ray:")
        st.image(image_file, width=600)

    diagnose_button = st.button("Get Diagnosis")
    if st.session_state.get("diagnose_button") != True:
        st.session_state["diagnose_button"] = diagnose_button
    
    if st.session_state["diagnose_button"] == True:
        with st.container():
            files = {"file": image_file.getvalue()}
            detection_response = requests.post(f"http://localhost:8080/diagnose/covid-abnormality/", files=files, stream=True)
            global image_id
            image_id = detection_response.headers["x-image-id"]
            st.header("Covid Abnormality Detection")
            st.image(detection_response.content, width=600)

            headers = {"image-id": image_id}
            condition_response = requests.post(f"http://localhost:8080/diagnose/covid-condition", files=files, headers=headers, stream=True)
            st.write(f"""Condition : {condition_response.json()["condition"]} """)

            report_button = st.button("Report Diagnosis", "report")
            if st.session_state.get("report_button") != True:
                st.session_state["report_button"] = report_button

            if st.session_state["report_button"] == True:
                option = st.multiselect(
                    "Which prediction you would like to report",
                    ["Abnormality Detection", "Condition Diagnosis"]
                )

                if "Condition Diagnosis" in option:
                    condition_feedback = st.selectbox(
                        "Do you like to give your feedback to our model on the correct condition diagnosis?",
                        ("No Feedback","Typical Appearance", "Atypical Appearance", "Indeterminate Appearance", "Pneumonia Negative"),
                    )
                    params = {"image_id":image_id, "update_condition":condition_feedback}

                    report_response = requests.patch(f"http://localhost:8080/report/covid-condition", params=params)
                    st.session_state["diagnose_button"] = False
                    st.session_state["report_button"] = False
                    st.checkbox("Reload")
            

            
