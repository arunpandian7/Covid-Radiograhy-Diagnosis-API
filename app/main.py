import os
from pathlib import Path
import streamlit as st

from api.models import CovidDetectron, CovidDiagnoser
from app.utils import clean_dir

temp_dir = Path(".temp")
temp_dir.mkdir(exist_ok=True)


if __name__ == "__main__":
    st.title("AI Based Medical Diagnosis System - COVID Abormality Detection")
    file = st.file_uploader("Upload an Image", type=["png"])
    if file is not None:
        clean_dir(temp_dir)
        file_path = temp_dir / file.name
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
        st.header("Uploaded X-Ray:")
        st.image(file, width=600)
    if st.button("Get Diagnosis"):
        img = CovidDetectron.inference(str(file_path))
        st.header("Lung Infection Detection")
        st.image(img, width=600)
        result = CovidDiagnoser.inference(str(file_path))
        st.write(f"""
            ## Condition : {result[0]} \n 
            ## Confidence : {(result[1]*100).round(2)}%
        """)