from io import BytesIO

import gradio as gr
import requests
from PIL import Image


def fetch_diagnosis(input_image):
    image = Image.fromarray(input_image)
    byte_io = BytesIO()
    image.save(byte_io, format="PNG")
    files = {"file": byte_io.getvalue()}
    det_response = requests.post(f"http://localhost:8080/diagnose/covid-abnormality/", files=files, stream=True)
    stream = BytesIO(det_response.content)
    detection = Image.open(stream)
    condition_response = requests.post(f"http://localhost:8080/diagnose/covid-condition", files=files, stream=True)
    condition = condition_response.json()["condition"]
    return detection, condition

with gr.Blocks() as diagnosis_app:
    with gr.Column():
        input_image = gr.Image(label="Upload Input Chest X-Ray Image")
        diagnose_btn = gr.Button("Diagnose")
    with gr.Row():
        output_detection = gr.Image(label="Covid Abnormality Detection")
        output_condition = gr.Textbox(label="COVID Condition Severity")

    diagnose_btn.click(fetch_diagnosis, inputs=[input_image], outputs=[output_detection, output_condition])

diagnosis_app.launch()
