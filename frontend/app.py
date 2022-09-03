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

def send_feedback(models, feedback):
    print("sent")
    return

def show_fb_block():
    return gr.update(visible=True)

def show_condition_db(model):
    if model == "Condition":
        return gr.update(visible=True)

with gr.Blocks() as diagnosis_app:
    with gr.Column():
        input_image = gr.Image(label="Upload Input Chest X-Ray Image")
        diagnose_btn = gr.Button("Diagnose")
    with gr.Row():
        output_detection = gr.Image(label="Covid Abnormality Detection")
        output_condition = gr.Textbox(label="COVID Condition Severity")
    diagnose_btn.click(fetch_diagnosis, inputs=[input_image], outputs=[output_detection, output_condition])
    flag_models = gr.CheckboxGroup(["Abnormality", "Condition"], visible=False)
    output_condition.change(show_fb_block, None, flag_models)
    flag_condition = gr.Dropdown(
        ["typical", "atypical", "indeterminate", "absense"], 
        visible=False
    )
    flag_models.change(show_condition_db, flag_models, flag_condition)
    send_fb_btn = gr.Button("Send Feedback", visible=False)
    flag_models.change(show_fb_block, None, send_fb_btn)
    send_fb_btn.click(send_feedback, [flag_models, flag_condition], None)
    
diagnosis_app.launch()