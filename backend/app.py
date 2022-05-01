from io import BytesIO
import cv2

import uvicorn
from fastapi import Depends, FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
import numpy as np
from PIL import Image

from backend import config
from backend.inference.covid_detection import abnormality_detector
from backend.inference.covid_classification import condition_classifer

app = FastAPI(
    title=config.APP_NAME, description=config.APP_DESCRIPTION 
)

@app.get("/")
def root():
    return {"message": config.APP_NAME+" is live and receiving requests...."}

@app.post("/diagnose/covid-abnormality")
def get_abnormalities_detection(file: UploadFile = File(...)):
    image = np.array(Image.open(file.file))
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    annotated_image = abnormality_detector.inference_with_annot_image(image)
    annotated_image = Image.fromarray(annotated_image.astype("uint8")).convert("RGB")
    output_image = BytesIO()
    annotated_image.save(output_image, "PNG")
    output_image.seek(0)
    return StreamingResponse(output_image, media_type="image/png")


@app.post("/diagnose/covid-condition")
def get_condition_classification(file: UploadFile = File(...)):
    image = np.array(Image.open(file.file))
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    classification_output = condition_classifer.inference(image)
    return classification_output
