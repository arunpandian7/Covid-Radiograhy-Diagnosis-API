from io import BytesIO
from typing import Optional
import uuid

from pathlib import Path
import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, Header, UploadFile, Response
from fastapi.responses import StreamingResponse
from PIL import Image
from sqlmodel import Session
from backend import config
from backend.db.database import get_session
from backend.db.models import AbnormalityDetectionLog, ConditionClassificationLog
from backend.inference.covid_classification import condition_classifer
from backend.inference.covid_detection import abnormality_detector


r = router = APIRouter(prefix="/diagnose")

@r.post("/covid-abnormality")
def get_abnormalities_detection(file: UploadFile = File(...), session: Session = Depends(get_session), image_id: Optional[str] = Header(None)):
    image, image_id, file_path = save_image(file, image_id)
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    annotated_image, bbox, inference_time = abnormality_detector.inference_with_annot_image(image)
    annotated_image = Image.fromarray(annotated_image.astype("uint8")).convert("RGB")
    output_image = BytesIO()
    annotated_image.save(output_image, "PNG")
    output_image.seek(0)
   
    inf_log = AbnormalityDetectionLog(image_id=image_id, file=str(file_path), inference_time=inference_time, predicted_bbox=bbox)
    session.add(inf_log)
    session.commit()
    session.refresh(inf_log)
    
    return StreamingResponse(output_image, media_type="image/png", headers={"x-image-id":image_id})


@r.post("/covid-condition")
def get_condition_classification(response: Response, file: UploadFile = File(...), session: Session = Depends(get_session), image_id: Optional[str] = Header(None)):
    image, image_id, file_path = save_image(file, image_id)
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    condition, confidence, inf_time = condition_classifer.inference(image)
    
    inf_log = ConditionClassificationLog(image_id=image_id, file=str(file_path), inference_time=inf_time, predicted_class=condition, confidence=confidence)
    session.add(inf_log)
    session.commit()
    session.refresh(inf_log)

    response.headers["x-image-id"] = image_id
    return {"condition":condition, "confidence":confidence}


def save_image(file: UploadFile, image_id=None):
    image = Image.open(file.file)
    image_id = str(uuid.uuid4()) if not image_id else image_id
    file_path = config.IMAGE_DB_HOME / Path(image_id+".png")
    image.save(file_path)
    return image, image_id, file_path.name
