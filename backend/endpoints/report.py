from backend.db.database import get_session
from backend.db.models import AbnormalityDetectionLog, ConditionClassificationLog
from fastapi import APIRouter, Depends
from sqlmodel import select, Session

r = router = APIRouter(prefix="/report")


@r.patch("/covid-abnormality")
def report_abnormalities_detection(image_id:str, update_bbox:dict = None, session: Session = Depends(get_session)):
    statement = select(AbnormalityDetectionLog).where(AbnormalityDetectionLog.image_id == image_id)
    results = session.exec(statement)
    log = results.one()
    log.misprediction = True
    if not update_bbox:
        log.feedback_bbox = update_bbox
    session.add(log)
    session.commit()
    session.refresh(log)
    return

@r.patch("/covid-condition")
def report_abnormalities_detection(image_id:str, update_condition:dict = None, session: Session = Depends(get_session)):
    statement = select(ConditionClassificationLog).where(ConditionClassificationLog.image_id == image_id)
    results = session.exec(statement)
    log = results.one()
    log.misprediction = True
    if not update_condition:
        log.feedback_class = update_condition
    session.add(log)
    session.commit()
    session.refresh(log)
    return
