from datetime import datetime
from typing import Optional, List
from sqlmodel import Field, SQLModel, JSON, Column

class AbnormalityDetectionLog(SQLModel, table=True):
    image_id : str = Field(primary_key=True)
    file : str
    timestamp : datetime = Field(default_factory=datetime.utcnow, nullable=False)
    inference_time : float
    predicted_bbox : dict = Field(default={}, sa_column=Column(JSON))
    confidence : List[float]
    misprediction : bool = False
    feedback_bbox : dict = Field(default={}, sa_column=Column(JSON))
    
class ConditionClassificationLog(SQLModel, table=True):
    image_id : str = Field(primary_key=True)
    file : str
    timestamp : datetime = Field(default_factory=datetime.utcnow, nullable=False)
    inference_time : float
    misprediction : bool
    predicted_class : str
    confidence : float
    misprediction : bool = False
    feedback_class : Optional[str] = None
