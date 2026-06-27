from pydantic import BaseModel
from datetime import datetime


class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    gradcam_image: str


class HistoryResponse(BaseModel):
    id: int
    filename: str
    predicted_class: str
    confidence: float
    gradcam_path: str
    prediction_time: datetime

    class Config:
        from_attributes = True