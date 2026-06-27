from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    DateTime
)

from database.connection import Base
from datetime import datetime


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(
        Integer,
        primary_key=True,
        index=True
    )

    user_email = Column(
        String(255)
    )

    filename = Column(
        String(255)
    )

    predicted_class = Column(
        String(100)
    )

    confidence = Column(
        Float
    )

    gradcam_path = Column(
        String(255)
    )

    prediction_time = Column(
        DateTime,
        default=datetime.utcnow
    )