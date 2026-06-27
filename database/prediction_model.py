from sqlalchemy import (

    Column,

    Integer,

    String,

    Float,

    DateTime

)

from datetime import datetime

from database.connection import Base


class Prediction(Base):

    __tablename__ = "predictions"

    id = Column(

        Integer,

        primary_key=True,

        index=True

    )

    filename = Column(

        String(255),

        nullable=False

    )

    predicted_class = Column(

        String(100),

        nullable=False

    )

    confidence = Column(

        Float,

        nullable=False

    )

    gradcam_path = Column(

        String(500),

        nullable=True

    )

    report_path = Column(

        String(500),

        nullable=True

    )

    prediction_time = Column(

        DateTime,

        default=datetime.utcnow

    )

    user_email = Column(

        String(255),

        nullable=False

    )