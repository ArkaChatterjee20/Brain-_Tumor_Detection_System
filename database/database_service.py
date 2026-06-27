from sqlalchemy.orm import Session
from database.models import Prediction


def save_prediction(
        db: Session,
        user_email,
        filename,
        predicted_class,
        confidence,
        gradcam_path
):

    prediction = Prediction(

        user_email=user_email,

        filename=filename,

        predicted_class=predicted_class,

        confidence=confidence,

        gradcam_path=gradcam_path

    )

    db.add(prediction)

    db.commit()

    db.refresh(prediction)

    return prediction


def get_prediction_history(
        db: Session,
        user_email
):

    return (

        db.query(Prediction)

        .filter(
            Prediction.user_email == user_email
        )

        .order_by(
            Prediction.prediction_time.desc()
        )

        .all()

    )