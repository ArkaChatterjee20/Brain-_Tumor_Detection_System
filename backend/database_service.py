from sqlalchemy.orm import Session
from sqlalchemy import func

from database.prediction_model import Prediction
from database.user_model import User


# ---------------------------------------------------
# Save Prediction
# ---------------------------------------------------
def save_prediction(
    db: Session,
    filename: str,
    predicted_class: str,
    confidence: float,
    gradcam_path: str,
    report_path: str,
    user_email: str
):

    prediction = Prediction(
        filename=filename,
        predicted_class=predicted_class,
        confidence=confidence,
        gradcam_path=gradcam_path,
        report_path=report_path,
        user_email=user_email
    )

    db.add(prediction)
    db.commit()
    db.refresh(prediction)

    return prediction


# ---------------------------------------------------
# Prediction History
# ---------------------------------------------------
def get_prediction_history(
    db: Session,
    user_email: str = None
):

    query = db.query(Prediction)

    if user_email:
        query = query.filter(
            Prediction.user_email == user_email
        )

    return (
        query.order_by(
            Prediction.prediction_time.desc()
        )
        .all()
    )


# ---------------------------------------------------
# Dashboard Statistics
# ---------------------------------------------------
def get_dashboard_stats(
    db: Session,
    user_email: str = None
):

    query = db.query(Prediction)

    if user_email:
        query = query.filter(
            Prediction.user_email == user_email
        )

    total_predictions = query.count()

    total_users = db.query(User).count()

    glioma = query.filter(
        Prediction.predicted_class == "Glioma"
    ).count()

    meningioma = query.filter(
        Prediction.predicted_class == "Meningioma"
    ).count()

    pituitary = query.filter(
        Prediction.predicted_class == "Pituitary"
    ).count()

    notumor = query.filter(
        Prediction.predicted_class == "No Tumor"
    ).count()

    average_confidence = query.with_entities(
        func.avg(Prediction.confidence)
    ).scalar()

    if average_confidence is None:
        average_confidence = 0

    return {

        "total_users": total_users,

        "total_predictions": total_predictions,

        "glioma": glioma,

        "meningioma": meningioma,

        "pituitary": pituitary,

        "notumor": notumor,

        "average_confidence": round(
            average_confidence,
            2
        )

    }