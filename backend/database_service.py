from sqlalchemy.orm import Session
from sqlalchemy import func, case

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

    # Base query
    query = db.query(Prediction)

    if user_email:
        query = query.filter(
            Prediction.user_email == user_email
        )

    # Get prediction statistics in ONE database query
    stats = query.with_entities(
        func.count(Prediction.id).label("total_predictions"),

        func.sum(
            case(
                (Prediction.predicted_class == "glioma", 1),
                else_=0
            )
        ).label("glioma"),

        func.sum(
            case(
                (Prediction.predicted_class == "meningioma", 1),
                else_=0
            )
        ).label("meningioma"),

        func.sum(
            case(
                (Prediction.predicted_class == "pituitary", 1),
                else_=0
            )
        ).label("pituitary"),

        func.sum(
            case(
                (Prediction.predicted_class == "no_tumor", 1),
                else_=0
            )
        ).label("notumor"),

        func.avg(
            Prediction.confidence
        ).label("average_confidence")

    ).first()

    # Total users
    total_users = db.query(
        func.count(User.id)
    ).scalar()

    return {

        "total_users": total_users or 0,

        "total_predictions": stats.total_predictions or 0,

        "glioma": stats.glioma or 0,

        "meningioma": stats.meningioma or 0,

        "pituitary": stats.pituitary or 0,

        "notumor": stats.notumor or 0,

        "average_confidence": round(
            float(stats.average_confidence or 0),
            2
        )
    }