import os
import shutil
import logging

from dotenv import load_dotenv

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    Depends,
    HTTPException
)

from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from sqlalchemy.orm import Session

from backend.predict import predict_image
from backend.report_generator import generate_report
from backend.database_service import (
    save_prediction,
    get_prediction_history,
    get_dashboard_stats
)

from ml.explain import explain_prediction

from database.connection import SessionLocal
from database.prediction_model import Prediction

from auth.auth import router as auth_router
from auth.oauth2 import get_current_user

# ----------------------------------------------------
# Load Environment Variables
# ----------------------------------------------------

load_dotenv()

UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "uploads")

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)

# ----------------------------------------------------
# Logging
# ----------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

# ----------------------------------------------------
# FastAPI
# ----------------------------------------------------

app = FastAPI(
    title="Brain Tumor Detection API",
    version="1.0.0"
)

# ----------------------------------------------------
# CORS
# ----------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # Replace with Streamlit URL after deployment
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------------------
# Authentication Router
# ----------------------------------------------------

app.include_router(auth_router)

# ----------------------------------------------------
# Database Dependency
# ----------------------------------------------------

def get_db():

    db = SessionLocal()

    try:
        yield db

    finally:
        db.close()

# ----------------------------------------------------
# Health Check
# ----------------------------------------------------

@app.get("/health")
def health():

    return {
        "status": "healthy"
    }

# ----------------------------------------------------
# Home
# ----------------------------------------------------

@app.get("/")
def home():

    return {
        "message": "Brain Tumor Detection API is running."
    }

# ----------------------------------------------------
# Prediction API
# ----------------------------------------------------

@app.post("/predict")
async def predict(

    file: UploadFile = File(...),

    db: Session = Depends(get_db),

    current_user: str = Depends(get_current_user)

):

    try:

        logger.info(f"Prediction requested by {current_user}")

        file_path = os.path.join(
            UPLOAD_FOLDER,
            file.filename
        )

        with open(
            file_path,
            "wb"
        ) as buffer:

            shutil.copyfileobj(
                file.file,
                buffer
            )

        prediction, confidence = predict_image(
            file_path
        )

        gradcam_path = explain_prediction(
            file_path
        )

        report_path = generate_report(

            filename=file.filename,

            prediction=prediction,

            confidence=round(
                confidence * 100,
                2
            ),

            gradcam_path=gradcam_path,

            user_email=current_user
        )

        prediction_record = save_prediction(

            db=db,

            filename=file.filename,

            predicted_class=prediction,

            confidence=round(
                confidence * 100,
                2
            ),

            gradcam_path=gradcam_path,

            report_path=report_path,

            user_email=current_user
        )

        logger.info(
            f"Prediction completed for {current_user}"
        )

        return {

            "prediction_id": prediction_record.id,

            "prediction": prediction,

            "confidence": round(
                confidence * 100,
                2
            ),

            "gradcam_image": gradcam_path,

            "report_path": report_path,

            "message": "Prediction completed successfully."

        }

    except Exception as e:

        logger.exception("Prediction Error")

        raise HTTPException(

            status_code=500,

            detail="Prediction failed."

        )

# ----------------------------------------------------
# Prediction History
# ----------------------------------------------------

@app.get("/history")
def history(

    db: Session = Depends(get_db),

    current_user: str = Depends(get_current_user)

):

    records = get_prediction_history(

        db=db,

        user_email=current_user

    )

    history = []

    for record in records:

        history.append({

            "id": record.id,

            "filename": record.filename,

            "predicted_class": record.predicted_class,

            "confidence": record.confidence,

            "gradcam_path": record.gradcam_path,

            "report_path": record.report_path,

            "prediction_time": record.prediction_time,

            "user_email": record.user_email

        })

    return history

# ----------------------------------------------------
# Dashboard
# ----------------------------------------------------

@app.get("/dashboard")
def dashboard(

    db: Session = Depends(get_db),

    current_user: str = Depends(get_current_user)

):

    return get_dashboard_stats(

        db=db,

        user_email=current_user

    )

# ----------------------------------------------------
# Download Report
# ----------------------------------------------------

@app.get("/download-report/{prediction_id}")
def download_report(

    prediction_id: int,

    db: Session = Depends(get_db),

    current_user: str = Depends(get_current_user)

):

    prediction = (

        db.query(Prediction)

        .filter(

            Prediction.id == prediction_id,

            Prediction.user_email == current_user

        )

        .first()

    )

    if prediction is None:

        raise HTTPException(

            status_code=404,

            detail="Prediction not found."

        )

    if not prediction.report_path:

        raise HTTPException(

            status_code=404,

            detail="Report not available."

        )

    if not os.path.exists(

        prediction.report_path

    ):

        raise HTTPException(

            status_code=404,

            detail="PDF file not found."

        )

    return FileResponse(

        path=prediction.report_path,

        filename=os.path.basename(
            prediction.report_path
        ),

        media_type="application/pdf"

    )