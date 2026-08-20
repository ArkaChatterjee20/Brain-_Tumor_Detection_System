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


from auth.auth import router as auth_router
from auth.oauth2 import get_current_user
from database.connection import Base, engine

from database.prediction_model import Prediction
from backend.supabase_storage import (
    upload_file,
    create_signed_url
)
from uuid import uuid4



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
# Create database tables
Base.metadata.create_all(bind=engine)
app = FastAPI(
    title="Brain Tumor Detection API",
    version="1.0.0"
)

# ----------------------------------------------------
# CORS
# ----------------------------------------------------
FRONTEND_URL = os.getenv(
    "FRONTEND_URL",
    "http://localhost:8501"
)



app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_URL],      # Replace with Streamlit URL after deployment
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

        logger.info(
            f"Prediction requested by {current_user}"
        )

        # ------------------------------------------------
        # Save uploaded MRI temporarily
        # ------------------------------------------------

        safe_filename = f"{uuid4().hex}_{file.filename}"

        file_path = os.path.join(
            UPLOAD_FOLDER,
            safe_filename
        )

        with open(file_path, "wb") as buffer:

            shutil.copyfileobj(
                file.file,
                buffer
            )

        # ------------------------------------------------
        # Prediction
        # ------------------------------------------------

        prediction, confidence = predict_image(
            file_path
        )

        confidence_percent = round(
            confidence * 100,
            2
        )

        # ------------------------------------------------
        # Generate Grad-CAM
        # ------------------------------------------------

        gradcam_path = explain_prediction(
            file_path
        )

        # ------------------------------------------------
        # Upload Grad-CAM to Supabase
        # ------------------------------------------------

        gradcam_storage_path = upload_file(
           gradcam_path,
           f"gradcam/{uuid4().hex}_{os.path.basename(gradcam_path)}",
           "image/jpeg"
        )

        # ------------------------------------------------
        # Generate PDF report
        # ------------------------------------------------

        report_path = generate_report(
            filename=file.filename,
            prediction=prediction,
            confidence=confidence_percent,
            gradcam_path=gradcam_path,
            user_email=current_user
        )

        # ------------------------------------------------
        # Upload PDF to Supabase
        # ------------------------------------------------

        report_storage_path = upload_file(
          report_path,
          f"reports/{uuid4().hex}_{os.path.basename(report_path)}",
         "application/pdf"
        )

        # ------------------------------------------------
        # Save prediction in database
        # ------------------------------------------------

        prediction_record = save_prediction(
            db=db,
            filename=file.filename,
            predicted_class=prediction,
            confidence=confidence_percent,
            gradcam_path=gradcam_storage_path,
            report_path=report_storage_path,
            user_email=current_user
        )

        logger.info(
            f"Prediction completed for {current_user}"
        )

        # ------------------------------------------------
        # Create signed URLs
        # ------------------------------------------------

        gradcam_url = create_signed_url(
            prediction_record.gradcam_path,
            expires_in=3600
        )

        report_url = create_signed_url(
            prediction_record.report_path,
            expires_in=3600
        )

        # ------------------------------------------------
        # Remove temporary local files
        # ------------------------------------------------

        try:

            if os.path.exists(file_path):
                os.remove(file_path)

            if os.path.exists(gradcam_path):
                os.remove(gradcam_path)

            if os.path.exists(report_path):
                os.remove(report_path)

        except Exception as cleanup_error:

            logger.warning(
                f"Temporary file cleanup failed: "
                f"{cleanup_error}"
            )

        # ------------------------------------------------
        # Response
        # ------------------------------------------------

        return {
            "prediction_id": prediction_record.id,
            "prediction": prediction,
            "confidence": confidence_percent,
            "gradcam_url": gradcam_url,
            "report_url": report_url,
            "message": "Prediction completed successfully."
        }

    except Exception as e:

        logger.exception(
            "Prediction Error"
        )

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

# ----------------------------------------------------
# Prediction History
# -----------------------------------------------
@app.get("/history")
def history(

    db: Session = Depends(get_db),

    current_user: str = Depends(
        get_current_user
    )

):

    records = get_prediction_history(

        db=db,

        user_email=current_user

    )

    history = []


    for record in records:

        gradcam_url = None
        report_url = None


        # --------------------------------------------
        # Grad-CAM signed URL
        # --------------------------------------------

        if record.gradcam_path:

            try:

                gradcam_url = create_signed_url(
                    record.gradcam_path,
                    expires_in=86400
                )

            except Exception as e:

                logger.exception(
                    f"Grad-CAM URL generation failed "
                    f"for prediction {record.id}"
                )


        # --------------------------------------------
        # Report signed URL
        # --------------------------------------------

        if record.report_path:

            try:

                report_url = create_signed_url(
                    record.report_path,
                    expires_in=86400
                )

            except Exception as e:

                logger.exception(
                    f"Report URL generation failed "
                    f"for prediction {record.id}"
                )


        # --------------------------------------------
        # Debug logging
        # --------------------------------------------

        logger.info(
            f"Prediction ID: {record.id}"
        )

        logger.info(
            f"GradCAM path: {record.gradcam_path}"
        )

        logger.info(
            f"GradCAM URL: {gradcam_url}"
        )

        logger.info(
            f"Report path: {record.report_path}"
        )

        logger.info(
            f"Report URL: {report_url}"
        )


        # --------------------------------------------
        # History record
        # --------------------------------------------

        history.append({

            "id": record.id,

            "filename": record.filename,

            "predicted_class":
                record.predicted_class,

            "confidence":
                record.confidence,

            "gradcam_url":
                gradcam_url,

            "report_url":
                report_url,

            "prediction_time":
                record.prediction_time,

            "user_email":
                record.user_email

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

    try:

        report_url = create_signed_url(
            prediction.report_path,
            expires_in=3600
        )

        return {
            "report_url": report_url
        }

    except Exception as e:

        logger.exception(
            "Report signed URL generation failed"
        )

        raise HTTPException(
            status_code=500,
            detail="Unable to access report."
        )
@app.get("/gradcam/{prediction_id}")
def download_gradcam(
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

    if not prediction.gradcam_path:

        raise HTTPException(
            status_code=404,
            detail="Grad-CAM unavailable."
        )

    try:

        gradcam_url = create_signed_url(
            prediction.gradcam_path,
            expires_in=3600
        )

        return {
            "gradcam_url": gradcam_url
        }

    except Exception as e:

        logger.exception(
            "Grad-CAM signed URL generation failed"
        )

        raise HTTPException(
            status_code=500,
            detail="Unable to access Grad-CAM."
        )