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
    file_path = None
    gradcam_path = None
    report_path = None

    try:
        logger.info(
            f"Prediction requested by {current_user}"
        )

        # ------------------------------------------------
        # Save uploaded MRI
        # ------------------------------------------------

        original_filename = os.path.basename(file.filename)

        safe_filename = f"{uuid4().hex}_{original_filename}" 

        file_path = os.path.join(
            UPLOAD_FOLDER,
            safe_filename
        )

        logger.info("Saving uploaded MRI...")

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(
                file.file,
                buffer
            )

        logger.info(
            f"MRI saved successfully: {file_path}"
        )

        # ------------------------------------------------
        # Prediction
        # ------------------------------------------------

        logger.info(
            "Starting TensorFlow prediction..."
        )

        prediction, confidence = predict_image(
            file_path
        )

        logger.info(
            f"TensorFlow prediction completed: "
            f"{prediction}"
        )

        confidence_percent = round(
            confidence * 100,
            2
        )

        logger.info(
            f"Confidence: {confidence_percent}%"
        )

        # ------------------------------------------------
        # Grad-CAM
        # ------------------------------------------------

        logger.info("Grad-CAM temporarily disabled for deployment testing.")

        gradcam_path = None

        # ------------------------------------------------
        # Upload Grad-CAM
        # ------------------------------------------------

        logger.info(
            "Uploading Grad-CAM to Supabase..."
        )

        gradcam_storage_path = None
        if gradcam_path:
          gradcam_storage_path = upload_file(
            gradcam_path,
            f"gradcam/{os.path.basename(gradcam_path)}",
            "image/jpeg"
        )

        logger.info(
            f"Grad-CAM uploaded: "
            f"{gradcam_storage_path}"
        )

        # ------------------------------------------------
        # Generate PDF
        # ------------------------------------------------

        logger.info(
            "Generating PDF report..."
        )

        report_path = generate_report(
            filename=safe_filename,
            prediction=prediction,
            confidence=confidence_percent,
            gradcam_path=gradcam_path,
            user_email=current_user
        )

        logger.info(
            f"PDF generated: {report_path}"
        )

        # ------------------------------------------------
        # Upload PDF
        # ------------------------------------------------

        logger.info(
            "Uploading PDF to Supabase..."
        )

        report_storage_path = upload_file(
            report_path,
            f"reports/{os.path.basename(report_path)}",
            "application/pdf"
        )

        logger.info(
            f"PDF uploaded: {report_storage_path}"
        )

        # ------------------------------------------------
        # Save database record
        # ------------------------------------------------

        logger.info(
            "Saving prediction to database..."
        )

        prediction_record = save_prediction(
            db=db,
            filename=safe_filename,
            predicted_class=prediction,
            confidence=confidence_percent,
            gradcam_path=gradcam_storage_path,
            report_path=report_storage_path,
            user_email=current_user
        )

        logger.info(
            f"Database record saved: "
            f"{prediction_record.id}"
        )

        # ------------------------------------------------
        # Signed URLs
        # ------------------------------------------------

        logger.info(
            "Creating signed URLs..."
        )

        gradcam_url =  None
        if prediction_record.gradcam_path:
          gradcam_url = create_signed_url(
              prediction_record.gradcam_path,
              expires_in=3600
    )

        report_url = create_signed_url(
            prediction_record.report_path,
            expires_in=3600
        )

        # ------------------------------------------------
        # Cleanup
        # ------------------------------------------------

        logger.info(
            "Cleaning temporary files..."
        )

        for path in [
            file_path,
            gradcam_path,
            report_path
        ]:
            try:
                if path and os.path.exists(path):
                    os.remove(path)
            except Exception as cleanup_error:
                logger.warning(
                    f"Cleanup failed for {path}: "
                    f"{cleanup_error}"
                )

        logger.info(
            f"Prediction completed successfully "
            f"for {current_user}"
        )

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

        # Cleanup even when an error occurs
        for path in [
            file_path,
            gradcam_path,
            report_path
        ]:
            try:
                if path and os.path.exists(path):
                    os.remove(path)
            except Exception:
                pass

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