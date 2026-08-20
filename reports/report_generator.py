from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image as ReportImage
)

from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4

from datetime import datetime

import os


def generate_report(
    image_path,
    prediction,
    confidence,
    gradcam_path
):

    # ------------------------------------------------
    # Create report directory
    # ------------------------------------------------

    report_directory = (
        "reports/generated_reports"
    )

    os.makedirs(
        report_directory,
        exist_ok=True
    )

    # ------------------------------------------------
    # Unique report filename
    # ------------------------------------------------

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    report_path = os.path.join(
        report_directory,
        f"report_{timestamp}.pdf"
    )

    # ------------------------------------------------
    # PDF document
    # ------------------------------------------------

    doc = SimpleDocTemplate(
        report_path,
        pagesize=A4
    )

    styles = getSampleStyleSheet()

    elements = []

    # ------------------------------------------------
    # Title
    # ------------------------------------------------

    elements.append(
        Paragraph(
            "BRAIN TUMOR DETECTION REPORT",
            styles["Title"]
        )
    )

    elements.append(
        Spacer(1, 20)
    )

    # ------------------------------------------------
    # Prediction
    # ------------------------------------------------

    elements.append(
        Paragraph(
            f"<b>Prediction:</b> {prediction}",
            styles["BodyText"]
        )
    )

    elements.append(
        Spacer(1, 10)
    )

    # ------------------------------------------------
    # Confidence
    # ------------------------------------------------

    elements.append(
        Paragraph(
            f"<b>Confidence:</b> {confidence:.2f}%",
            styles["BodyText"]
        )
    )

    elements.append(
        Spacer(1, 20)
    )

    # ------------------------------------------------
    # MRI Image
    # ------------------------------------------------

    if image_path and os.path.exists(
        image_path
    ):

        elements.append(
            Paragraph(
                "<b>Uploaded MRI:</b>",
                styles["Heading2"]
            )
        )

        elements.append(
            Spacer(1, 10)
        )

        try:

            image = ReportImage(
                image_path,
                width=300,
                height=300
            )

            elements.append(image)

            elements.append(
                Spacer(1, 20)
            )

        except Exception:
            pass

    # ------------------------------------------------
    # Grad-CAM Image
    # ------------------------------------------------

    if gradcam_path and os.path.exists(
        gradcam_path
    ):

        elements.append(
            Paragraph(
                "<b>Grad-CAM Explanation:</b>",
                styles["Heading2"]
            )
        )

        elements.append(
            Spacer(1, 10)
        )

        try:

            gradcam_image = ReportImage(
                gradcam_path,
                width=300,
                height=300
            )

            elements.append(
                gradcam_image
            )

            elements.append(
                Spacer(1, 20)
            )

        except Exception:
            pass

    # ------------------------------------------------
    # Timestamp
    # ------------------------------------------------

    elements.append(
        Paragraph(
            f"<b>Generated:</b> "
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            styles["BodyText"]
        )
    )

    # ------------------------------------------------
    # Build PDF
    # ------------------------------------------------

    doc.build(
        elements
    )

    return report_path