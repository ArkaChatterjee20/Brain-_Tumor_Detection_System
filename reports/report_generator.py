from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer
)
from reportlab.lib.styles import getSampleStyleSheet
from datetime import datetime
import os


def generate_report(
    image_path,
    prediction,
    confidence,
    gradcam_path
):
    os.makedirs(
        "reports/generated_reports",
        exist_ok=True
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    report_path = (
        f"reports/generated_reports/"
        f"report_{timestamp}.pdf"
    )

    doc = SimpleDocTemplate(report_path)

    styles = getSampleStyleSheet()

    elements = []

    title = Paragraph(
        "BRAIN TUMOR DETECTION REPORT",
        styles["Title"]
    )

    elements.append(title)

    elements.append(Spacer(1, 20))

    prediction_text = Paragraph(
        f"Prediction : {prediction}",
        styles["BodyText"]
    )

    confidence_text = Paragraph(
        f"Confidence : {confidence:.2f} %",
        styles["BodyText"]
    )

    elements.append(prediction_text)

    elements.append(Spacer(1, 10))

    elements.append(confidence_text)

    doc.build(elements)

    return report_path