from reports.report_generator import generate_report


report_path = generate_report(
    image_path="uploads/test.jpg",
    prediction="Pituitary",
    confidence=99.87,
    gradcam_path="outputs/gradcam/test.jpg"
)

print("PDF report generated successfully!")

print("Saved at:")

print(report_path)