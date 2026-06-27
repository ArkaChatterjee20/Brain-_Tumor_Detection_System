from ml.explain import explain_prediction

IMAGE_PATH = r"uploads\Te-aug-me_1.jpg"

try:
    result = explain_prediction(IMAGE_PATH)

    print("\nGrad-CAM generated successfully!")

    print("Saved at:")

    print(result)

except Exception as e:
    print("\nError:")

    print(e)