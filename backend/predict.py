import numpy as np
from PIL import Image
import tensorflow as tf

MODEL_PATH = "models/final/brain_tumor_classifier.keras"
LABEL_PATH = "models/final/class_labels.txt"

# Load model once
model = tf.keras.models.load_model(MODEL_PATH)

# Load labels
with open(LABEL_PATH, "r") as f:
    CLASS_NAMES = [line.strip() for line in f]


def predict_image(image_path):
    """
    Predict tumor type from MRI image.
    """

    img = Image.open(image_path).convert("RGB")

    img = img.resize((224, 224))

    img_array = np.array(img) / 255.0

    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array, verbose=0)

    predicted_index = np.argmax(predictions)

    confidence = float(np.max(predictions))

    prediction = CLASS_NAMES[predicted_index]

    return prediction, confidence