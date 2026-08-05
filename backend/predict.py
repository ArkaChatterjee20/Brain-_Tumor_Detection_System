import os
import requests
import numpy as np
from PIL import Image
import tensorflow as tf

MODEL_DIR = "models/final"
MODEL_PATH = os.path.join(MODEL_DIR, "brain_tumor_classifier.keras")
LABEL_PATH = os.path.join(MODEL_DIR, "class_labels.txt")

MODEL_URL = "https://huggingface.co/Arka172/brain-tumor-classifier/resolve/main/brain_tumor_classifier.keras"
LABEL_URL = "https://huggingface.co/Arka172/brain-tumor-classifier/resolve/main/class_labels.txt"


def download_file(url, path):
    if os.path.exists(path):
        return

    os.makedirs(os.path.dirname(path), exist_ok=True)

    print(f"Downloading {os.path.basename(path)}...")

    response = requests.get(
    url,
    stream=True,
    timeout=60
)
    response.raise_for_status()

    with open(path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)


download_file(MODEL_URL, MODEL_PATH)
download_file(LABEL_URL, LABEL_PATH)

model = tf.keras.models.load_model(MODEL_PATH)

with open(LABEL_PATH, "r") as f:
    CLASS_NAMES = [line.strip() for line in f]


def predict_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array, verbose=0)

    predicted_index = np.argmax(predictions)
    confidence = float(np.max(predictions))
    prediction = CLASS_NAMES[predicted_index]

    return prediction, confidence