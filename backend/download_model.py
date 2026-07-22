import os
import requests

MODEL_DIR = "models/final"
MODEL_PATH = os.path.join(MODEL_DIR, "brain_tumor_classifier.keras")

MODEL_URL = "https://huggingface.co/Arka172/brain-tumor-classifier/resolve/main/brain_tumor_classifier.keras"


def download_model():
    if os.path.exists(MODEL_PATH):
        print("✅ Model already exists.")
        return

    os.makedirs(MODEL_DIR, exist_ok=True)

    print("Downloading model from Hugging Face...")

    response = requests.get(MODEL_URL, stream=True)
    response.raise_for_status()

    with open(MODEL_PATH, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    print("✅ Model downloaded successfully.")