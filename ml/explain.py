import os
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

from ml.gradcam import generate_gradcam

MODEL_PATH = "models/final/brain_tumor_classifier.keras"

LAST_CONV_LAYER = "conv2d_2"

OUTPUT_DIR = "outputs/gradcam"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

print("Loading model for Grad-CAM...")

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

dummy = tf.random.normal(
    (1, 224, 224, 3)
)

_ = model(dummy)

print("Model loaded successfully.")


def explain_prediction(image_path):

    try:

        img = Image.open(
            image_path
        ).convert("RGB")

        img = img.resize(
            (224, 224)
        )

        img_array = np.array(
            img,
            dtype=np.float32
        ) / 255.0

        input_image = np.expand_dims(
            img_array,
            axis=0
        )

        heatmap = generate_gradcam(
            model,
            input_image,
            LAST_CONV_LAYER
        )

        original = cv2.imread(
            image_path
        )

        original = cv2.resize(
            original,
            (224, 224)
        )

        heatmap = cv2.resize(
            heatmap,
            (224, 224)
        )

        heatmap = np.uint8(
            255 * heatmap
        )

        heatmap = cv2.applyColorMap(
            heatmap,
            cv2.COLORMAP_JET
        )

        overlay = cv2.addWeighted(
            original,
            0.6,
            heatmap,
            0.4,
            0
        )

        filename = os.path.basename(
            image_path
        )

        output_path = os.path.join(
            OUTPUT_DIR,
            "gradcam_" + filename
        )

        cv2.imwrite(
            output_path,
            overlay
        )

        return output_path

    except Exception as e:

        print(
            "GradCAM Error:",
            str(e)
        )

        return None