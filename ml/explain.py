import os
import cv2
import numpy as np
from PIL import Image

from backend.predict import model
from ml.gradcam import generate_gradcam


LAST_CONV_LAYER = "conv2d_2"


BASE_DIR = os.getcwd()

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs",
    "gradcam"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


print(
    "Grad-CAM will use the shared TensorFlow model."
)


# ----------------------------------------------------
# Explain prediction
# ----------------------------------------------------

def explain_prediction(image_path):

    try:

        # ---------------------------------------------
        # Load image
        # ---------------------------------------------

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


        # ---------------------------------------------
        # Generate Grad-CAM
        # ---------------------------------------------

        heatmap = generate_gradcam(
            model,
            input_image,
            LAST_CONV_LAYER
        )


        # ---------------------------------------------
        # Load original image
        # ---------------------------------------------

        original = cv2.imread(
            image_path
        )

        if original is None:

            raise ValueError(
                "Unable to read original image."
            )

        original = cv2.resize(
            original,
            (224, 224)
        )


        # ---------------------------------------------
        # Resize heatmap
        # ---------------------------------------------

        heatmap = cv2.resize(
            heatmap,
            (224, 224)
        )

        heatmap = np.uint8(
            255 * heatmap
        )


        # ---------------------------------------------
        # Apply color map
        # ---------------------------------------------

        heatmap = cv2.applyColorMap(
            heatmap,
            cv2.COLORMAP_JET
        )


        # ---------------------------------------------
        # Overlay
        # ---------------------------------------------

        overlay = cv2.addWeighted(
            original,
            0.6,
            heatmap,
            0.4,
            0
        )


        # ---------------------------------------------
        # Save Grad-CAM
        # ---------------------------------------------

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