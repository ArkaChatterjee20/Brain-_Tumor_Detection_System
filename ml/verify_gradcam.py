import tensorflow as tf

MODEL_PATH = "models/final/brain_tumor_classifier.keras"
LAST_CONV_LAYER = "conv2d_2"


def verify_gradcam_compatibility():
    """
    Verify whether the trained CNN model
    is compatible with Grad-CAM.
    """

    try:
        print("Loading model...")

        model = tf.keras.models.load_model(MODEL_PATH)

        print("Model loaded successfully.")

        # Build the graph
        dummy = tf.random.normal((1, 224, 224, 3))
        _ = model(dummy)

        print("\nModel Information")
        print("-" * 40)

        print("Built:", model.built)

        print("Inputs:", model.inputs)

        print("Outputs:", model.outputs)

        # Verify Conv Layer
        conv_layer = model.get_layer(LAST_CONV_LAYER)

        print("\nLast Conv Layer Found:")
        print(conv_layer)

        # Create Grad-CAM model
        grad_model = tf.keras.Model(
            inputs=model.inputs,
            outputs=[
                conv_layer.output,
                model.outputs[0]
            ]
        )

        print("\nGrad-CAM model created successfully!")

        print("\nVerification Successful")
        print("Your CNN model supports Grad-CAM.")

        return True

    except Exception as e:

        print("\nVerification Failed")
        print("Error:")

        print(e)

        return False


if __name__ == "__main__":

    verify_gradcam_compatibility()