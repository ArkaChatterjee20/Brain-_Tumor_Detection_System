import tensorflow as tf
import numpy as np


def generate_gradcam(model, image, last_conv_layer_name=None):
    """
    Generate Grad-CAM heatmap for a TensorFlow/Keras classification model.

    Parameters
    ----------
    model : tf.keras.Model
        Loaded TensorFlow model.

    image : np.ndarray
        Preprocessed image with shape (1, 224, 224, 3).

    last_conv_layer_name : str, optional
        Name of the convolutional layer to use.
        If None, the last 4D convolutional layer is detected automatically.

    Returns
    -------
    np.ndarray
        Normalized Grad-CAM heatmap.
    """

    # --------------------------------------------------
    # 1. Validate input
    # --------------------------------------------------

    if model is None:
        raise ValueError("TensorFlow model is None.")

    if image is None:
        raise ValueError("Input image is None.")

    image = tf.convert_to_tensor(image, dtype=tf.float32)

    if len(image.shape) != 4:
        raise ValueError(
            f"Expected image shape (batch, height, width, channels), "
            f"got {image.shape}"
        )

    print("Grad-CAM image shape:", image.shape)

    # --------------------------------------------------
    # 2. Find convolutional layer
    # --------------------------------------------------

    if last_conv_layer_name is not None:

        try:
            last_conv_layer = model.get_layer(
                last_conv_layer_name
            )

            print(
                "Using Grad-CAM layer:",
                last_conv_layer.name
            )

        except Exception:

            print(
                f"Layer '{last_conv_layer_name}' not found."
            )

            last_conv_layer = None

    else:
        last_conv_layer = None

    # --------------------------------------------------
    # 3. Automatically find last 4D layer if needed
    # --------------------------------------------------

    if last_conv_layer is None:

        for layer in reversed(model.layers):
             if isinstance(layer, tf.keras.layers.Conv2D):
    
                last_conv_layer = layer

                print(
                "Automatically selected Grad-CAM layer:",
                layer.name
                )

                break

            

    if last_conv_layer is None:

        raise ValueError(
            "Could not find a suitable convolutional layer."
        )

    # --------------------------------------------------
    # 4. Create Grad-CAM model
    # --------------------------------------------------
    print("Grad-CAM model creation started")

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            last_conv_layer.output,
            model.output
        ]
    )
    print("Grad-CAM model created successfully")

    # --------------------------------------------------
    # 5. Calculate gradients
    # --------------------------------------------------
    print("Starting GradientTape...")

    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(
            image,
            training=False
        )

        # Handle list/tuple model outputs
        if isinstance(predictions, (list, tuple)):
            predictions = predictions[0]

        print(
            "Prediction shape:",
            predictions.shape
        )

        # Get predicted class
        if len(predictions.shape) == 2:

            pred_index = tf.argmax(
                predictions[0]
            )

            class_channel = predictions[
                :, pred_index
            ]

        else:

            pred_index = tf.argmax(
                predictions
            )

            class_channel = predictions[
                0, pred_index
            ]

        print(
            "Grad-CAM predicted class:",
            int(pred_index.numpy())
        )

    # --------------------------------------------------
    # 6. Calculate gradients
    # --------------------------------------------------

    grads = tape.gradient(
        class_channel,
        conv_outputs
    )
    print("Gradient calculation finished")

    print(
        "Gradients calculated:",
        grads is not None
    )

    if grads is None:

        raise ValueError(
            "Gradients are None. "
            "Grad-CAM cannot be generated for this layer."
        )

    # --------------------------------------------------
    # 7. Global average pooling
    # --------------------------------------------------

    pooled_grads = tf.reduce_mean(
        grads,
        axis=(1, 2)
    )

    # --------------------------------------------------
    # 8. Remove batch dimension
    # --------------------------------------------------

    conv_outputs = conv_outputs[0]

    pooled_grads = pooled_grads[0]

    # --------------------------------------------------
    # 9. Create heatmap
    # --------------------------------------------------

    heatmap = tf.reduce_sum(
        conv_outputs * pooled_grads,
        axis=-1
    )

    # --------------------------------------------------
    # 10. ReLU
    # --------------------------------------------------

    heatmap = tf.maximum(
        heatmap,
        0
    )

    # --------------------------------------------------
    # 11. Normalize
    # --------------------------------------------------

    max_value = tf.reduce_max(
        heatmap
    )

    if float(max_value.numpy()) <= 0:

        raise ValueError(
            "Grad-CAM heatmap is empty."
        )

    heatmap = heatmap / (
        max_value + tf.keras.backend.epsilon()
    )

    # --------------------------------------------------
    # 12. Convert to NumPy
    # --------------------------------------------------

    heatmap = heatmap.numpy()

    print(
        "Grad-CAM generated successfully."
    )

    print(
        "Heatmap shape:",
        heatmap.shape
    )

    print(
        "Heatmap min:",
        float(np.min(heatmap))
    )

    print(
        "Heatmap max:",
        float(np.max(heatmap))
    )

    return heatmap