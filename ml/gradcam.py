import tensorflow as tf
import numpy as np


def generate_gradcam(model, image, last_conv_layer_name):

    print("Generating Grad-CAM using layer:", last_conv_layer_name)
    print("Grad-CAM image shape:", image.shape)

    # --------------------------------------------------
    # Make sure the model is built/called
    # --------------------------------------------------
    _ = model(image, training=False)

    # --------------------------------------------------
    # Find the requested convolution layer
    # --------------------------------------------------
    try:
        last_conv_layer = model.get_layer(last_conv_layer_name)
    except ValueError:
        raise ValueError(
            f"Layer '{last_conv_layer_name}' not found in model."
        )

    print("Grad-CAM layer found:", last_conv_layer.name)

    # --------------------------------------------------
    # Create Grad-CAM model
    # --------------------------------------------------
    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[
            last_conv_layer.output,
            model.outputs[0]
        ]
    )

    print("Grad-CAM model creation successful")

    # --------------------------------------------------
    # Calculate gradients
    # --------------------------------------------------
    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(
            image,
            training=False
        )

        # For binary/single-output model
        if len(predictions.shape) == 1:
            loss = predictions[0]

        elif predictions.shape[-1] == 1:
            loss = predictions[:, 0]

        else:
            pred_index = tf.argmax(
                predictions[0]
            )

            loss = predictions[:, pred_index]

    # --------------------------------------------------
    # Calculate gradients
    # --------------------------------------------------
    grads = tape.gradient(
        loss,
        conv_outputs
    )

    if grads is None:
        raise ValueError(
            "Gradients are None. "
            "Grad-CAM cannot be generated."
        )

    print("Gradients calculated successfully")

    # --------------------------------------------------
    # Global average pooling
    # --------------------------------------------------
    pooled_grads = tf.reduce_mean(
        grads,
        axis=(0, 1, 2)
    )

    # --------------------------------------------------
    # Remove batch dimension
    # --------------------------------------------------
    conv_outputs = conv_outputs[0]

    # --------------------------------------------------
    # Weighted feature maps
    # --------------------------------------------------
    heatmap = tf.reduce_sum(
        conv_outputs * pooled_grads,
        axis=-1
    )

    # --------------------------------------------------
    # ReLU
    # --------------------------------------------------
    heatmap = tf.maximum(
        heatmap,
        0
    )

    # --------------------------------------------------
    # Normalize
    # --------------------------------------------------
    max_value = tf.reduce_max(
        heatmap
    )

    heatmap = heatmap / (
        max_value + tf.keras.backend.epsilon()
    )

    heatmap = heatmap.numpy()

    print(
        "Grad-CAM heatmap generated:",
        heatmap.shape
    )

    return heatmap