import tensorflow as tf
import numpy as np


def generate_gradcam(model, image, last_conv_layer_name=None):

    print("========================================")
    print("Grad-CAM generation started")
    print("Model type:", type(model).__name__)
    print("Input shape:", image.shape)

    # --------------------------------------------------
    # Find the convolution layer
    # --------------------------------------------------

    if last_conv_layer_name is None:

        for layer in reversed(model.layers):

            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_layer_name = layer.name
                break

    if last_conv_layer_name is None:
        raise ValueError("No Conv2D layer found in model.")

    print(
        "Using Grad-CAM layer:",
        last_conv_layer_name
    )

    last_conv_layer = model.get_layer(
        last_conv_layer_name
    )

    # --------------------------------------------------
    # IMPORTANT:
    # Do NOT use model.output.
    # Build the Grad-CAM graph manually.
    # --------------------------------------------------

    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=last_conv_layer.output
    )

    print("Convolution model created")

    # --------------------------------------------------
    # Forward pass through convolution layers
    # --------------------------------------------------

    with tf.GradientTape() as tape:

        # Watch the convolution output
        conv_output = grad_model(
            image,
            training=False
        )

        tape.watch(conv_output)

        x = conv_output

        # --------------------------------------------------
        # Pass convolution output through remaining layers
        # --------------------------------------------------

        found_layer = False

        for layer in model.layers:

            if layer.name == last_conv_layer_name:
                found_layer = True
                continue

            if found_layer:

                x = layer(
                    x,
                    training=False
                )

        predictions = x

        print(
            "Conv output shape:",
            conv_output.shape
        )

        print(
            "Prediction shape:",
            predictions.shape
        )

        # --------------------------------------------------
        # Predicted class
        # --------------------------------------------------

        pred_index = tf.argmax(
            predictions[0]
        )

        class_score = predictions[:, pred_index]

    # --------------------------------------------------
    # Calculate gradients
    # --------------------------------------------------

    grads = tape.gradient(
        class_score,
        conv_output
    )

    print(
        "Gradient object:",
        grads
    )

    if grads is None:

        raise ValueError(
            "Gradients are None. "
            "Grad-CAM cannot be generated."
        )

    print(
        "Gradient shape:",
        grads.shape
    )

    # --------------------------------------------------
    # Global average pooling
    # --------------------------------------------------

    pooled_grads = tf.reduce_mean(
        grads,
        axis=(1, 2)
    )

    # Remove batch dimension
    conv_output = conv_output[0]

    pooled_grads = pooled_grads[0]

    # --------------------------------------------------
    # Weighted activation maps
    # --------------------------------------------------

    heatmap = tf.reduce_sum(
        conv_output * pooled_grads,
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
        "Heatmap shape:",
        heatmap.shape
    )

    print(
        "Heatmap min:",
        heatmap.min()
    )

    print(
        "Heatmap max:",
        heatmap.max()
    )

    print(
        "Grad-CAM generation completed"
    )

    print("========================================")

    return heatmap