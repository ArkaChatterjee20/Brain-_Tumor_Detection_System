import tensorflow as tf
import numpy as np


# ---------------------------------------------------------
# Build a Grad-CAM model safely for a Sequential model
# ---------------------------------------------------------
def _build_gradcam_model(model, last_conv_layer_name):

    print("Building Grad-CAM model...")

    # Create a fresh Functional input
    inputs = tf.keras.Input(
        shape=(224, 224, 3),
        name="gradcam_input"
    )

    x = inputs
    conv_output = None

    # Reuse the existing trained layers
    for layer in model.layers:

        x = layer(x)

        if layer.name == last_conv_layer_name:
            conv_output = x

    if conv_output is None:
        raise ValueError(
            f"Could not find convolution layer "
            f"'{last_conv_layer_name}' while rebuilding "
            f"the Grad-CAM graph."
        )

    predictions = x

    grad_model = tf.keras.Model(
        inputs=inputs,
        outputs=[
            conv_output,
            predictions
        ],
        name="gradcam_model"
    )

    print(
        "Grad-CAM model initialized successfully."
    )

    return grad_model


# ---------------------------------------------------------
# Grad-CAM
# ---------------------------------------------------------
def generate_gradcam(
    model,
    image,
    last_conv_layer_name="conv2d_2"
):

    print("========================================")
    print("Grad-CAM generation started")
    print("Input shape:", image.shape)
    print("Model type:", type(model).__name__)

    # -----------------------------------------------------
    # Ensure correct input shape
    # -----------------------------------------------------
    if len(image.shape) == 3:

        image = tf.expand_dims(
            image,
            axis=0
        )

    image = tf.cast(
        image,
        tf.float32
    )

    print(
        "Grad-CAM input shape:",
        image.shape
    )

    # -----------------------------------------------------
    # Find requested convolution layer
    # -----------------------------------------------------
    try:

        model.get_layer(
            last_conv_layer_name
        )

    except ValueError:

        print(
            f"Layer '{last_conv_layer_name}' "
            f"not found."
        )

        last_conv_layer_name = None

        # Automatically find the last Conv2D
        for layer in reversed(model.layers):

            if isinstance(
                layer,
                tf.keras.layers.Conv2D
            ):

                last_conv_layer_name = layer.name
                break

        if last_conv_layer_name is None:

            raise ValueError(
                "No Conv2D layer found in model."
            )

    print(
        "Using Grad-CAM layer:",
        last_conv_layer_name
    )

    # -----------------------------------------------------
    # IMPORTANT:
    #
    # Do NOT use:
    #
    #     model.output
    #
    # because your loaded Sequential model does not expose
    # a usable output tensor.
    #
    # Instead, rebuild a small Functional graph using the
    # SAME trained layers and weights.
    # -----------------------------------------------------
    grad_model = _build_gradcam_model(
        model,
        last_conv_layer_name
    )

    # -----------------------------------------------------
    # Gradient calculation
    # -----------------------------------------------------
    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(
            image,
            training=False
        )

        print(
            "Conv output shape:",
            conv_outputs.shape
        )

        print(
            "Prediction shape:",
            predictions.shape
        )

        # -------------------------------------------------
        # Find predicted class
        # -------------------------------------------------
        pred_index = tf.argmax(
            predictions[0]
        )

        pred_index_int = int(
            pred_index.numpy()
        )

        print(
            "Predicted class index:",
            pred_index_int
        )

        # -------------------------------------------------
        # Select predicted class score
        # -------------------------------------------------
        class_score = predictions[
            :,
            pred_index
        ]

    # -----------------------------------------------------
    # Calculate gradients
    # -----------------------------------------------------
    grads = tape.gradient(
        class_score,
        conv_outputs
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

    # -----------------------------------------------------
    # Global average pooling
    # -----------------------------------------------------
    pooled_grads = tf.reduce_mean(
        grads,
        axis=(1, 2)
    )

    # -----------------------------------------------------
    # Remove batch dimension
    # -----------------------------------------------------
    conv_outputs = conv_outputs[0]

    pooled_grads = pooled_grads[0]

    # -----------------------------------------------------
    # Weighted feature maps
    # -----------------------------------------------------
    heatmap = tf.reduce_sum(
        conv_outputs * pooled_grads,
        axis=-1
    )

    # -----------------------------------------------------
    # ReLU
    # -----------------------------------------------------
    heatmap = tf.maximum(
        heatmap,
        0
    )

    # -----------------------------------------------------
    # Normalize
    # -----------------------------------------------------
    max_value = tf.reduce_max(
        heatmap
    )

    if float(max_value.numpy()) > 0:

        heatmap = (
            heatmap /
            max_value
        )

    else:

        print(
            "Warning: Grad-CAM heatmap "
            "maximum is zero."
        )

        heatmap = tf.zeros_like(
            heatmap
        )

    # -----------------------------------------------------
    # Convert to NumPy
    # -----------------------------------------------------
    heatmap = heatmap.numpy()

    # -----------------------------------------------------
    # Validate heatmap
    # -----------------------------------------------------
    if not np.isfinite(
        heatmap
    ).all():

        raise ValueError(
            "Grad-CAM heatmap contains "
            "NaN or infinite values."
        )

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