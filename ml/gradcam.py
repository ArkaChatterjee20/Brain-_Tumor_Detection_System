import tensorflow as tf
import numpy as np


def generate_gradcam(model, image, last_conv_layer_name):

    # Last convolution layer
    last_conv_layer = model.get_layer(last_conv_layer_name)

    # Model from input to last conv layer
    conv_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=last_conv_layer.output
    )

    # Model from last conv layer to output
    classifier_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])

    x = classifier_input

    start = False

    for layer in model.layers:
        if layer.name == last_conv_layer_name:
            start = True
            continue

        if start:
            x = layer(x)

    classifier_model = tf.keras.Model(
        classifier_input,
        x
    )

    with tf.GradientTape() as tape:

        conv_output = conv_model(image, training=False)

        tape.watch(conv_output)

        predictions = classifier_model(conv_output, training=False)

        pred_index = tf.argmax(predictions[0])

        loss = predictions[:, pred_index]

    grads = tape.gradient(loss, conv_output)

    print("grads:", grads)

    if grads is None:
        raise ValueError("Gradients are None")

    pooled_grads = tf.reduce_mean(
        grads,
        axis=(0, 1, 2)
    )

    conv_output = conv_output[0]

    heatmap = tf.reduce_sum(
        conv_output * pooled_grads,
        axis=-1
    )

    heatmap = tf.maximum(heatmap, 0)

    heatmap /= (
        tf.reduce_max(heatmap) + 1e-8
    )

    return heatmap.numpy()