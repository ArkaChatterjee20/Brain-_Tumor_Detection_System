import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout
)

MODEL_PATH = "models/final/brain_tumor_classifier.keras"


def rebuild_cnn():

    model = Sequential([

        Input(
            shape=(224, 224, 3),
            name="input_layer"
        ),

        Conv2D(
            32,
            (3, 3),
            activation="relu",
            name="conv2d"
        ),

        MaxPooling2D(
            (2, 2),
            name="max_pooling2d"
        ),

        Conv2D(
            64,
            (3, 3),
            activation="relu",
            name="conv2d_1"
        ),

        MaxPooling2D(
            (2, 2),
            name="max_pooling2d_1"
        ),

        Conv2D(
            128,
            (3, 3),
            activation="relu",
            name="conv2d_2"
        ),

        MaxPooling2D(
            (2, 2),
            name="max_pooling2d_2"
        ),

        Flatten(
            name="flatten"
        ),

        Dense(
            256,
            activation="relu",
            name="dense"
        ),

        Dropout(
            0.5,
            name="dropout"
        ),

        Dense(
            4,
            activation="softmax",
            name="dense_1"
        )
    ])

    print("Loading saved model...")

    saved_model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    model.set_weights(
        saved_model.get_weights()
    )

    print(
        "CNN rebuilt successfully."
    )

    return model