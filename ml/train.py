import os
import pandas as pd
import tensorflow as tf

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    Flatten,
    Conv2D,
    MaxPooling2D,
    GlobalAveragePooling2D
)

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint
)

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.applications import (
    MobileNetV2,
    ResNet50,
    EfficientNetB0
)

# ==================================================
# CONFIGURATION
# ==================================================

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 25

TRAIN_DIR = "data/split/train"
VAL_DIR = "data/split/val"

os.makedirs("models/checkpoints", exist_ok=True)
os.makedirs("results", exist_ok=True)

# ==================================================
# DATA GENERATOR
# ==================================================

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    zoom_range=0.15,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
)

val_datagen = ImageDataGenerator(
    rescale=1./255
)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

NUM_CLASSES = train_generator.num_classes

print("\nClasses Found:")
print(train_generator.class_indices)

# ==================================================
# CUSTOM CNN
# ==================================================

def build_cnn():

    model = Sequential([

        Conv2D(
            32,
            (3,3),
            activation="relu",
            input_shape=(224,224,3)
        ),

        MaxPooling2D(2,2),

        Conv2D(
            64,
            (3,3),
            activation="relu"
        ),

        MaxPooling2D(2,2),

        Conv2D(
            128,
            (3,3),
            activation="relu"
        ),

        MaxPooling2D(2,2),

        Flatten(),

        Dense(
            256,
            activation="relu"
        ),

        Dropout(0.5),

        Dense(
            NUM_CLASSES,
            activation="softmax"
        )
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model

# ==================================================
# TRANSFER LEARNING MODELS
# ==================================================

def build_transfer_model(base_model):

    base_model.trainable = False

    x = GlobalAveragePooling2D()(base_model.output)

    x = Dropout(0.3)(x)

    outputs = Dense(
        NUM_CLASSES,
        activation="softmax"
    )(x)

    model = Model(
        inputs=base_model.input,
        outputs=outputs
    )

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model

# ==================================================
# MODELS TO TRAIN
# ==================================================

models = {
    "CNN": build_cnn(),

    "MobileNetV2": build_transfer_model(
        MobileNetV2(
            weights="imagenet",
            include_top=False,
            input_shape=(224,224,3)
        )
    ),

    "ResNet50": build_transfer_model(
        ResNet50(
            weights="imagenet",
            include_top=False,
            input_shape=(224,224,3)
        )
    ),

    "EfficientNetB0": build_transfer_model(
        EfficientNetB0(
            weights="imagenet",
            include_top=False,
            input_shape=(224,224,3)
        )
    )
}

# ==================================================
# TRAINING LOOP
# ==================================================

comparison_results = []

for model_name, model in models.items():

    print("\n" + "="*60)
    print(f"TRAINING {model_name}")
    print("="*60)

    checkpoint = ModelCheckpoint(
        filepath=f"models/checkpoints/{model_name}.keras",
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    )

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=EPOCHS,
        callbacks=[
            checkpoint,
            early_stopping
        ]
    )

    best_accuracy = max(
        history.history["val_accuracy"]
    )

    comparison_results.append([
        model_name,
        round(best_accuracy * 100, 2)
    ])

# ==================================================
# SAVE COMPARISON
# ==================================================

comparison_df = pd.DataFrame(
    comparison_results,
    columns=[
        "Model",
        "Validation Accuracy (%)"
    ]
)

comparison_df.to_csv(
    "results/comparison.csv",
    index=False
)

print("\nMODEL COMPARISON")
print(comparison_df)