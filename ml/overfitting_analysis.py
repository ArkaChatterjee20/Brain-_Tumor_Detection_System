import os
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ======================================
# CONFIGURATION
# ======================================

TRAIN_DIR = "data/split/train"
VAL_DIR = "data/split/val"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

MODEL_PATHS = {
    "CNN": "models/checkpoints/CNN.keras",
    "MobileNetV2": "models/checkpoints/MobileNetV2.keras",
    "ResNet50": "models/checkpoints/ResNet50.keras",
    "EfficientNetB0": "models/checkpoints/EfficientNetB0.keras"
}

os.makedirs("results", exist_ok=True)

# ======================================
# DATA GENERATORS
# ======================================

datagen = ImageDataGenerator(rescale=1./255)

train_generator = datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

val_generator = datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

# ======================================
# ANALYSIS
# ======================================

results = []

for model_name, model_path in MODEL_PATHS.items():

    print("\n" + "=" * 60)
    print(f"ANALYZING {model_name}")
    print("=" * 60)

    model = load_model(model_path)

    # Evaluate on train dataset
    train_loss, train_acc = model.evaluate(
        train_generator,
        verbose=0
    )

    # Evaluate on validation dataset
    val_loss, val_acc = model.evaluate(
        val_generator,
        verbose=0
    )

    gap = train_acc - val_acc

    # Determine status
    if gap <= 0.05:
        status = "Good Fit"
    elif gap <= 0.10:
        status = "Mild Overfitting"
    else:
        status = "Severe Overfitting"

    print(f"Train Accuracy : {train_acc*100:.2f}%")
    print(f"Validation Accuracy : {val_acc*100:.2f}%")
    print(f"Accuracy Gap : {gap*100:.2f}%")
    print(f"Status : {status}")

    results.append([
        model_name,
        round(train_acc * 100, 2),
        round(val_acc * 100, 2),
        round(gap * 100, 2),
        status
    ])

# ======================================
# SAVE RESULTS
# ======================================

df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "Train Accuracy (%)",
        "Validation Accuracy (%)",
        "Accuracy Gap (%)",
        "Status"
    ]
)

df.to_csv(
    "results/overfitting_analysis.csv",
    index=False
)

print("\nAnalysis saved to:")
print("results/overfitting_analysis.csv")

print("\nFinal Summary:")
print(df)