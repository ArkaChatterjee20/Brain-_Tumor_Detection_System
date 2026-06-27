import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_fscore_support
)

# =====================================================
# CONFIGURATION
# =====================================================

TEST_DIR = "data/split/test"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

MODEL_PATHS = {
    "CNN": "models/checkpoints/CNN.keras",
    "MobileNetV2": "models/checkpoints/MobileNetV2.keras",
    "ResNet50": "models/checkpoints/ResNet50.keras",
    "EfficientNetB0": "models/checkpoints/EfficientNetB0.keras"
}

os.makedirs("results/evaluation", exist_ok=True)

# =====================================================
# TEST DATA
# =====================================================

test_datagen = ImageDataGenerator(
    rescale=1./255
)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

class_names = list(test_generator.class_indices.keys())

# =====================================================
# EVALUATION
# =====================================================

evaluation_results = []

for model_name, model_path in MODEL_PATHS.items():

    print("\n" + "=" * 60)
    print(f"EVALUATING {model_name}")
    print("=" * 60)

    model = load_model(model_path)

    # -----------------------------
    # Evaluate Loss and Accuracy
    # -----------------------------
    loss, accuracy = model.evaluate(
        test_generator,
        verbose=0
    )

    # -----------------------------
    # Predictions
    # -----------------------------
    predictions = model.predict(
        test_generator,
        verbose=0
    )

    y_pred = np.argmax(predictions, axis=1)
    y_true = test_generator.classes

    # -----------------------------
    # Precision, Recall, F1
    # -----------------------------
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted"
    )

    # -----------------------------
    # Classification Report
    # -----------------------------
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names
    )

    print("\nClassification Report:")
    print(report)

    with open(
        f"results/evaluation/{model_name}_classification_report.txt",
        "w"
    ) as f:
        f.write(report)

    # -----------------------------
    # Confusion Matrix
    # -----------------------------
    cm = confusion_matrix(
        y_true,
        y_pred
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )

    fig, ax = plt.subplots(figsize=(8, 8))

    disp.plot(
        cmap="Blues",
        ax=ax,
        values_format="d"
    )

    plt.title(f"{model_name} Confusion Matrix")

    plt.savefig(
        f"results/evaluation/{model_name}_confusion_matrix.png"
    )

    plt.close()

    # -----------------------------
    # Store Results
    # -----------------------------
    evaluation_results.append([
        model_name,
        round(loss, 4),
        round(accuracy * 100, 2),
        round(precision * 100, 2),
        round(recall * 100, 2),
        round(f1 * 100, 2)
    ])

# =====================================================
# SAVE COMPARISON CSV
# =====================================================

results_df = pd.DataFrame(
    evaluation_results,
    columns=[
        "Model",
        "Test Loss",
        "Test Accuracy (%)",
        "Precision (%)",
        "Recall (%)",
        "F1 Score (%)"
    ]
)

results_df.to_csv(
    "results/evaluation/model_evaluation_summary.csv",
    index=False
)

print("\n" + "=" * 60)
print("FINAL MODEL EVALUATION SUMMARY")
print("=" * 60)

print(results_df)

print("\nSaved to:")
print("results/evaluation/model_evaluation_summary.csv")