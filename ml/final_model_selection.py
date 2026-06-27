import os
import shutil
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ==========================================
# CONFIGURATION
# ==========================================

BEST_MODEL_PATH = "models/checkpoints/CNN.keras"

FINAL_MODEL_PATH = "models/final/brain_tumor_classifier.keras"

TEST_DIR = "data/split/test"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

os.makedirs("models/final", exist_ok=True)

# ==========================================
# LOAD TEST DATA
# ==========================================

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

# ==========================================
# LOAD CNN MODEL
# ==========================================

print("\nLoading Final CNN Model...")

model = load_model(BEST_MODEL_PATH)

print("CNN Model Loaded Successfully!")

# ==========================================
# FINAL EVALUATION
# ==========================================

print("\nEvaluating Final Model...")

loss, accuracy = model.evaluate(
    test_generator,
    verbose=1
)

print("\n" + "=" * 50)
print("FINAL MODEL PERFORMANCE")
print("=" * 50)

print(f"Test Loss     : {loss:.4f}")
print(f"Test Accuracy : {accuracy * 100:.2f}%")

# ==========================================
# SAVE FINAL MODEL
# ==========================================

print("\nSaving Final Deployment Model...")

model.save(FINAL_MODEL_PATH)

print("Final model saved!")

# ==========================================
# SAVE CLASS LABELS
# ==========================================

class_labels = list(
    test_generator.class_indices.keys()
)

with open(
    "models/final/class_labels.txt",
    "w"
) as f:

    for label in class_labels:
        f.write(label + "\n")

print("Class labels saved!")

# ==========================================
# SAVE MODEL INFO
# ==========================================

with open(
    "models/final/model_info.txt",
    "w"
) as f:

    f.write("Brain Tumor Detection System\n")
    f.write("=============================\n")
    f.write("Selected Model : CNN\n")
    f.write(f"Input Size     : {IMG_SIZE}\n")
    f.write(f"Classes        : {class_labels}\n")
    f.write(f"Test Accuracy  : {accuracy*100:.2f}%\n")
    f.write(f"Test Loss      : {loss:.4f}\n")

print("Model metadata saved!")

print("\nDeployment Artifacts Created Successfully!")

print("\nSaved Files:")
print("--------------------------------")
print(FINAL_MODEL_PATH)
print("models/final/class_labels.txt")
print("models/final/model_info.txt")