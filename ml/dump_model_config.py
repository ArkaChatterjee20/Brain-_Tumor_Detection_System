import tensorflow as tf
import json

MODEL_PATH = "models/final/brain_tumor_classifier.keras"

model = tf.keras.models.load_model(MODEL_PATH)

config = model.get_config()

with open("model_config.json", "w") as f:
    json.dump(config, f, indent=4)

print("Configuration saved to model_config.json")