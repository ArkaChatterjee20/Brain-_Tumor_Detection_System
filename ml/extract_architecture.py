import tensorflow as tf

MODEL_PATH = "models/final/brain_tumor_classifier.keras"

model = tf.keras.models.load_model(MODEL_PATH)

print("\nMODEL SUMMARY")
print("=" * 60)

model.summary()

print("\nCONFIG")
print("=" * 60)

for i, layer in enumerate(model.layers):
    print(f"\nLayer {i}")
    print("Name :", layer.name)
    print("Type :", type(layer).__name__)

    try:
        print("Config :", layer.get_config())
    except:
        print("Config not available")