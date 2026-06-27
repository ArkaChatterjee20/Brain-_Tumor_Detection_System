import tensorflow as tf

MODEL_PATH = "models/final/brain_tumor_classifier.keras"

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

print("=" * 60)
print("MODEL SUMMARY")
print("=" * 60)

model.summary()

print("\n" + "=" * 60)
print("CONVOLUTIONAL LAYERS")
print("=" * 60)

last_conv_layer = None

for layer in model.layers:
    if isinstance(layer, tf.keras.layers.Conv2D):
        print(layer.name)
        last_conv_layer = layer.name

print("\n" + "=" * 60)
print("LAST CONVOLUTIONAL LAYER")
print("=" * 60)

if last_conv_layer:
    print(last_conv_layer)
else:
    print("No Conv2D layer found.")