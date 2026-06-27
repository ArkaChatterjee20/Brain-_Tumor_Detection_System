import tensorflow as tf

print("=" * 50)
print("TensorFlow Information")
print("=" * 50)

print("TensorFlow version :", tf.__version__)

print("Eager execution enabled :", tf.executing_eagerly())

print("\nAvailable Physical Devices:")
for device in tf.config.list_physical_devices():
    print(device)