import tensorflow as tf
from ml.rebuild_cnn import rebuild_cnn

model = rebuild_cnn()

# Dummy input
image = tf.random.normal((1, 224, 224, 3))

with tf.GradientTape() as tape:
    tape.watch(image)

    preds = model(image, training=False)

    pred_index = tf.argmax(preds[0])

    loss = preds[:, pred_index]

grads = tape.gradient(loss, image)

print("Predictions:")
print(preds)

print("\nGradient:")
print(grads)
