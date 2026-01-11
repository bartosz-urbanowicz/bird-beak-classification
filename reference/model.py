from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import tensorflow as tf

model = Sequential([
    Dense(4, activation='relu'),
    Dense(2, activation='relu'),
    Dense(2, activation='softmax'),
])

model.compile(
    # optimizer=tf.keras.optimizers.SGD(learning_rate=0.1),
    optimizer=tf.keras.optimizers.RMSprop(learning_rate=1.0),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.build(input_shape=(None, 4))

model.save_weights("model.weights.h5")

# print(model.predict(np.array([[0.1, 0.2, 0.3, 0.4]])))

x = tf.constant([[0.1, 0.2, 0.3, 0.4]], dtype=tf.float32)
y_true = tf.constant([[0, 1]], dtype=tf.float32)

with tf.GradientTape() as tape:
    y_pred = model(x, training=True)
    loss = tf.keras.losses.MeanSquaredError()(y_true, y_pred)

# Compute gradients of loss wrt weights
gradients = tape.gradient(loss, model.trainable_variables)

# Print gradient layer-by-layer
# for i, (var, grad) in enumerate(zip(model.trainable_variables, gradients)):
#     print(f"\nVariable {i}: {var.name}")
#     print("Shape:", grad.shape)
#     print(grad.numpy())


model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

for var in model.trainable_variables:
    print(f"\nVariable: {var.name}")
    print(var.numpy())

print(model.predict(np.array([[0.1, 0.2, 0.3, 0.4]])))
