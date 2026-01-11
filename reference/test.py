import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist

# Load MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize and reshape
x_train = x_train.reshape(-1, 784).astype("float32") / 255.0
x_test = x_test.reshape(-1, 784).astype("float32") / 255.0

# One-hot encode labels
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Use only 4000 training samples
x_train = x_train[:4000]
y_train = y_train[:4000]

# Build model
model = models.Sequential([
    layers.Input(shape=(784,)),
    layers.Dense(784, activation="sigmoid"),
    layers.Dense(64, activation="sigmoid"),
    layers.Dense(10, activation="sigmoid")
])

# Compile with plain SGD (no momentum)
model.compile(
    optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
    # optimizer=tf.keras.optimizers.SGD(learning_rate=1.0),
    # loss="categorical_crossentropy",
    loss="mse",
    metrics=["accuracy"]
)

# Train
history = model.fit(
    x_train,
    y_train,
    batch_size=32,
    epochs=10,
    validation_split=0.2,
    verbose=1,
    shuffle=True
)

# Evaluate
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print("\nTest accuracy:", test_accuracy)
