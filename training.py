import os
import tensorflow as tf
from tensorflow.keras.models import model_from_json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# ----------------- Paths -----------------
DATA_DIR = "./data/beak_crops"
MODEL_JSON = "./model_architecture.json"
MODEL_WEIGHTS = "./model_weights.h5"

IMG_SIZE = (64, 64)
BATCH_SIZE = 32
VAL_SPLIT = 0.2
TEST_SPLIT = 0.2
EPOCHS = 100

CLASS_NAMES_FILE = "./data/beaks.txt"

class_names = []
with open(CLASS_NAMES_FILE, "r") as f:
    for line in f:
        _, name = line.strip().split(maxsplit=1)
        class_names.append(name)

def normalize_images(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

# Load Model from JSON
print("✔ Loading model from JSON...")
with open(MODEL_JSON, "r") as f:
    model = model_from_json(f.read())

# Load Model Weights
# if os.path.exists(MODEL_WEIGHTS):
#     model.load_weights(MODEL_WEIGHTS)
#     print("✔ Loaded weights")

# Compile Model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
print("✔ Model compiled")

# Load Full Dataset
print("✔ Loading full dataset from folders...")
full_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    image_size=IMG_SIZE,
    color_mode="grayscale",
    label_mode="int",
    shuffle=True,
    # seed=123,
    batch_size=None  # IMPORTANT
)

full_ds = full_ds.map(normalize_images)

total = tf.data.experimental.cardinality(full_ds).numpy()
train_size = int(0.6 * total)
val_size = int(0.2 * total)

train_ds = full_ds.take(train_size).batch(BATCH_SIZE)
val_ds = full_ds.skip(train_size).take(val_size).batch(BATCH_SIZE)
test_ds = full_ds.skip(train_size + val_size).batch(BATCH_SIZE)

# Augmentation
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.05),
    tf.keras.layers.RandomZoom(0.05),
])

train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))

print(f"✔ Train batches: {len(train_ds)}")
print(f"✔ Validation batches: {len(val_ds)}")
print(f"✔ Test batches: {len(test_ds)}")

# Train Model
print("✔ Starting training...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

# Save Trained Weights
# print("✔ Saving updated weights...")
# model.save_weights("model.weights.h5")

# Evaluate
print("✔ Evaluating model on test set...")
loss, acc = model.evaluate(test_ds)
print(f"\nFINAL TEST ACCURACY: {acc:.4f}")
print(f"FINAL TEST LOSS: {loss:.4f}")

y_true = []
y_pred = []

for images, labels in test_ds:
    preds = model.predict(images)
    preds_labels = np.argmax(preds, axis=1)

    y_true.extend(labels.numpy())
    y_pred.extend(preds_labels)

y_true = np.array(y_true)
y_pred = np.array(y_pred)



# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

fig, ax = plt.subplots(figsize=(14, 14))
disp.plot(
    ax=ax,
    cmap=plt.cm.Blues,
    colorbar=True,
    xticks_rotation=45  # diagonal labels
)

plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()


# Plot training & validation loss
plt.figure(figsize=(8, 6))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()