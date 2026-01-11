import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import model_from_json

# ----------------- Paths -----------------
DATA_DIR = "./data/beak_crops"
MODEL_JSON = "./model_architecture.json"
MODEL_WEIGHTS = "./model_weights.h5"
CLASS_NAMES_FILE = "./data/beaks.txt"

IMG_SIZE = (64, 64)
BATCH_SIZE = 32
NUM_TO_PLOT = 50

# ----------------- Load Class Names -----------------
# Build a mapping from directory name -> human-readable class name
class_name_map = {}
with open(CLASS_NAMES_FILE, "r") as f:
    for line in f:
        dir_name, class_name = line.strip().split(maxsplit=1)
        class_name_map[dir_name] = class_name

# ----------------- Load Model -----------------
print("✔ Loading model from JSON...")
with open(MODEL_JSON, "r") as f:
    model = model_from_json(f.read())

if os.path.exists(MODEL_WEIGHTS):
    model.load_weights(MODEL_WEIGHTS)
    print("✔ Loaded weights")

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
print("✔ Model compiled")

# ----------------- Load Test Dataset -----------------
print("✔ Loading test dataset...")
test_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    image_size=IMG_SIZE,
    color_mode="grayscale",
    label_mode="int",
    shuffle=False,
    batch_size=BATCH_SIZE
)

# Map integer labels to human-readable class names
dir_names = test_ds.class_names  # directory names as used by image_dataset_from_directory
int_to_class_name = {i: class_name_map[dir_name] for i, dir_name in enumerate(dir_names)}

# ----------------- Normalize Dataset -----------------
test_ds = test_ds.map(lambda x, y: (tf.cast(x, tf.float32)/255.0, y))

# ----------------- Flatten Dataset -----------------
all_images = []
all_labels = []
for images, labels in test_ds:
    all_images.append(images.numpy())
    all_labels.append(labels.numpy())
all_images = np.concatenate(all_images, axis=0)
all_labels = np.concatenate(all_labels, axis=0)

print(f"Total test images: {all_images.shape[0]}")

# ----------------- Predict -----------------
preds = model.predict(all_images, batch_size=BATCH_SIZE)
y_pred = np.argmax(preds, axis=1)

# ----------------- Find Misclassified -----------------
misclassified_indices = np.where(all_labels != y_pred)[0]
print(f"Total misclassified images: {len(misclassified_indices)}")

# Shuffle misclassified indices
np.random.shuffle(misclassified_indices)

# ----------------- Plot First N Misclassified -----------------
num_to_plot = min(NUM_TO_PLOT, len(misclassified_indices))

for i, idx in enumerate(misclassified_indices[:num_to_plot]):
    image = all_images[idx].squeeze()
    true_label = int_to_class_name[all_labels[idx]]
    pred_label = int_to_class_name[y_pred[idx]]

    plt.figure(figsize=(4, 4))
    plt.imshow(image, cmap="gray")
    plt.title(f"True: {true_label}\nPredicted: {pred_label}", fontsize=12)
    plt.axis("off")
    plt.show()
