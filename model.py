from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D, AveragePooling2D, Flatten, Dense, Dropout
from tensorflow.keras import regularizers

IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64
IMAGE_CHANNELS = 1
NUM_CLASSES = 13

model = Sequential([
    Conv2D(8, (5,5), activation='relu', padding='same', input_shape=(IMAGE_WIDTH,IMAGE_HEIGHT,1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(16, (5,5), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    Dropout(0.5),
    Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    Dropout(0.5),
    Dense(NUM_CLASSES, activation='softmax')
])


model_json = model.to_json()
with open("model_architecture.json", "w") as f:
    f.write(model_json)