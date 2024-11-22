# %%

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical # type: ignore

# Define paths
dataset_dir = r"D:\Python_projects\Thermal_anomaly_detection\ThermalImaging\images" # Replace with the path to your dataset
categories = ["normal", "abnormal"]  # Two categories: normal and anomaly

# Image size
IMG_SIZE = 224

# Load and preprocess images
def load_images(dataset_dir, categories):
    images = []
    labels = []
    for idx, category in enumerate(categories):
        folder_path = os.path.join(dataset_dir, category)
        for img_name in os.listdir(folder_path):
            try:
                img_path = os.path.join(folder_path, img_name)
                img = cv2.imread(img_path)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))  # Resize to uniform size
                img = img / 255.0  # Normalize to range [0, 1]
                images.append(img)
                labels.append(idx)  # Use index as label
            except Exception as e:
                print(f"Error loading image {img_name}: {e}")
    return np.array(images), np.array(labels)

X, y = load_images(dataset_dir, categories)

# Split data into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert labels to one-hot encoding
y_train = to_categorical(y_train, num_classes=len(categories))
y_test = to_categorical(y_test, num_classes=len(categories))



# %%
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Build CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(categories), activation='softmax')  # Output layer with number of categories
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

# %%
history = model.fit(
    X_train, y_train,
    epochs=10,  # Adjust the number of epochs as needed
    batch_size=16,  # Adjust batch size based on system capacity
    validation_data=(X_test, y_test)
)


test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")


plt.figure(figsize=(12, 4))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Model Accuracy')

# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Model Loss')

plt.show()


# Predict
predictions = model.predict(X_test)
predicted_labels = np.argmax(predictions, axis=1)
true_labels = np.argmax(y_test, axis=1)

# Check for anomalies
for idx, pred in enumerate(predicted_labels):
    if pred == categories.index("abnormal"):
        print(f"Anomaly detected in image {idx}")


# Save model
model.save("thermal_anomaly_detector.h5")

# Load model
from tensorflow.keras.models import load_model
model = load_model("thermal_anomaly_detector.h5")


