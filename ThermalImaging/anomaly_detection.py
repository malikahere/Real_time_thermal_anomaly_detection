import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import base64
from io import BytesIO

# Constants
IMG_SIZE = 224  # Size used during training
ANOMALY_LABEL = 1  # Index of the "anomaly" class
MODEL_PATH = r"D:\Python_projects\Thermal_anomaly_detection\thermal_anomaly_detector.h5"  # Update with the actual path

# Load the trained CNN model
model = load_model(MODEL_PATH)

def correct_base64_padding(base64_string):
    """
    Corrects the base64 string padding if necessary.
    """
    padding = len(base64_string) % 4
    if padding:
        base64_string += '=' * (4 - padding)
    return base64_string

def preprocess_image_from_buffer(image_buffer):
    """
    Preprocess image data (from buffer) to match the model's input requirements.
    """
    try:
        # Ensure base64 string has correct padding
        image_buffer = correct_base64_padding(image_buffer)
        
        # Decode base64 string to bytes
        img_data = base64.b64decode(image_buffer)
        
        # Convert bytes to a numpy array
        img_array = np.frombuffer(img_data, dtype=np.uint8)
        
        # Decode the image
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("Unable to decode the image buffer.")
        
        # Resize to match model's input size
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        
        # Normalize pixel values to [0, 1]
        img = img / 255.0
        
        # Expand dimensions to match the input shape for the model
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        return img
    except Exception as e:
        print(f"Error preprocessing image from buffer: {e}")
        return None

def check_anomaly_from_buffer(image_buffer):
    """
    Check if the given image (from buffer) is anomalous.
    """
    img = preprocess_image_from_buffer(image_buffer)
    if img is None:
        return "Error: Image preprocessing failed."
    
    prediction = model.predict(img)  # Get the prediction
    predicted_class = np.argmax(prediction, axis=1)[0]  # Get class index
    
    if predicted_class == ANOMALY_LABEL:
        return "Anomaly detected."
    else:
        return "Normal image."

# Example of image buffer received from Node.js (simulated)
# Replace this with the actual base64 image string received from your application
example_image_buffer = 'iVBORw0KGgoAAAANSUhEUgAAACAAAAAYCAYAAACbU/80AAAABmJLR0QA/wD/AP+gvaeTAAAEDElEQVRIiV2WwW4kNwxEHyl1j2dsJ8B6D0EO+f9/2A/KJQhgr9ee6ZaoHEiO2jFg9LRaEovFYkny468fwwZUHXQTlsUoxehd2Xbl5w0AXm8wBty6P5vBAGz4O0BRUJljVeecU/E1+ffHkz/r1kEAwYObCbetYhFsN99MgM2gCFy7fxNAImAGPpUAcACXgIv6WDf4tcPjArVqfhzsu7IbfOw+1gxuzZHmhteWgD244KBqZF8UMNi6z12Kj4v4WFX//Rkxag9abAh9wK/NJ249x/2Z1Au+0IYHvoPQeMdZ2yPj/Ls2ONfJiiYgEU/v1iLjQ+CkN9+zXONQ76o+dq7+/efm+6g41UPAAtjWA6hOPVQLsbx9BtVRr637JgPfKIPnnAGowhoUt8g657XxtUw9EkpwKcoqkV0f/szfGtFSCzYmgBKBL4v/Xxv88zFLBHM9EbwEIw/V3zOeZn1uzSf0UGkGbQZ7n1mnios47c1cN1kyFd+DEGrWO9lsNjujKNTdHMBanL4MmjXvEXiE2jXEthZ43+Dfz1A5k6XMfgSAKpPVY8sWCR9IgyGEIgItnmYzCxKETMGOrG/UO1nKAMfv2Y53hgbUt9tBsUFRZkPWVOYG58W/vV6j1WLT7JTsiqS8/K+TTiX8Ibto71N4d4s9CDHRVvVFLUqWXZJAc06uPVKebMDUT7Z4VfFMUt1pmZXZMul8NuD96vHkkGHWezcHeSxVBj/VEDNTrBAaSDfUcrDTyOhjn96+7bOeGSSFpoeAYzjiDJatnGUYAyi+T12Lo/vt5Bs8LtN8VODvd/ft920GTbWnoglKR7TiEoeO2VyTZ85aZiuLhA9o9PT3J7u7VDrcWibybNF8h3laJuBjGx/FeW2zXDYOVv5y8SDfft9ggGpxSrvX4WkVVLwUAyCQ16B41XkfSK3cxRllzJJmoiKzfPWhwuOpM0wwm/7ZTb4ctVlrERfTUjxrk+lq2UkQ9osfRjn/sngb52XlvEB9+XalVGOY0JoyAkSpRm/KI4VThe+XqYWleJ3TfI7dknooB6Gqwp/P8LQ6sFuD55ODqH33oGMIqsbeCyLDx0zutV+LLzzeD9LACHbuN66jLwQbp+p7fMYlJc+DWtdOqYZ1RXWgdaftCk0pD43xubDWwctQztW1kK2YwLp5MPjaDccOADev55PRTeepud8qZkpvyhhC2/QLlWNA74oNR34fiyBp4fm/9+meRea98nNPjciXW1XVMhAGl6cNM0GLoWVgTbAhtF0RHahWbk24LLBpCDAM58jEEjej9IKHoP68eHuXaojA03lnWTv19W3lVBceLjvI8DKUgTWld2Hf3LJ6ly93uT0umGnjo4UOxqS7qJ+q3Vy8fuYUqsJ6amzXyn+o2PmpQexydAAAAABJRU5ErkJggg=='

# Check for anomaly from buffer
result = check_anomaly_from_buffer(example_image_buffer)
print(f"Result: {result}")

import time
latencies = []
for _ in range(10):  # Run 100 inferences
    start_time = time.time()
    model.predict(input_data)
    end_time = time.time()
    latencies.append(end_time - start_time)

average_latency = sum(latencies) / len(latencies)
print(f"Average Latency: {average_latency:.6f} seconds")



import numpy as np
import matplotlib.pyplot as plt

# Example batch sizes and the number of runs for averaging latency
batch_sizes = [1, 2, 4, 8, 16, 32]
num_runs = 1  # Number of runs for each batch size to calculate average latency

latencies = []

# Measure latency for each batch size
for batch_size in batch_sizes:
    run_latencies = []  # Store latencies for multiple runs
    
    for _ in range(num_runs):
        input_data = np.random.rand(batch_size, 224, 224, 3)  # Random input for the model
        
        # Start timing
        start_time = time.time()

        # Model inference (replace 'model' with your actual model)
        output = model.predict(input_data)

        # End timing
        end_time = time.time()

        # Calculate latency for the current run
        latency = end_time - start_time
        run_latencies.append(latency)
    
    # Calculate the average latency for the current batch size
    avg_latency = np.mean(run_latencies)
    latencies.append(avg_latency)
    

# Plotting the average latency graph
plt.figure(figsize=(8, 6))
plt.plot(batch_sizes, latencies, marker='o', linestyle='-', color='b')
plt.title('Average Model Latency vs Batch Size')
plt.xlabel('Batch Size')
plt.ylabel('Average Latency (seconds)')
plt.grid(True)
plt.show()

