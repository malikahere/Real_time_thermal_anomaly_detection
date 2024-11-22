import base64
import numpy as np
from channels.generic.websocket import AsyncWebsocketConsumer
from django.core.files.base import ContentFile
from ThermalImaging.anomaly_detection import detect_anomaly
from io import BytesIO
from PIL import Image
import tensorflow as tf
from asgiref.sync import sync_to_async

class ThermalStreamConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.room_group_name = 'thermal_stream'
        # Accept the WebSocket connection
        await self.accept()

    async def receive(self, text_data):
        data = text_data  # Assuming `text_data` contains the base64 image string
        image_data = self.decode_base64_image(data['data'])  # Decode base64 image

        # Process the image using the anomaly detection model
        anomaly_result = await sync_to_async(detect_anomaly)(image_data)

        # Send the result back to Node.js
        await self.send(text_data={
            'type': 'anomaly_result',
            'result': anomaly_result
        })

    def decode_base64_image(self, data):
        """Decode base64 image string into image"""
        image_data = base64.b64decode(data)
        image = Image.open(BytesIO(image_data))
        return np.array(image)

    async def disconnect(self, close_code):
        pass
