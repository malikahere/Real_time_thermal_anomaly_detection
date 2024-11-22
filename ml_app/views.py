# from django.shortcuts import render

# # Create your views here.
# import io
# import numpy as np
# from django.http import JsonResponse
# from PIL import Image
# from django.views.decorators.csrf import csrf_exempt
# import tensorflow as tf  # Replace with your model library

# # Dummy ML model (replace with your trained model)
# model.save("model.h5")
# MODEL = tf.keras.models.load_model("path/to/your/model")

# @csrf_exempt
# def process_image(request):
#     if request.method == 'POST':
#         try:
#             # Read the image buffer
#             image_file = request.FILES.get('image')
#             if not image_file:
#                 return JsonResponse({'error': 'No image provided'}, status=400)

#             # Open the image using Pillow
#             image = Image.open(image_file)
#             image = image.resize((224, 224))  # Resize to model input size if required

#             # Convert to numpy array for model processing
#             image_array = np.array(image) / 255.0  # Normalize pixel values
#             image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

#             # Run the image through the model
#             prediction = MODEL.predict(image_array)
#             result = np.argmax(prediction, axis=1)[0]  # Example: classification result

#             # Return the prediction result
#             return JsonResponse({'result': int(result)}, status=200)

#         except Exception as e:
#             return JsonResponse({'error': str(e)}, status=500)

#     return JsonResponse({'error': 'Invalid request method'}, status=405)

