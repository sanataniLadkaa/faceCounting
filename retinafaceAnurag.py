# Import necessary libraries
import os
import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision.transforms import functional as F
from retinaface import RetinaFace
import matplotlib.pyplot as plt

# Load the image
image_path = 'WhatsApp Image 2024-09-22 at 15.51.11_a540d0f4.jpg'  # Change this to your image path
orgimg = Image.open(image_path).convert("RGB")

# Use RetinaFace to detect faces
resp = RetinaFace.detect_faces(image_path)

# Create a draw object
draw = ImageDraw.Draw(orgimg)

# Loop through detected faces (in this case, we have one face)
for face_key, face_data in resp.items():
    # Get the facial area
    facial_area = face_data['facial_area']  # [x1, y1, x2, y2]
    draw.rectangle(facial_area, outline="red", width=2)  # Draw a red rectangle

    # Draw landmarks
    landmarks = face_data['landmarks']
    for landmark in landmarks.values():
        draw.ellipse((landmark[0] - 5, landmark[1] - 5, landmark[0] + 5, landmark[1] + 5), fill="blue")

# Save the modified image with bounding boxes and landmarks
output_image_path = 'detected_faces_image.png'  # Change this to your desired output path
orgimg.save(output_image_path)

# Display the modified image using Matplotlib
plt.figure(figsize=(10, 10))
plt.imshow(orgimg)
plt.axis('off')  # Hide axes
plt.show()

print(f"Detected image saved as: {output_image_path}")