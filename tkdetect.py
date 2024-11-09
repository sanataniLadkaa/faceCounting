import os
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageTk
from tkinter import filedialog, Label, Button, Tk
from retinaface import RetinaFace
import matplotlib.pyplot as plt

# Function to detect faces and display the processed image
def detect_faces_and_display(image_path):
    # Load the image
    orgimg = Image.open(image_path).convert("RGB")
    
    # Use RetinaFace to detect faces
    resp = RetinaFace.detect_faces(image_path)
    
    # Create a draw object
    draw = ImageDraw.Draw(orgimg)
    
    # Loop through detected faces
    for face_key, face_data in resp.items():
        # Get the facial area and draw a rectangle
        facial_area = face_data['facial_area']  # [x1, y1, x2, y2]
        draw.rectangle(facial_area, outline="red", width=2)
        
        # Draw landmarks
        landmarks = face_data['landmarks']
        for landmark in landmarks.values():
            draw.ellipse((landmark[0] - 5, landmark[1] - 5, landmark[0] + 5, landmark[1] + 5), fill="blue")
    
    # Save the modified image
    output_image_path = 'detected_faces_image.png'
    orgimg.save(output_image_path)
    
    # Display the processed image in the GUI
    processed_image = ImageTk.PhotoImage(orgimg)
    processed_image_label.config(image=processed_image)
    processed_image_label.image = processed_image  # Keep a reference to avoid garbage collection

# Function to open an image file dialog
def open_image():
    image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if image_path:
        # Display the original image
        original_image = Image.open(image_path)
        original_image.thumbnail((300, 300))
        tk_image = ImageTk.PhotoImage(original_image)
        original_image_label.config(image=tk_image)
        original_image_label.image = tk_image  # Keep a reference to avoid garbage collection
        
        # Detect faces and display processed image
        detect_faces_and_display(image_path)

# Set up the Tkinter window
root = Tk()
root.title("Face Detection UI")

# Button to browse for an image
browse_button = Button(root, text="Browse Image", command=open_image)
browse_button.pack(pady=10)

# Label to display the original image
original_image_label = Label(root)
original_image_label.pack(pady=10)

# Label to display the processed image with face detection
processed_image_label = Label(root)
processed_image_label.pack(pady=10)

# Run the Tkinter event loop
root.mainloop()
