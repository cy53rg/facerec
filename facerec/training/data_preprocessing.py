import os
import cv2
import numpy as np

# Define the path to the dataset directory
dataset_dir = os.path.join(os.path.expanduser("~"), "Desktop", "facerec", "data_preprocessed_cropped_normalized")

# Define the path to the output directory
output_dir = os.path.join(os.path.expanduser("~"), "Desktop", "facerec", "preprocessed_data")

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Loop over all subdirectories in the dataset directory
for subdir in os.listdir(dataset_dir):
    subdir_path = os.path.join(dataset_dir, subdir)
    if not os.path.isdir(subdir_path):
        continue

    # Loop over all images in the current subdirectory
    for file_name in os.listdir(subdir_path):
        if not file_name.endswith(".jpg"):
            continue

        # Load the current image and convert it to grayscale
        img_path = os.path.join(subdir_path, file_name)
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply face detection using Haar cascades
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        # Crop the face from the image and resize it to a fixed size
        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (96, 96))

            # Save the preprocessed face image to the output directory
            output_path = os.path.join(output_dir, subdir, file_name)
            if not os.path.exists(os.path.join(output_dir, subdir)):
                os.makedirs(os.path.join(output_dir, subdir))
            cv2.imwrite(output_path, face)
