import cv2
import os
import numpy as np

def train_model():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    faces = []
    labels = []

    # Load images and labels
    for name in os.listdir('dataset'):
        label = int(name.split('_')[1])
        for image_file in os.listdir(f'dataset/{name}'):
            image_path = f'dataset/{name}/{image_file}'
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            faces.append(image)
            labels.append(label)

    recognizer.train(faces, np.array(labels))
    recognizer.save('face_recognizer.yml')

# Assign label 1 to the captured person


# Train the model
train_model()

os.rename('dataset/User_1', 'dataset/Nanthan_Shetty_1')
