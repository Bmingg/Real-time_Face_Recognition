import cv2
import os
import numpy as np
from collections import Counter

def load_trained_model(model_path='models/trained_on_test.yml'):
    if os.path.exists(model_path):
        face_recognizer = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=7, grid_x=7, grid_y=7, threshold=45)
        face_recognizer.read(model_path)
        return face_recognizer
    return None

def labels_for_training_data(directory):
    faces = []
    faceID = []

    for path, subdirnames, filenames in os.walk(directory):
        for filename in filenames:
            if filename.startswith("."):
                continue
            id = os.path.basename(path)
            img_path = os.path.join(path, filename)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (230, 238))
            img = cv2.GaussianBlur(img, (5,5),0)
            if img is None:
                continue
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces.append(gray_img)
            faceID.append(int(id))

    return faces, faceID

def update_classifier(face_recognizer, faces, faceID):
    face_recognizer.update(faces, np.array(faceID))
    return face_recognizer

def train_classifier(faces, faceID):
    face_recognizer = cv2.face.LBPHFaceRecognizer_create(
        radius=1,
        neighbors=7,
        grid_x=7,
        grid_y=7,
        threshold=60)
    face_recognizer.train(faces, np.array(faceID))
    return face_recognizer

folder = "new_augmented_dataset_crop_test"
faces, faceID = labels_for_training_data(folder)
face_recognizer = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=7, grid_x=7, grid_y=7, threshold=45)
face_recognizer = train_classifier(faces, faceID)
face_recognizer.save('models/model_test.yml')

