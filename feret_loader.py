import cv2
import os
import numpy as np
import face_alignment
from skimage import io
import torch
import bz2
import time
import json
from sklearn.model_selection import train_test_split

# Load pre-trained Caffe model for face detection
net = cv2.dnn.readNetFromCaffe("ssd/deploy.prototxt.txt", "ssd/res10_300x300_ssd_iter_140000.caffemodel")
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device='cuda' if torch.cuda.is_available() else 'cpu')

# Define FERET dataset loader
def load_feret_data(base_dir, image_size=(231, 314)):
    images = []
    labels = []
    label_map = {}
    label_id = 0

    for dvd_folder in ['dvd1', 'dvd2']:
        dvd_path = os.path.join(base_dir, dvd_folder, 'data', 'images')
        if not os.path.exists(dvd_path):
            continue

        for subject_folder in os.listdir(dvd_path):
            subject_path = os.path.join(dvd_path, subject_folder)
            if not os.path.isdir(subject_path):
                continue

            for file in os.listdir(subject_path):
                if file.endswith('.bz2'):
                    try:
                        compressed_file_path = os.path.join(subject_path, file)
                        with bz2.BZ2File(compressed_file_path, 'rb') as f:
                            decompressed_data = f.read()
                        img_array = np.frombuffer(decompressed_data, dtype=np.uint8)
                        img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
                        if img is not None:
                            img = cv2.resize(img, image_size)
                            images.append(img)
                            if subject_folder not in label_map:
                                label_map[subject_folder] = label_id
                                label_id += 1
                            labels.append(label_map[subject_folder])
                    except Exception as e:
                        print(f"Error processing file {compressed_file_path}: {e}")

    return np.array(images), np.array(labels), label_map

# Train LBPH face recognizer
def train_classifier(faces, faceID):
    face_recognizer = cv2.face.LBPHFaceRecognizer_create(
        radius=1,
        neighbors=7,
        grid_x=7,
        grid_y=7
    )
    face_recognizer.train(faces, np.array(faceID))
    return face_recognizer

# Apply CLAHE to an image
def apply_clahe(image):
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    return clahe.apply(image)

# Evaluate the recognizer on the test dataset
def evaluate_feret_recognition(X_test, y_test, face_recognizer):
    times = []
    y_pred = []

    for img, true_label in zip(X_test, y_test):
        img = apply_clahe(img)
        start = time.time()

        # Detect faces (mock detection as there's only one face per image here)
        faces = detect_faces_dnn(img)

        if len(faces) == 0:
            y_pred.append(-1)  # Append invalid label
            continue

        x_start, y_start, x_end, y_end = faces[0]
        roi_gray = img[y_start:y_end, x_start:x_end]

        if roi_gray is None or roi_gray.size == 0:
            y_pred.append(-1)
            continue

        label, confidence = face_recognizer.predict(roi_gray)
        end = time.time()
        times.append(end - start)

        y_pred.append(label if confidence < 50 else -1)

    avg_time = np.mean(times)
    accuracy = np.sum(np.array(y_pred) == y_test) / len(y_test)

    evaluation = {
        "Average Recognition Time (sec)": avg_time,
        "Accuracy": accuracy
    }
    print("FERET Evaluation Results:", evaluation)
    return evaluation

# Detect faces using DNN
def detect_faces_dnn(image):
    h, w = image.shape[:2]

    # Convert grayscale image to 3-channel BGR
    if len(image.shape) == 2:  # Check if it's grayscale
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    blob = cv2.dnn.blobFromImage(cv2.resize(image, (230, 238)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.7:
            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            faces.append(box.astype("int"))
    return faces


# Main script
if __name__ == "__main__":
    base_dir = "/content/colorferet"  # Path to FERET dataset
    image_size = (231, 314)

    # Load FERET dataset
    images, labels, label_map = load_feret_data(base_dir, image_size)
    print(f"Loaded {len(images)} images with {len(label_map)} unique labels.")

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

    # Train the LBPH classifier
    print("Training the LBPH face recognizer...")
    face_recognizer = train_classifier(X_train, y_train)

    # Save the trained model
    face_recognizer.save("models/trained_on_feret.yml")
    print("Model saved as models/trained_on_feret.yml")

    # Evaluate the model on the test set
    evaluation_results = evaluate_feret_recognition(X_test, y_test, face_recognizer)

    # Save the evaluation results
    with open("feret_evaluation_results.json", "w") as f:
        json.dump(evaluation_results, f, indent=4)
    print("Evaluation results saved to feret_evaluation_results.json")
