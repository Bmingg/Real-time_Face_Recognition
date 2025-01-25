import os
import bz2
import cv2
import numpy as np
import face_alignment
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import torch
import time
import json
import random

# Load pre-trained Caffe model for face detection
net = cv2.dnn.readNetFromCaffe("ssd/deploy.prototxt.txt", "ssd/res10_300x300_ssd_iter_140000.caffemodel")
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device='cuda' if torch.cuda.is_available() else 'cpu')

# Define recognition threshold
RECOGNITION_THRESHOLD = 50

def load_feret_data_with_unknown(base_dir, image_size=(231, 314)):
    images = []
    labels = []
    label_map = {}
    label_id = 0
    test_unknown_images = []
    test_unknown_labels = []

    for dvd_folder in ['dvd1', 'dvd2']:
        dvd_path = os.path.join(base_dir, dvd_folder, 'data', 'images')
        if not os.path.exists(dvd_path):
            print(f"Directory not found: {dvd_path}")
            continue

        # Get all subject folders
        subject_folders = [f for f in os.listdir(dvd_path) if os.path.isdir(os.path.join(dvd_path, f))]
        random.shuffle(subject_folders)

        # Select 10% of subject folders as unknown
        num_unknown = max(1, int(0.1 * len(subject_folders)))
        unknown_folders = subject_folders[:num_unknown]
        known_folders = subject_folders[num_unknown:]

        for subject_folder in subject_folders:
            subject_path = os.path.join(dvd_path, subject_folder)
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

                            if subject_folder in unknown_folders:
                                # Assign label 9999 for unknown folders
                                test_unknown_images.append(img)
                                test_unknown_labels.append(9999)
                            else:
                                # Assign regular labels for known folders
                                if subject_folder not in label_map:
                                    label_map[subject_folder] = label_id
                                    label_id += 1
                                images.append(img)
                                labels.append(label_map[subject_folder])
                    except Exception as e:
                        print(f"Error processing file {compressed_file_path}: {e}")

    # Split known images and labels into training and testing sets
    X_train, X_test_known, y_train, y_test_known = train_test_split(
        images, labels, test_size=0.1, stratify=labels, random_state=42
    )

    # Combine known test data with unknown test data
    X_test = np.array(X_test_known + test_unknown_images)
    y_test = np.array(y_test_known + test_unknown_labels)

    # Shuffle the test data
    test_indices = np.arange(len(X_test))
    np.random.shuffle(test_indices)
    X_test = X_test[test_indices]
    y_test = y_test[test_indices]

    return np.array(X_train), X_test, np.array(y_train), y_test, label_map

def train_classifier(faces, faceID):
    face_recognizer = cv2.face.LBPHFaceRecognizer_create(
        radius=1,
        neighbors=7,
        grid_x=7,
        grid_y=7
    )
    face_recognizer.train(faces, np.array(faceID))
    return face_recognizer

def apply_clahe(image):
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    return clahe.apply(image)

def evaluate_model(y_test, y_pred):
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    cm = confusion_matrix(y_test, y_pred, labels=np.unique(y_test))
    false_accepts = cm.sum(axis=0) - np.diag(cm)
    false_rejects = cm.sum(axis=1) - np.diag(cm)
    total = cm.sum()

    far = false_accepts.sum() / total  # False Accept Rate
    frr = false_rejects.sum() / total  # False Reject Rate

    avg_error = (far + frr) / 2
    fail_rate = (1 - accuracy_score(y_test, y_pred)) * 100

    return {
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "FAR": far,
        "FRR": frr,
        "Average Error": avg_error,
        "Fail Rate (%)": fail_rate
    }

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

def evaluate_feret_recognition(X_test, y_test, face_recognizer, label_map):
    times = []
    y_pred = []
    name = {v: k for k, v in label_map.items()}

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

        if confidence < RECOGNITION_THRESHOLD:
            predicted_name = name.get(label, "Unknown")
        else:
            predicted_name = "Unrecognized"

        y_pred.append(label if predicted_name != "Unrecognized" else -1)

    avg_time = np.mean(times)
    metrics = evaluate_model(y_test, np.array(y_pred))
    metrics["Average Recognition Time (sec)"] = avg_time

    print("FERET Evaluation Results:", metrics)
    return metrics

# Main script
if __name__ == "__main__":
    base_dir = "/Users/tunglambg131003/Downloads/colorferet"  # Path to FERET dataset
    image_size = (231, 314)

    # Load FERET dataset
    X_train, X_test, y_train, y_test, label_map = load_feret_data_with_unknown(base_dir, image_size)
    print(f"Loaded {len(X_train)} training images and {len(X_test)} testing images.")

    # Train the LBPH classifier
    print("Training the LBPH face recognizer...")
    face_recognizer = train_classifier(X_train, y_train)

    # Save the trained model
    face_recognizer.save("models/trained_on_feret.yml")
    print("Model saved as models/trained_on_feret.yml")

    # Evaluate the model on the test set
    evaluation_results = evaluate_feret_recognition(X_test, y_test, face_recognizer, label_map)

    # Save the evaluation results
    with open("feret_evaluation_results.json", "w") as f:
        json.dump(evaluation_results, f, indent=4)
    print("Evaluation results saved to feret_evaluation_results.json")
