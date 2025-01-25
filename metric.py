import cv2
import os
import numpy as np
from collections import Counter
from labels import load_labels
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# def labels_for_training_data(directory):
#     faces = []
#     faceID = []

#     for path, subdirnames, filenames in os.walk(directory):
#         for filename in filenames:
#             if filename.startswith("."):
#                 continue
#             id = os.path.basename(path)
#             img_path = os.path.join(path, filename)
#             img = cv2.imread(img_path)
#             img = cv2.resize(img, (230, 238))
#             img = cv2.GaussianBlur(img, (5,5),0)
#             if img is None:
#                 continue
#             gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#             faces.append(gray_img)
#             faceID.append(int(id))

#     return faces, faceID

# def train_classifier(faces, faceID):
#     face_recognizer = cv2.face.LBPHFaceRecognizer_create(
#         radius=1,
#         neighbors=7,
#         grid_x=7,
#         grid_y=7,
#         threshold=45)
#     face_recognizer.train(faces, np.array(faceID))
#     return face_recognizer

# folder = "new_augmented_dataset_crop_test"
# faces, faceID = labels_for_training_data(folder)
# face_recognizer = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=7, grid_x=7, grid_y=7, threshold=45)
# face_recognizer = train_classifier(faces, faceID)
# face_recognizer.save('models/model_test.yml')

def load_trained_model(model_path='models/model_test.yml'):
    if os.path.exists(model_path):
        face_recognizer = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=7, grid_x=7, grid_y=7, threshold=45)
        face_recognizer.read(model_path)
        return face_recognizer
    return None

net = cv2.dnn.readNetFromCaffe("ssd/deploy.prototxt.txt", "ssd/res10_300x300_ssd_iter_140000.caffemodel")

def detect_faces_dnn(image):
    h, w = image.shape[:2]
    # Prepare image for DNN processing
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

def get_dominant_label(collector, n, threshold):
    list_of_conf_id_tuples = collector.getResults(sorted=True)
    filtered_tuples = [(label, conf) for label, conf in list_of_conf_id_tuples if conf <= threshold]
    if n >= len(filtered_tuples):
        top_n = filtered_tuples
    else:
        top_n = filtered_tuples[:n]
    labels = [label for label, _ in top_n]
    label_counts = Counter(labels)
    if not label_counts:
        return None, 100
    else:
        dominant_label = label_counts.most_common(1)[0][0]  
        dominant_label_confidences = [conf for label, conf in top_n if label == dominant_label]
        avg_confidence = sum(dominant_label_confidences) / len(dominant_label_confidences) if dominant_label_confidences else 0
        return dominant_label, avg_confidence

def evaluate_model_on_directory(directory, model_path='models/model_test.yml'):
    face_recognizer = load_trained_model(model_path)
    if face_recognizer is None:
        print("Failed to load model.")
        return

    true_labels_b = []
    predicted_labels_b = []
    true_labels_c = []
    predicted_labels_c = []

    for path, subdirnames, filenames in os.walk(directory):
        for filename in filenames:
            if filename.startswith("."):
                continue

            img_path = os.path.join(path, filename)
            img = cv2.imread(img_path)
            if img is None:
                continue

            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces_detected = detect_faces_dnn(img)

            if not faces_detected:
                print(f"No faces detected in {filename}.")
                continue

            for face in faces_detected:
                x_start, y_start, x_end, y_end = face
                if x_end > img.shape[1] or y_end > img.shape[0]:
                    continue

                off_set = 15
                y_start = y_start + int((y_end - y_start) * 0.269) + off_set

                roi_gray = gray_img[y_start:y_end, x_start:x_end]

                if roi_gray is None or roi_gray.size == 0:
                    continue

                roi_gray = cv2.resize(roi_gray, (300, 300))
                roi_gray = cv2.GaussianBlur(roi_gray, (5, 5), 0)

                collector = cv2.face.StandardCollector_create()
                face_recognizer.predict_collect(roi_gray, collector)

                label, confidence = get_dominant_label(collector, 7, 45)

                true_label = int(os.path.basename(path))  

                if true_label == 9999:
                    true_labels_c.append(true_label)
                    predicted_labels_c.append(label)
                else:
                    true_labels_b.append(true_label)
                    predicted_labels_b.append(label)

    metrics = {}
    metrics["Precision_B"] = precision_score(true_labels_b, predicted_labels_b, average="weighted", zero_division=0)
    metrics["Recall_B"] = recall_score(true_labels_b, predicted_labels_b, average="weighted", zero_division=0)
    metrics["F1_Score_B"] = f1_score(true_labels_b, predicted_labels_b, average="weighted", zero_division=0)

    # Metrics for C (unknown individuals)
    metrics["Precision_C"] = precision_score(true_labels_c, predicted_labels_c, average="binary", pos_label=9999, zero_division=0)
    metrics["Recall_C"] = recall_score(true_labels_c, predicted_labels_c, average="binary", pos_label=9999, zero_division=0)
    metrics["F1_Score_C"] = f1_score(true_labels_c, predicted_labels_c, average="binary", pos_label=9999, zero_division=0)

    # False Acceptance Rate (FAR) for C
    false_accepts = np.sum((np.array(predicted_labels_c) != 9999) & (np.array(true_labels_c) == 9999))
    total_unknown = np.sum(np.array(true_labels_c) == 9999)
    metrics["FAR"] = false_accepts / total_unknown if total_unknown > 0 else 0

    # False Rejection Rate (FRR) for B
    false_rejects = np.sum((np.array(predicted_labels_b) == 9999) & (np.array(true_labels_b) != 9999))
    total_known = len(true_labels_b)
    metrics["FRR"] = false_rejects / total_known if total_known > 0 else 0

    # Accuracy
    correct_b = np.sum(np.array(predicted_labels_b) == np.array(true_labels_b))
    correct_c = np.sum(np.array(predicted_labels_c) == np.array(true_labels_c))
    total_predictions = len(true_labels_b) + len(true_labels_c)
    metrics["Accuracy"] = (correct_b + correct_c) / total_predictions if total_predictions > 0 else 0

    # Recognition Rate
    metrics["Recognition_Rate"] = (correct_b / len(true_labels_b)) * 100 if len(true_labels_b) > 0 else 0

    # Average Error Rate
    metrics["Average_Error"] = (metrics["FAR"] + metrics["FRR"]) / 2

    # Fail Rate
    metrics["Fail_Rate"] = (1 - metrics["Accuracy"]) * 100

    return metrics

metrics = evaluate_model_on_directory("/Users/tunglambg131003/Real-time_Face_Recognition/test_data", model_path='models/model_test.yml')

print(metrics)
