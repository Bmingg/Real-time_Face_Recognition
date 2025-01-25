import cv2
import numpy as np
import face_alignment
from collections import Counter
from labels import load_labels
import torch
import time

# Load pre-trained Caffe model for face detection
net = cv2.dnn.readNetFromCaffe("ssd/deploy.prototxt.txt", "ssd/res10_300x300_ssd_iter_140000.caffemodel")
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device='cuda' if torch.cuda.is_available() else 'cpu')

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
        dominant_label = label_counts.most_common(1)[0][0]  # (label, count)
        dominant_label_confidences = [conf for label, conf in top_n if label == dominant_label]
        avg_confidence = sum(dominant_label_confidences) / len(dominant_label_confidences) if dominant_label_confidences else 0
        return dominant_label, avg_confidence

def put_text(confidence, img, name, x_start, y_start):
    cv2.putText(img, f'{name} - Confidence: {confidence:.2f}', (x_start, y_start - 10), 
                cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)

# Load your image
image_path = "/Users/tunglambg131003/Real-time_Face_Recognition/test_data/20/duongtest_1.jpg"  # Replace with your image path
frame = cv2.imread(image_path)

if frame is None:
    print("Error: Could not load the image.")
    exit()

gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces_detected = detect_faces_dnn(frame)

if not faces_detected:
    print("No faces detected in the image.")
else:
    for face in faces_detected:
        x_start, y_start, x_end, y_end = face

            # Ensure face region is not out of bounds
        if x_end > frame.shape[1] or y_end > frame.shape[0]:
                continue  # Skip if the face region exceeds the frame dimensions
            
        off_set = 15

        y_start = y_start + int((y_end-y_start)*0.269) + off_set


        roi_gray = gray_img[y_start:y_end, x_start:x_end]
            # landmarks = get_landmark(gray_img, face)
            # y_start = get_crop_img(landmarks)
            # roi_gray = gray_img[y_start:y_end, x_start:x_end]

        if roi_gray is None or roi_gray.size == 0:  # Ensure ROI is not empty
                continue  # Skip if the ROI is empty or None

        roi_gray = cv2.resize(roi_gray, (300, 300))
        roi_gray = cv2.GaussianBlur(roi_gray, (5, 5), 0)


        collector = cv2.face.StandardCollector_create()
        face_recognizer = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=7, grid_x=7, grid_y=7, threshold=45)
        face_recognizer.read('models/model_test.yml')
        
        LABELS_FILE = 'labels.json'
        name = load_labels(LABELS_FILE)

        face_recognizer.predict_collect(roi_gray, collector)
        label, confidence = get_dominant_label(collector, 7, 45)

        if confidence < 45:
            predicted_name = name.get(str(label), "Unknown")
        else:
            predicted_name = "Unrecognized"

        cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
        put_text(confidence, frame, predicted_name, x_start, y_start)


cv2.imshow("Output", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
