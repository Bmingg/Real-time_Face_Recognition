import cv2
import numpy as np
import face_alignment
from collections import Counter
from labels import load_labels
import torch

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
        # Calculate the average confidence for the dominant label
        avg_confidence = sum(dominant_label_confidences) / len(dominant_label_confidences) if dominant_label_confidences else 0
        return dominant_label, avg_confidence

def put_text(confidence, img, name, x_start, y_start):
    cv2.putText(img, f'{name} - Confidence: {confidence:.2f}', (x_start, y_start - 10), 
                cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)

# Initialize webcam
webcam = cv2.VideoCapture(1)

if not webcam.isOpened():
    print("Error: Could not access the camera.")
    exit()

face_recognizer = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=7, grid_x=7, grid_y=7, threshold=70)
face_recognizer.read('models/trained_on_test4.yml')

LABELS_FILE = 'labels.json'

name = load_labels(LABELS_FILE)

RECOGNITION_THRESHOLD = 45

REALTIME_RECOGNITION_THRESHOLD = 45
while True:
    ret, frame = webcam.read()  # Capture frame
    if not ret:  # If the webcam stream ends
        break

    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces_detected = detect_faces_dnn(frame)
    # print("Faces Detected: ", faces_detected)

    if not faces_detected:  # If no faces are detected
        print("No faces detected in the frame.")
    else:
        for face in faces_detected:
            x_start, y_start, x_end, y_end = face

            # Ensure face region is not out of bounds
            if x_end > frame.shape[1] or y_end > frame.shape[0]:
                continue  # Skip if the face region exceeds the frame dimensions
            
            off_set = 15

            y_start = y_start + int((y_end-y_start)*0.269) - off_set


            roi_gray = gray_img[y_start:y_end, x_start:x_end]
            # landmarks = get_landmark(gray_img, face)
            # y_start = get_crop_img(landmarks)
            # roi_gray = gray_img[y_start:y_end, x_start:x_end]

            if roi_gray is None or roi_gray.size == 0:  # Ensure ROI is not empty
                continue  # Skip if the ROI is empty or None

            roi_gray = cv2.resize(roi_gray, (300, 300))
            roi_gray = cv2.GaussianBlur(roi_gray, (5, 5), 0)

            # roi_gray = apply_clahe(roi_gray)
            collector = cv2.face.StandardCollector_create()
            
            face_recognizer.predict_collect(roi_gray, collector)
            label, confidence = get_dominant_label(collector, 7, RECOGNITION_THRESHOLD)
            # confidence = collector.getMinDist()
            # label, confidence = face_recognizer.predict(roi_gray)
            # print("Confidence:", confidence)
            # print("Label:", label)

            # Check if confidence is above the threshold for recognition
            if confidence < REALTIME_RECOGNITION_THRESHOLD:
                predicted_name = name.get(str(label), "Unknown")
            else:
                predicted_name = "Unrecognized"

            cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
            put_text(confidence, frame, predicted_name, x_start, y_start)

    # Show the frame with the recognized faces
    cv2.imshow('Face Recognition', frame)

    # Exit if ESC is pressed
    key = cv2.waitKey(10)
    if key == 27:
        break

webcam.release()
cv2.destroyAllWindows()
