import cv2
import numpy as np
import face_alignment
from collections import Counter
from labels import load_labels
import torch
from scipy.spatial.distance import euclidean, cityblock

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

def compare_histograms(hist1, hist2):
    hist1 = np.array(hist1).flatten()
    hist2 = np.array(hist2).flatten()

    # Check if histograms have the same size
    if hist1.size != hist2.size:
        raise ValueError(f"Histograms have different sizes: {hist1.size} and {hist2.size}")
    
    distances = {
        "Manhattan": cityblock(hist1, hist2)
    }

    return distances

def put_text(distances, img, name, x_start, y_start):
    text = f'{name}:, Euc={distances["Manhattan"]:.2f} '
    cv2.putText(img, text, (x_start, y_start - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)

# Load labels
LABELS_FILE = 'labels.json'
name = load_labels(LABELS_FILE)

# Initialize webcam
cap = cv2.VideoCapture(1)

face_recognizer = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=7, grid_x=7, grid_y=7, threshold=70)
face_recognizer.read('models/trained_on_test4.yml')

face_recognizer2 = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=7, grid_x=7, grid_y=7, threshold=70)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame.")
        break

    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces_detected = detect_faces_dnn(frame)

    if not faces_detected:
        print("No faces detected in the frame.")
    else:
        for face in faces_detected:
            x_start, y_start, x_end, y_end = face

            if x_end > frame.shape[1] or y_end > frame.shape[0]:
                continue

            off_set = 15
            y_start = y_start + int((y_end - y_start) * 0.269) - off_set

            roi_gray = gray_img[y_start:y_end, x_start:x_end]

            if roi_gray is None or roi_gray.size == 0:
                continue

            roi_gray = cv2.resize(roi_gray, (300, 300))
            roi_gray = cv2.GaussianBlur(roi_gray, (5, 5), 0)

            face_recognizer2.train(np.array([roi_gray]), np.array(3))

            hist_roi = face_recognizer2.getHistograms()[0][0]

            distances = {}
            for i in range(len(face_recognizer.getLabels())):
                trained_hist = face_recognizer.getHistograms()[i]
                distances[i] = compare_histograms(hist_roi, trained_hist)

            best_id = min(distances, key=lambda id_: distances[id_]["Manhattan"])
            predicted_name = name.get(str(face_recognizer.getLabels()[best_id][0]), "Unknown")
           

            cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
            put_text(distances[best_id], frame, predicted_name, x_start, y_start)
            print(predicted_name)
            print(distances[best_id])

    # Display the resulting frame
    cv2.imshow('Face Recognition', frame)

    # Exit on 'Esc' key press
    if cv2.waitKey(1) & 0xFF == 27:  # 27 is the ASCII value for the 'Esc' key
        break

# Release the capture object and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()
