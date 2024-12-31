import cv2
import os
import numpy as np

# Paths to the pre-trained models
LBF_MODEL_PATH = "models/lbfmodel.yaml"
CAFFE_MODEL_PROTOTXT = "ssd/deploy.prototxt.txt"
CAFFE_MODEL_WEIGHTS = "ssd/res10_300x300_ssd_iter_140000.caffemodel"

# Initialize the Facemark and load the LBF model
facemark = cv2.face.createFacemarkLBF()
facemark.loadModel(LBF_MODEL_PATH)

# Load pre-trained Caffe model for face detection
net = cv2.dnn.readNetFromCaffe(CAFFE_MODEL_PROTOTXT, CAFFE_MODEL_WEIGHTS)

def detect_faces_dnn(image):
    h, w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    faces = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.7:  # Confidence threshold
            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            x_start, y_start, x_end, y_end = box.astype("int")
            if (x_end - x_start) > 30 and (y_end - y_start) > 30:  # Validate face size
                faces.append((x_start, y_start, x_end, y_end))

    faces = non_max_suppression(faces)
    return faces

def non_max_suppression(boxes, overlap_thresh=0.5):
    if len(boxes) == 0:
        return []

    boxes = np.array(boxes)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = areas.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / areas[order[1:]]

        order = order[np.where(overlap <= overlap_thresh)[0] + 1]

    return boxes[keep].tolist()

def get_landmarks(image, face):
    x_start, y_start, x_end, y_end = face
    face_array = np.array([[x_start, y_start, x_end, y_end]], dtype=int)
    _, landmarks = facemark.fit(image, face_array)
    return landmarks[0][0] if landmarks else None

# Function to crop the face dynamically based on landmarks
def crop_dynamic_face(image, landmarks):
    if landmarks is None:
        return None

    left_eyebrow = landmarks[17:22]
    right_eyebrow = landmarks[22:27]
    min_y_eyebrows = min(np.min(left_eyebrow[:, 1]), np.min(right_eyebrow[:, 1]))
    eyebrow_offset = 5
    y_start = int(min_y_eyebrows + eyebrow_offset)
    x_min = int(np.min(landmarks[:, 0]))
    x_max = int(np.max(landmarks[:, 0]))
    y_end = int(image.shape[0])

    return image[max(0, y_start):y_end, max(0, x_min):min(image.shape[1], x_max)]

def apply_clahe(image):
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    return clahe.apply(image)

def train_classifier(faces, faceID):
    face_recognizer = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=7, grid_x=7, grid_y=7)
    face_recognizer.train(faces, np.array(faceID))
    return face_recognizer

def draw_rect(frame, face):
    x_start, y_start, x_end, y_end = face
    cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), (0, 255, 0), thickness=2)

def put_text(confidence, frame, name, x_start, y_start):
    cv2.putText(frame, f'{name} - {confidence:.2f}', (x_start, y_start - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# Mapping labels to names
name = {0: "hiepnm", 1: "lamnt", 2: "minhvb"}
RECOGNITION_THRESHOLD = 60

# Load trained model
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('models/trained_on_test.yml')

# Start webcam
webcam = cv2.VideoCapture(1)
if not webcam.isOpened():
    print("Error: Could not access the camera.")
    exit()

while True:
    ret, frame = webcam.read()
    if not ret:
        break

    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces_detected = detect_faces_dnn(frame)
    print("Faces Detected:", faces_detected)

    for face in faces_detected:
        landmarks = get_landmarks(gray_img, face)
        if landmarks is None:
            continue

        roi_gray = crop_dynamic_face(gray_img, landmarks)
        if roi_gray is None or roi_gray.size == 0:
            continue

        roi_gray = apply_clahe(cv2.GaussianBlur(roi_gray, (5, 5), 0))
        label, confidence = face_recognizer.predict(roi_gray)

        predicted_name = name.get(label, "Unknown") if confidence < RECOGNITION_THRESHOLD else "Unrecognized"

        draw_rect(frame, face)
        put_text(confidence, frame, predicted_name, face[0], face[1])

    cv2.imshow('Face Recognition', frame)
    if cv2.waitKey(10) == 27:
        break

webcam.release()
cv2.destroyAllWindows()
