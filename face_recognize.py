import cv2
import os
import numpy as np

# Load pre-trained Caffe model for face detection
net = cv2.dnn.readNetFromCaffe("ssd/deploy.prototxt.txt", "ssd/res10_300x300_ssd_iter_140000.caffemodel")

# def detect_faces_dnn(image):
#     h, w = image.shape[:2]
#     # Prepare image for DNN processing
#     blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
#     net.setInput(blob)
#     detections = net.forward()
#     faces = []
#     for i in range(detections.shape[2]):
#         confidence = detections[0, 0, i, 2]
#         if confidence > 0.7:  
#             box = detections[0, 0, i, 3:7] * [w, h, w, h]
#             faces.append(box.astype("int"))
#     return faces
def detect_faces_dnn(image):
    h, w = image.shape[:2]
    # Prepare image for DNN processing
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.7:  
            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            x_start, y_start, x_end, y_end = box.astype("int")
            
            # Adjust bounding box to start below eyebrows
            box_height = y_end - y_start
            y_start += int(box_height * 0.2)  # Adjust top by 20% of height
            
            # Clamp values to image bounds
            y_start = max(0, y_start)
            y_end = min(h, y_end)
            x_start = max(0, x_start)
            x_end = min(w, x_end)
            
            # Save adjusted box
            faces.append([x_start, y_start, x_end, y_end])
    return faces

def labels_for_training_data(directory):
    faces = []
    faceID = []

    for path, subdirnames, filenames in os.walk(directory):
        for filename in filenames:
            if filename.startswith("."):
                print("skipping system file")
                continue
            id = os.path.basename(path)
            img_path = os.path.join(path, filename)
            print("img_path", img_path)
            print("id: ", id)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (231, 314))
            img = cv2.GaussianBlur(img, (5,5),0)
            if img is None:
                print ("Not Loaded Properly")
                continue

            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            roi_gray = gray_img
            faces.append(roi_gray)
            faceID.append(int(id))

    return faces, faceID

# Train Classifier
def train_classifier(faces, faceID):
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.train(faces, np.array(faceID))
    return face_recognizer

# Drawing a Rectangle on the Face Function
def draw_rect(test_img, face):
    x_start, y_start, x_end, y_end = face
    
    # # Adjust the top of the bounding box to start below the eyebrows
    # box_height = y_end - y_start
    # y_start += int(box_height * 0.2)  # Adjust the top by 20% of the height
    
    # # Ensure the adjusted box stays within the image bounds
    # y_start = max(0, y_start)
    
    # # Draw the rectangle with the adjusted coordinates
    cv2.rectangle(test_img, (x_start, y_start), (x_end, y_end), (0, 255, 0), thickness=3)



# Putting text on images
def put_text(confidence, img, name, x_start, y_start):
    cv2.putText(img, f'{name} - Confidence: {confidence:.2f}', (x_start, y_start - 10), 
                cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)

# Path to your dataset
folder = "augmented_dataset_crop_test"

faces, faceID = labels_for_training_data(folder)
face_recognizer = train_classifier(faces, faceID)
face_recognizer.save('models/trained_on_test.yml')

# Name dictionary (change according to your training labels)
name = {0: "hiepnm", 1: "lamnt", 2: "minhvb"}  # Add more names if needed

# Initialize webcam
webcam = cv2.VideoCapture(0)

if not webcam.isOpened():
    print("Error: Could not access the camera.")
    exit()

# Define a confidence threshold for recognition
RECOGNITION_THRESHOLD = 60

while True:
    ret, frame = webcam.read()  # Capture frame
    if not ret:  # If the webcam stream ends
        break

    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces_detected = detect_faces_dnn(frame)
    print("Faces Detected: ", faces_detected)

    if not faces_detected:  # If no faces are detected
        print("No faces detected in the frame.")
    else:
        for face in faces_detected:

            x_start, y_start, x_end, y_end = face
  

            # Ensure face region is not out of bounds
            if x_end > frame.shape[1] or y_end > frame.shape[0]:
                continue  # Skip if the face region exceeds the frame dimensions

            roi_gray = gray_img[y_start:y_end, x_start:x_end]
            
            if roi_gray is None or roi_gray.size == 0:  # Ensure ROI is not empty
                continue  # Skip if the ROI is empty or None

            roi_gray = cv2.resize(roi_gray, (231, 314))
            roi_gray = cv2.GaussianBlur(roi_gray, (5, 5), 0)

            label, confidence = face_recognizer.predict(roi_gray)
            print("Confidence:", confidence)
            print("Label:", label)

            # Check if confidence is above the threshold for recognition
            if confidence < RECOGNITION_THRESHOLD:
                predicted_name = name.get(label, "Unknown")
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

# Release the webcam and close all OpenCV windows
webcam.release()
cv2.destroyAllWindows()
