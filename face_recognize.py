import cv2
import numpy as np
import face_alignment
from collections import Counter
from labels import load_labels
import torch
import time
import json
# Load pre-trained Caffe model for face detection
net = cv2.dnn.readNetFromCaffe("ssd/deploy.prototxt.txt", "ssd/res10_300x300_ssd_iter_140000.caffemodel")
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device='cuda' if torch.cuda.is_available() else 'cpu')

def measure_time(step_name, start, end, times):
    elapsed_time = end - start
    times[step_name] = times.get(step_name, 0) + elapsed_time
    return elapsed_time

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
webcam = cv2.VideoCapture(0)

if not webcam.isOpened():
    print("Error: Could not access the camera.")
    exit()

face_recognizer = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=7, grid_x=7, grid_y=7, threshold=70)
face_recognizer.read('models/trained_on_test.yml')

LABELS_FILE = 'labels.json'

name = load_labels(LABELS_FILE)

RECOGNITION_THRESHOLD = 45

REALTIME_RECOGNITION_THRESHOLD = 45

execution_times = {
    "Face Detection": 0,
    "Face Recognition": 0,
    "Total": 0
}
# Initialize FPS calculation
frame_count = 0
fps_start_time = time.time()
while True:
    total_start = time.time()
    ret, frame = webcam.read()  # Capture frame
    if not ret:  # If the webcam stream ends
        break

    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # time face detection
    start_detection = time.time()
    faces_detected = detect_faces_dnn(frame)
    end_detection = time.time()
    detection_time = measure_time("Face Detection", start_detection, end_detection, execution_times)

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


            # Time face recognition
            start_recognition = time.time()
            label, confidence = face_recognizer.predict(roi_gray)
            end_recognition = time.time()
            recognition_time = measure_time("Face Recognition", start_recognition, end_recognition, execution_times)

            # Check if confidence is above the threshold for recognition
            if confidence < REALTIME_RECOGNITION_THRESHOLD:
                predicted_name = name.get(str(label), "Unknown")
            else:
                predicted_name = "Unrecognized"

            cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
            put_text(confidence, frame, predicted_name, x_start, y_start)
    
    total_end = time.time()
    measure_time("Total", total_start, total_end, execution_times)
    
    # Update FPS calculation
    frame_count += 1
    fps_elapsed_time = time.time() - fps_start_time
    fps = frame_count / fps_elapsed_time if fps_elapsed_time > 0 else 0

    # Display FPS on the frame
    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the frame with the recognized faces
    cv2.imshow('Face Recognition', frame)

    # Exit if ESC is pressed
    key = cv2.waitKey(10)
    if key == 27:
        break

webcam.release()
cv2.destroyAllWindows()


# Print execution times and calculate percentages
print("Execution Times (in seconds):")
for step, time_spent in execution_times.items():
    print(f"{step}: {time_spent:.4f} sec")

total_time = execution_times["Total"]
recognition_percentage = (execution_times["Face Recognition"] / total_time) * 100
other_percentage = 100 - recognition_percentage

time_difference = total_time - execution_times["Face Recognition"]

print("\nSummary:")
print(f"Time Difference (Total - Recognition): {time_difference:.4f} sec")
print(f"Face Recognition: {recognition_percentage:.2f}% of total time")
print(f"Other Processes: {other_percentage:.2f}% of total time")


# Final FPS calculation
fps_final = frame_count / (time.time() - fps_start_time)
print(f"Final FPS: {fps_final:.2f}")

# Save results to JSON
results = {
    "Execution Times": execution_times,
    "Recognition Percentage": recognition_percentage,
    "Other Percentage": other_percentage,
    "Time Difference": time_difference,
    "FPS": fps_final
}

with open("execution_times_summary.json", "w") as f:
    json.dump(results, f, indent=4)
