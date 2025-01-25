# import cv2
# import numpy as np
# import face_alignment
# from collections import Counter
# from labels import load_labels
# import torch
# from scipy.spatial.distance import euclidean, cityblock
# from sklearn.decomposition import PCA
# from annoy import AnnoyIndex
# import time 

# frame_count = 0
# start_time = time.time()

# # Load pre-trained Caffe model for face detection
# net = cv2.dnn.readNetFromCaffe("ssd/deploy.prototxt.txt", "ssd/res10_300x300_ssd_iter_140000.caffemodel")
# fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device='cuda' if torch.cuda.is_available() else 'cpu')

# def detect_faces_dnn(image):
#     h, w = image.shape[:2]
#     # Prepare image for DNN processing
#     blob = cv2.dnn.blobFromImage(cv2.resize(image, (230, 238)), 1.0, (300, 300), (104.0, 177.0, 123.0))
#     net.setInput(blob)
#     detections = net.forward()
#     faces = []
#     for i in range(detections.shape[2]):
#         confidence = detections[0, 0, i, 2]
#         if confidence > 0.7:
#             box = detections[0, 0, i, 3:7] * [w, h, w, h]
#             faces.append(box.astype("int"))
#     return faces

# def compare_histograms(hist1, hist2):

#     hist1 = np.array(hist1).flatten()
#     hist2 = np.array(hist2).flatten()

#     epsilon = 1e-10
#     hist1 = np.maximum(hist1, epsilon)  
#     hist2 = np.maximum(hist2, epsilon)

#     chi_square_stat = np.sum(((hist1 - hist2) ** 2) / (hist1 + hist2))

    
#     distances = {
#        "Chisquare": chi_square_stat 
#     }


#     return distances

# def put_text(distances, img, name, x_start, y_start):
#     # text = f'{name}:, Euc={distances["Manhattan"]:.2f} '
#     text = f'{name}:, Error: {distances} '
#     cv2.putText(img, text, (x_start, y_start - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)

# # Load labels
# LABELS_FILE = 'labels.json'
# name = load_labels(LABELS_FILE)

# # Initialize webcam
# cap = cv2.VideoCapture(0)

# face_recognizer = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=7, grid_x=7, grid_y=7, threshold=45)
# face_recognizer.read('models/model_test.yml')

# face_recognizer2 = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=7, grid_x=7, grid_y=7, threshold=45)

# features = np.array(face_recognizer.getHistograms())
# size = len(np.array(face_recognizer.getHistograms()))
# features = np.reshape(features, (size, -1))
# labels = np.array(face_recognizer.getLabels())
# labels = labels.ravel()

# pca = PCA(n_components=250)
# features_pca = pca.fit_transform(features)

# def build_annoy_index(dataset, path, num_trees=30):
#     """
#     Build an Annoy index for approximate nearest neighbor search.
#     """
#     feature_dim = dataset.shape[1]
#     index = AnnoyIndex(feature_dim, metric='euclidean')
#     # index = AnnoyIndex(feature_dim, metric='manhattan')
#     # index = AnnoyIndex(feature_dim, metric='angular')
#     # index = AnnoyIndex(feature_dim, metric='dot')
#     # index = AnnoyIndex(feature_dim, metric='hamming')
#     for i, vector in enumerate(dataset):
#         index.add_item(i, vector)
#     index.build(num_trees)
#     index.save(path)
#     return index

# def load_annoy_index(dataset, path):
#     annoy = AnnoyIndex(dataset.shape[1], metric='euclidean')
#     annoy.load(path)
#     return annoy

# def lsh_search(query, index, dataset_labels, k=1):
#     """
#     Perform LSH search using Annoy.
#     """
#     indices, error = index.get_nns_by_vector(query, k, include_distances=True)
#     return dataset_labels[indices[0]], error

# annoy_index_pca = build_annoy_index(features_pca, path="models/annoy_index_pca_250_euc")
# annoy_index_pca = load_annoy_index(features_pca, path="models/annoy_index_pca_250_euc")

# while True:
#     # Capture frame-by-frame
#     ret, frame = cap.read()
#     loop_start_time = time.time()

#     if not ret:
#         print("Error: Could not read frame.")
#         break

#     gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces_detected = detect_faces_dnn(frame)

#     if not faces_detected:
#         print("No faces detected in the frame.")
#     else:
#         for face in faces_detected:
#             x_start, y_start, x_end, y_end = face

#             if x_end > frame.shape[1] or y_end > frame.shape[0]:
#                 continue

#             off_set = 15
#             y_start = y_start + int((y_end - y_start) * 0.269) + off_set

#             roi_gray = gray_img[y_start:y_end, x_start:x_end]

#             if roi_gray is None or roi_gray.size == 0:
#                 continue

#             roi_gray = cv2.resize(roi_gray, (300, 300))
#             roi_gray = cv2.GaussianBlur(roi_gray, (5, 5), 0)

#             face_recognizer2.train(np.array([roi_gray]), np.array(3))

#             # hist_roi = face_recognizer2.getHistograms()[0][0]

#             test_instance = face_recognizer2.getHistograms()[0][0]
#             test_feature = pca.transform([np.array(test_instance)])
#             predicted_label, error = lsh_search(test_feature[0], annoy_index_pca, labels)
#             predicted_name = name.get(str(predicted_label), "Unknown")
          
#             # distances = {}
#             # for i in range(len(face_recognizer.getLabels())):
#             #     trained_hist = features_pca[i]
#             #     distances[i] = compare_histograms(test_feature, trained_hist)

#             # best_id = min(distances, key=lambda id_: distances[id_]["Chisquare"])
#             # predicted_name = name.get(str(face_recognizer.getLabels()[best_id][0]), "Unknown")
           
#             cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
#             # put_text(distances[best_id], frame, predicted_name, x_start, y_start)
#             put_text(error, frame, predicted_name, x_start, y_start) 

#     frame_count += 1

#     # Calculate FPS
#     elapsed_time = time.time() - start_time
#     if elapsed_time > 1.0:
#         fps = frame_count / elapsed_time
#         frame_count = 0
#         start_time = time.time()

#     # Display FPS on the video feed
#     cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

#     # Display the resulting frame
#     cv2.imshow('Face Recognition', frame)

#     # Exit on 'Esc' key press
#     if cv2.waitKey(1) & 0xFF == 27:  # 27 is the ASCII value for the 'Esc' key
#         break

# # Release the capture object and close any OpenCV windows
# cap.release()
# cv2.destroyAllWindows()

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
frame_count = 0
start_time = time.time()

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

face_recognizer = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=7, grid_x=7, grid_y=7, threshold=45)
face_recognizer.read('models/model_test.yml')

LABELS_FILE = 'labels.json'

name = load_labels(LABELS_FILE)

RECOGNITION_THRESHOLD = 45

REALTIME_RECOGNITION_THRESHOLD = 45
while True:
    ret, frame = webcam.read()  # Capture frame
    loop_start_time = time.time()
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

            y_start = y_start + int((y_end-y_start)*0.269) + off_set


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

    frame_count += 1

    # Calculate FPS
    elapsed_time = time.time() - start_time
    if elapsed_time > 1.0:
        fps = frame_count / elapsed_time
        frame_count = 0
        start_time = time.time()

    # Display FPS on the video feed
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)


    # Show the frame with the recognized faces
    cv2.imshow('Face Recognition', frame)

    # Exit if ESC is pressed
    key = cv2.waitKey(10)
    if key == 27:
        break

webcam.release()
cv2.destroyAllWindows()

# import cv2
# import numpy as np
# import face_alignment
# from collections import Counter
# from labels import load_labels
# import torch
# import time
# import json
# from evaluation import evaluate_model
# # Load pre-trained Caffe model for face detection
# net = cv2.dnn.readNetFromCaffe("ssd/deploy.prototxt.txt", "ssd/res10_300x300_ssd_iter_140000.caffemodel")
# fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device='cuda' if torch.cuda.is_available() else 'cpu')

# true_positive- The model predict that the person is Paul and the person was Paul (true Acceptance )
# false_negative- The model predict that person is unrecognized or other people and the person was Paul (false rejection)
# false_positive- The model predict that the person is Paul or orther people and the person was unrecognized( false acceptance)
# true_negative- The model predict that the person is unrecognized and the person was unrecognized (true rejection)

# true_label1 = "hiepnm"
# true_label2 = "Unrecognized"
# current_label = "Unrecognized"
# false_negative_count = 0 
# false_positive_count = 0
# true_positive_count = 0
# true_negative_count = 0

# total_count = 0
# def measure_time(step_name, start, end, times):
#     elapsed_time = end - start
#     times[step_name] = times.get(step_name, 0) + elapsed_time
#     return elapsed_time

# def detect_faces_dnn(image):
#     h, w = image.shape[:2]
#     # Prepare image for DNN processing
#     blob = cv2.dnn.blobFromImage(cv2.resize(image, (230, 238)), 1.0, (300, 300), (104.0, 177.0, 123.0))
#     net.setInput(blob)
#     detections = net.forward()
#     faces = []
#     for i in range(detections.shape[2]):
#         confidence = detections[0, 0, i, 2]
#         if confidence > 0.7:  
#             box = detections[0, 0, i, 3:7] * [w, h, w, h]
#             faces.append(box.astype("int"))
#     return faces

# def get_dominant_label(collector, n, threshold):
#     list_of_conf_id_tuples = collector.getResults(sorted=True)
#     filtered_tuples = [(label, conf) for label, conf in list_of_conf_id_tuples if conf <= threshold]
#     if n >= len(filtered_tuples):
#         top_n = filtered_tuples
#     else:
#         top_n = filtered_tuples[:n]
#     labels = [label for label, _ in top_n]
#     label_counts = Counter(labels)
#     if not label_counts:
#         return None, 100
#     else:
#         dominant_label = label_counts.most_common(1)[0][0]  # (label, count)
#         dominant_label_confidences = [conf for label, conf in top_n if label == dominant_label]
#         # Calculate the average confidence for the dominant label
#         avg_confidence = sum(dominant_label_confidences) / len(dominant_label_confidences) if dominant_label_confidences else 0
#         return dominant_label, avg_confidence

# def put_text(confidence, img, name, x_start, y_start):
#     cv2.putText(img, f'{name} - Confidence: {confidence:.2f}', (x_start, y_start - 10), 
#                 cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)

# # Initialize webcam
# webcam = cv2.VideoCapture(0)

# if not webcam.isOpened():
#     print("Error: Could not access the camera.")
#     exit()

# face_recognizer = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=7, grid_x=7, grid_y=7, threshold=45)
# face_recognizer.read('models/model_test.yml')

# LABELS_FILE = 'labels.json'

# name = load_labels(LABELS_FILE)

# RECOGNITION_THRESHOLD = 45

# REALTIME_RECOGNITION_THRESHOLD = 45

# execution_times = {
#     "Face Detection": 0,
#     "Face Recognition": 0,
#     "Total": 0
# }

# # Initialize FPS calculation
# frame_count = 0
# fps_start_time = time.time()
# while True:
#     total_start = time.time()
#     ret, frame = webcam.read()  # Capture frame
#     if not ret:  # If the webcam stream ends
#         break

#     gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # time face detection
#     start_detection = time.time()
#     faces_detected = detect_faces_dnn(frame)
#     end_detection = time.time()
#     detection_time = measure_time("Face Detection", start_detection, end_detection, execution_times)

#     faces_detected = detect_faces_dnn(frame)
#     # print("Faces Detected: ", faces_detected)

#     if not faces_detected:  # If no faces are detected
#         print("No faces detected in the frame.")

#     else:
#         for face in faces_detected:
#             x_start, y_start, x_end, y_end = face

#             # Ensure face region is not out of bounds
#             if x_end > frame.shape[1] or y_end > frame.shape[0]:
#                 continue  # Skip if the face region exceeds the frame dimensions
            
#             off_set = 15

#             y_start = y_start + int((y_end-y_start)*0.269) + off_set


#             roi_gray = gray_img[y_start:y_end, x_start:x_end]
#             # landmarks = get_landmark(gray_img, face)
#             # y_start = get_crop_img(landmarks)
#             # roi_gray = gray_img[y_start:y_end, x_start:x_end]

#             if roi_gray is None or roi_gray.size == 0:  # Ensure ROI is not empty
#                 continue  # Skip if the ROI is empty or None

#             roi_gray = cv2.resize(roi_gray, (300, 300))
#             roi_gray = cv2.GaussianBlur(roi_gray, (5, 5), 0)

#             # roi_gray = apply_clahe(roi_gray)
#             collector = cv2.face.StandardCollector_create()
#             start_recognition = time.time()

#             face_recognizer.predict_collect(roi_gray, collector)
#             label, confidence = get_dominant_label(collector, 7, RECOGNITION_THRESHOLD)
#             end_recognition = time.time()
#             recognition_time = measure_time("Face Recognition", start_recognition, end_recognition, execution_times)
#             # confidence = collector.getMinDist()
#             # label, confidence = face_recognizer.predict(roi_gray)
#             # print("Confidence:", confidence)
#             # print("Label:", label)


#             # Time face recognition
#             # label, confidence = face_recognizer.predict(roi_gray)


#             # Check if confidence is above the threshold for recognition
#             if confidence < REALTIME_RECOGNITION_THRESHOLD:
#                 predicted_name = name.get(str(label), "Unknown")
#             else:
#                 predicted_name = "Unrecognized"

#             # if current_label == true_label1:
#             #     if predicted_name == true_label1:
#             #         true_positive_count += 1
#             #     else:
#             #         false_negative_count += 1
#             # elif current_label == true_label2:
#             #     if predicted_name == true_label2:
#             #         true_negative_count += 1
#             #     else:
#             #         false_positive_count +=1
                
#             total_count += 1

#             cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
#             put_text(confidence, frame, predicted_name, x_start, y_start)
    
#     total_end = time.time()
#     measure_time("Total", total_start, total_end, execution_times)
    
#     # Update FPS calculation
#     frame_count += 1
#     fps_elapsed_time = time.time() - fps_start_time
#     fps = frame_count / fps_elapsed_time if fps_elapsed_time > 0 else 0

#     # Display FPS on the frame
#     cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

#     # Show the frame with the recognized faces
#     cv2.imshow('Face Recognition', frame)

#     # Exit if ESC is pressed
#     key = cv2.waitKey(10)
#     if key == 27:
#         break

# webcam.release()
# cv2.destroyAllWindows()


# # Print execution times and calculate percentages
# print("Execution Times (in seconds):")
# for step, time_spent in execution_times.items():
#     print(f"{step}: {time_spent:.4f} sec")

# total_time = execution_times["Total"]
# recognition_percentage = (execution_times["Face Recognition"] / total_time) * 100
# other_percentage = 100 - recognition_percentage

# time_difference = total_time - execution_times["Face Recognition"]

# print("\nSummary:")
# print(f"Face Recognition: {recognition_percentage:.2f}% of total time")
# print(f"Other Processes: {other_percentage:.2f}% of total time")


# # Final FPS calculation
# fps_final = frame_count / (time.time() - fps_start_time)
# print(f"Final FPS: {fps_final:.2f}")
# recognition_percentage = f"{recognition_percentage:.2f}%"
# # Save results to JSON
# time_result = {
#     "Execution Times": execution_times,
#     "Recognition Percentage": recognition_percentage,
#     "FPS": fps_final
# }

# model_result = {
#     "True Positive": true_positive_count,
#     "False Positive": false_positive_count,
#     "False Negative": false_negative_count,
#     "True Negative": true_negative_count 
# }

# with open("execution_model_summary.txt", "a") as f:
#     f.write(str(model_result) + "\n")

# with open("execution_times_summary.txt", "a") as f:
#     f.write(str(time_result) + "\n")

