import cv2
import numpy as np
import face_alignment
from collections import Counter
from labels import load_labels
import torch
from scipy.spatial.distance import euclidean, cityblock, mahalanobis, canberra

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


    
#     distances = {
 
#         "Canberra": canberra(hist1, hist2),
        
#     }

#     return distances


# def put_text(distances, img, name, x_start, y_start):
#     text = f'{name}:, Can={distances["Canberra"]:.2f} '
#     cv2.putText(img, text, (x_start, y_start - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)

# image_path = "/Users/tunglambg131003/Real-time_Face_Recognition/test_data/20/duongtest_3.jpg"  
# frame = cv2.imread(image_path)

# if frame is None:
#     print("Error: Could not load the image.")
#     exit()

# face_recognizer = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=7, grid_x=7, grid_y=7, threshold=70)
# face_recognizer.read('models/trained_on_test4.yml')

# face_recognizer2 = cv2.face.LBPHFaceRecognizer.create(radius=1, neighbors=7, grid_x=7, grid_y=7, threshold=70)

# LABELS_FILE = 'labels.json'

# name = load_labels(LABELS_FILE)

# gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# faces_detected = detect_faces_dnn(frame)

# if not faces_detected:
#     print("No faces detected in the image.")
# else:
#     for face in faces_detected:
#         x_start, y_start, x_end, y_end = face

#         if x_end > frame.shape[1] or y_end > frame.shape[0]:
#             continue

#         off_set = 15
#         y_start = y_start + int((y_end - y_start) * 0.269) - off_set

#         roi_gray = gray_img[y_start:y_end, x_start:x_end]

#         if roi_gray is None or roi_gray.size == 0:
#             continue

#         roi_gray = cv2.resize(roi_gray, (300, 300))
#         roi_gray = cv2.GaussianBlur(roi_gray, (5, 5), 0)

#         face_recognizer2.train(np.array([roi_gray]), np.array(3))

#         hist_roi = face_recognizer2.getHistograms()[0][0]

#         distances = {}
#         for i in range(len(face_recognizer.getLabels())):
#             trained_hist = face_recognizer.getHistograms()[i]
#             distances[i] = compare_histograms(hist_roi, trained_hist)

#         best_id = min(distances, key=lambda id_: distances[id_]["Canberra"])
#         predicted_name = name.get(str(face_recognizer.getLabels()[best_id][0]), "Unknown")
#         print(distances)

#         cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
#         put_text(distances[best_id], frame, predicted_name, x_start, y_start)

# cv2.imshow('Face Recognition', frame)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
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


    
    distances = {
 
        "Canberra": canberra(hist1, hist2),
        
    }

    return distances

image_paths = ["/Users/tunglambg131003/Real-time_Face_Recognition/test_img2.jpg",
    "/Users/tunglambg131003/Real-time_Face_Recognition/test_data/0/hiepnm_3.jpg",
    "/Users/tunglambg131003/Real-time_Face_Recognition/dataset_test/3/tungvd_1.jpg",
    "/Users/tunglambg131003/Real-time_Face_Recognition/dataset_test/4/thaohoang_1.jpg",
    "/Users/tunglambg131003/Real-time_Face_Recognition/dataset_test/5/vanhv_1.jpg",
    "/Users/tunglambg131003/Real-time_Face_Recognition/dataset_test/6/linhbm_1.jpg",
    "/Users/tunglambg131003/Real-time_Face_Recognition/dataset_test/7/poe_1.jpg",
    "/Users/tunglambg131003/Real-time_Face_Recognition/dataset_test/8/kiendhc_1.jpg",
    "/Users/tunglambg131003/Real-time_Face_Recognition/dataset_test/9/datnd_1.jpg",
    "/Users/tunglambg131003/Real-time_Face_Recognition/dataset_test/10/thanhtq_1.jpg",
    "/Users/tunglambg131003/Real-time_Face_Recognition/dataset_test/11/phunt_1.jpg",
    "/Users/tunglambg131003/Real-time_Face_Recognition/dataset_test/12/anhndh_1.jpg",
    "/Users/tunglambg131003/Real-time_Face_Recognition/dataset_test/13/namdd_1.jpg",
    "/Users/tunglambg131003/Real-time_Face_Recognition/dataset_test/14/anhhtt_1.jpg",
    "/Users/tunglambg131003/Real-time_Face_Recognition/dataset_test/15/manhdt_1.jpg",
    "/Users/tunglambg131003/Real-time_Face_Recognition/dataset_test/16/giangns_1.jpg",
]

face_recognizer = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=7, grid_x=7, grid_y=7, threshold=70)
face_recognizer.read('models/trained_on_test4.yml')

face_recognizer2 = cv2.face.LBPHFaceRecognizer.create(radius=1, neighbors=7, grid_x=7, grid_y=7, threshold=70)

LABELS_FILE = 'labels.json'
name = load_labels(LABELS_FILE)

for image_path in image_paths:
    frame = cv2.imread(image_path)

    if frame is None:
        print(f"Error: Could not load the image at {image_path}.")
        continue

    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces_detected = detect_faces_dnn(frame)

    if not faces_detected:
        print(f"No faces detected in the image: {image_path}")
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

            best_id = min(distances, key=lambda id_: distances[id_]["Canberra"])
            predicted_name = name.get(str(face_recognizer.getLabels()[best_id][0]), "Unknown")
            print(f"Image: {image_path} - {distances[best_id]["Canberra"]:.2f} ({predicted_name})")