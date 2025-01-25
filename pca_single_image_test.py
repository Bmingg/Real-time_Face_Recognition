import cv2
import numpy as np
import face_alignment
from collections import Counter
from labels import load_labels
import torch
from scipy.spatial.distance import euclidean, cityblock
from sklearn.decomposition import PCA
from annoy import AnnoyIndex

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

def put_text(error, img, name, x_start, y_start):
    text = f'{name}: Error: {error:.2f}'
    cv2.putText(img, text, (x_start, y_start - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)

# Load labels
LABELS_FILE = 'labels.json'
name = load_labels(LABELS_FILE)

# Load face recognizer model
face_recognizer = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=7, grid_x=7, grid_y=7, threshold=45)
face_recognizer.read('models/model_test.yml')

features = np.array(face_recognizer.getHistograms())
size = len(np.array(face_recognizer.getHistograms()))
features = np.reshape(features, (size, -1))
labels = np.array(face_recognizer.getLabels())
labels = labels.ravel()

pca = PCA(n_components=250)
features_pca = pca.fit_transform(features)

# Load Annoy index
def load_annoy_index(dataset, path):
    annoy = AnnoyIndex(dataset.shape[1], metric='euclidean')
    annoy.load(path)
    return annoy

def lsh_search(query, index, dataset_labels, k=1):
    """
    Perform LSH search using Annoy.
    """
    indices, error = index.get_nns_by_vector(query, k, include_distances=True)
    return dataset_labels[indices[0]], error

annoy_index_pca = load_annoy_index(features_pca, path="models/annoy_index_pca_250_euc")

# Load the test image
image_path = "/Users/tunglambg131003/Real-time_Face_Recognition/test_data/20/duongtest_1.jpg"
image = cv2.imread(image_path)

if image is None:
    print("Error: Could not load image.")
else:
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces_detected = detect_faces_dnn(image)

    if not faces_detected:
        print("No faces detected in the image.")
    else:
        for face in faces_detected:
            x_start, y_start, x_end, y_end = face

            # Ensure face region is not out of bounds
            if x_end > image.shape[1] or y_end > image.shape[0]:
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

            face_recognizer.train(np.array([roi_gray]), np.array(3))
            test_instance = face_recognizer.getHistograms()[0][0]
            test_feature = pca.transform([np.array(test_instance)])
            predicted_label, error = lsh_search(test_feature[0], annoy_index_pca, labels)
            predicted_name = name.get(str(predicted_label), "Unknown")

          
            cv2.rectangle(image, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
            put_text(error[0], image, predicted_name, x_start, y_start) 

    # Save or display the result

    cv2.imshow("Face Recognition", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    