import cv2
import numpy as np
import face_alignment
from collections import Counter
from labels import load_labels
import torch
from scipy.spatial.distance import euclidean, cityblock, mahalanobis, canberra
from scipy.stats import chisquare
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



def compare_histograms(hist1, hist2):

    hist1 = np.array(hist1).flatten()
    hist2 = np.array(hist2).flatten()
    
    distances = {
        "Euclidean": euclidean(hist1, hist2),
    }

    return distances

def put_text(distances, img, name, x_start, y_start):
    text = f'{name}:, Euc={distances["Euclidean"]:.2f} '
    cv2.putText(img, text, (x_start, y_start - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)

image_path = "/Users/tunglambg131003/Real-time_Face_Recognition/test_data/21/lamnt2_5.jpg"  
frame = cv2.imread(image_path)

if frame is None:
    print("Error: Could not load the image.")
    exit()

face_recognizer = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=7, grid_x=7, grid_y=7, threshold=70)
face_recognizer.read('models/trained_on_test4.yml')

face_recognizer2 = cv2.face.LBPHFaceRecognizer.create(radius=1, neighbors=7, grid_x=7, grid_y=7, threshold=70)

LABELS_FILE = 'labels.json'

name = load_labels(LABELS_FILE)

features = np.array(face_recognizer.getHistograms())
size = len(np.array(face_recognizer.getHistograms()))
features = np.reshape(features, (size, -1))
labels = np.array(face_recognizer.getLabels())
labels = labels.ravel()

pca = PCA(n_components=150, random_state=42)
features_pca = pca.fit_transform(features)

def build_annoy_index(dataset, path, num_trees=10):
    """
    Build an Annoy index for approximate nearest neighbor search.
    """
    feature_dim = dataset.shape[1]
    # index = AnnoyIndex(feature_dim, metric='euclidean')
    index = AnnoyIndex(feature_dim, metric='manhattan')
    # index = AnnoyIndex(feature_dim, metric='angular')
    # index = AnnoyIndex(feature_dim, metric='dot')
    # index = AnnoyIndex(feature_dim, metric='hamming')
    for i, vector in enumerate(dataset):
        index.add_item(i, vector)
    index.build(num_trees)
    index.save(path)
    return index

def load_annoy_index(dataset, path):
    annoy = AnnoyIndex(dataset.shape[1], metric='manhattan')
    annoy.load(path)
    return annoy

def lsh_search(query, index, dataset_labels, k=1):
    """
    Perform LSH search using Annoy.
    """
    indices, error = index.get_nns_by_vector(query, k, include_distances=True)
    return dataset_labels[indices[0]], error


annoy_index_pca = build_annoy_index(features_pca, path="models/annoy_index_pca_150")
annoy_index_pca = load_annoy_index(features_pca, path="models/annoy_index_pca_150")

gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
faces_detected = detect_faces_dnn(frame)

x_start, y_start, x_end, y_end = faces_detected[0]
off_set = 15
y_start = y_start + int((y_end-y_start)*0.269) + off_set
roi_gray = gray_img[y_start:y_end, x_start:x_end]
roi_gray = cv2.resize(roi_gray, (300, 300))

face_recognizer2.train(np.array([roi_gray]), np.array(2))
test_instance = face_recognizer2.getHistograms()[0][0]
test_feature = pca.transform([np.array(test_instance)])

predicted_label, error = lsh_search(test_feature[0], annoy_index_pca, labels)
print(predicted_label, error)

# if not faces_detected:
#     print("No faces detected in the image.")
# else:
#     for face in faces_detected:
#         x_start, y_start, x_end, y_end = face

#         if x_end > frame.shape[1] or y_end > frame.shape[0]:
#             continue

#         off_set = 15
#         y_start = y_start + int((y_end - y_start) * 0.269) + off_set

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

#         best_id = min(distances, key=lambda id_: distances[id_]["Euclidean"])
#         predicted_name = name.get(str(face_recognizer.getLabels()[best_id][0]), "Unknown")
#         print(distances)

#         cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
#         put_text(distances[best_id], frame, predicted_name, x_start, y_start)

# cv2.imshow('Face Recognition', frame)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
