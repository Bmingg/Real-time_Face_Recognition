import cv2
import os
import time

# Load the pre-trained DNN model
net = cv2.dnn.readNetFromCaffe("ssd/deploy.prototxt.txt", "ssd/res10_300x300_ssd_iter_140000.caffemodel")

# Function to detect faces
def detect_faces_dnn(image):
    h, w = image.shape[:2]
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

# Function to crop and save faces
def crop_faces(image_path, output_folder):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read {image_path}. Skipping.")
        return

    faces = detect_faces_dnn(image)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    for idx, (x1, y1, x2, y2) in enumerate(faces):
        face_crop = image[y1:y2, x1:x2]
        face_output_path = os.path.join(output_folder, f"{base_name}_face{idx}.jpg")
        cv2.imwrite(face_output_path, face_crop)
        print(f"Saved cropped face: {face_output_path}")

# Function to crop faces and save them
def crop_and_save_faces(input_folder, output_folder, target_folder):
    target_folder_path = os.path.join(input_folder, target_folder)
    if not os.path.exists(target_folder_path):
        print(f"Target folder {target_folder} does not exist. Exiting.")
        return

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    output_target_folder_path = os.path.join(output_folder, target_folder)
    if not os.path.exists(output_target_folder_path):
        os.makedirs(output_target_folder_path)

    for image_name in os.listdir(target_folder_path):
        image_path = os.path.join(target_folder_path, image_name)
        output_path = os.path.join(output_target_folder_path, image_name)

        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not read {image_path}. Skipping.")
            continue

        # Assuming detect_faces_dnn is already defined for face detection
        faces = detect_faces_dnn(image)
        for idx, (x1, y1, x2, y2) in enumerate(faces):
            face_crop = image[y1:y2, x1:x2]
            face_output_path = os.path.join(output_target_folder_path, f"{os.path.splitext(image_name)[0]}_face{idx}.jpg")
            cv2.imwrite(face_output_path, face_crop)
            print(f"Saved cropped face: {face_output_path}")
