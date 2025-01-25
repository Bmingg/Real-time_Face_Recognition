
import os
import bz2
import cv2
import numpy as np
import face_alignment
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import torch
import time
import json
import random

def load_feret_data_with_unknown(base_dir, image_size=(231, 314)):
    train_images = []
    train_labels = []
    test_known_images = []
    test_known_labels = []
    test_unknown_images = []
    test_unknown_labels = []
    label_map = {}
    label_id = 0

    for dvd_folder in ['dvd1', 'dvd2']:
        dvd_path = os.path.join(base_dir, dvd_folder, 'data', 'images')
        if not os.path.exists(dvd_path):
            print(f"Directory not found: {dvd_path}")
            continue

        # Get all subject folders
        subject_folders = [f for f in os.listdir(dvd_path) if os.path.isdir(os.path.join(dvd_path, f))]
        random.shuffle(subject_folders)

        # Select 10% of subject folders as unknown
        num_unknown = max(1, int(0.1 * len(subject_folders)))
        unknown_folders = subject_folders[:num_unknown]
        known_folders = subject_folders[num_unknown:]

        for subject_folder in subject_folders:
            subject_path = os.path.join(dvd_path, subject_folder)
            images = []

            for file in os.listdir(subject_path):
                if file.endswith('.bz2'):
                    try:
                        compressed_file_path = os.path.join(subject_path, file)
                        with bz2.BZ2File(compressed_file_path, 'rb') as f:
                            decompressed_data = f.read()
                        img_array = np.frombuffer(decompressed_data, dtype=np.uint8)
                        img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
                        if img is not None:
                            img = cv2.resize(img, image_size)
                            images.append(img)
                    except Exception as e:
                        print(f"Error processing file {compressed_file_path}: {e}")

            if subject_folder in unknown_folders:
                # Assign label 9999 for unknown folders
                test_unknown_images.extend(images)
                test_unknown_labels.extend([9999] * len(images))
            else:
                # Assign regular labels for known folders
                if subject_folder not in label_map:
                    label_map[subject_folder] = label_id
                    label_id += 1

                # Split folder data into 90% train (A) and 10% test (B)
                train_split, test_split = train_test_split(images, test_size=0.1, random_state=42, shuffle=True)
                train_images.extend(train_split)
                train_labels.extend([label_map[subject_folder]] * len(train_split))
                test_known_images.extend(test_split)
                test_known_labels.extend([label_map[subject_folder]] * len(test_split))

    return (
        np.array(train_images), np.array(test_known_images), np.array(test_unknown_images),
        np.array(train_labels), np.array(test_known_labels), np.array(test_unknown_labels),
        label_map
    )

train_A, test_B, test_C, train_A_labels, test_B_labels, test_C_labels, label_map = load_feret_data_with_unknown("/Users/tunglambg131003/Downloads/colorferet", image_size=(231, 314))

face_recognizer = cv2.face.LBPHFaceRecognizer_create(
        radius=1,
        neighbors=7,
        grid_x=7,
        grid_y=7,
        threshold=45)

face_recognizer.train(train_A, train_A_labels)
face_recognizer.save("models/model_feret.yml")
