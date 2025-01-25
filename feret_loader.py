import os
import bz2
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# Define image size for resizing
image_size = (100, 100)

def load_feret_data(base_dir):
    """
    Load and preprocess the FERET dataset.
    Traverses nested directories under `dvd1` and `dvd2`, extracts `.bz2` files, and processes `.ppm` images.

    Args:
        base_dir (str): Base directory containing `dvd1` and `dvd2`.

    Returns:
        images (np.ndarray): Flattened images.
        labels (np.ndarray): Corresponding labels.
        label_map (dict): Mapping of folder names to label indices.
    """
    images = []
    labels = []
    label_map = {}
    label_id = 0

    for dvd_folder in ['dvd1', 'dvd2']:
        dvd_path = os.path.join(base_dir, dvd_folder, 'data', 'images')
        if not os.path.exists(dvd_path):
            print(f"Directory not found: {dvd_path}")
            continue

        # Traverse subject directories (e.g., `00384`, `00153`)
        for subject_folder in os.listdir(dvd_path):
            subject_path = os.path.join(dvd_path, subject_folder)
            if not os.path.isdir(subject_path):
                continue

            for file in os.listdir(subject_path):
                if file.endswith('.bz2'):
                    compressed_file_path = os.path.join(subject_path, file)

                    # Decompress the `.bz2` file
                    try:
                        with bz2.BZ2File(compressed_file_path, 'rb') as f:
                            decompressed_data = f.read()

                        # Decode `.ppm` image
                        img_array = np.frombuffer(decompressed_data, dtype=np.uint8)
                        img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
                        if img is not None:
                            # Resize and flatten the image
                            img = cv2.resize(img, image_size)
                            images.append(img.flatten())

                            # Assign label based on folder name
                            if subject_folder not in label_map:
                                label_map[subject_folder] = label_id
                                label_id += 1
                            labels.append(label_map[subject_folder])
                    except Exception as e:
                        print(f"Error processing file {compressed_file_path}: {e}")

    return np.array(images), np.array(labels), label_map

def evaluate_model(y_test, y_pred):
    """
    Evaluate model performance using precision, FAR, FRR, and other metrics.

    Args:
        y_test (np.ndarray): Ground truth labels.
        y_pred (np.ndarray): Predicted labels.

    Returns:
        dict: Dictionary containing various metrics.
    """
    precision = precision_score(y_test, y_pred, average='weighted')

    cm = confusion_matrix(y_test, y_pred)
    false_accepts = cm.sum(axis=0) - np.diag(cm)
    false_rejects = cm.sum(axis=1) - np.diag(cm)
    total = cm.sum()

    far = false_accepts.sum() / total  # False Accept Rate
    frr = false_rejects.sum() / total  # False Reject Rate

    avg_error = (far + frr) / 2
    fail_rate = (1 - accuracy_score(y_test, y_pred)) * 100

    return {
        "Precision": precision,
        "FAR": far,
        "FRR": frr,
        "Average Error": avg_error,
        "Fail Rate (%)": fail_rate
    }

def eigenfaces_model(X_train, X_test, y_train, y_test, n_components=100):
    """Implements the Eigenfaces model using PCA."""
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    classifier = SVC(kernel="linear", random_state=42)
    classifier.fit(X_train_pca, y_train)
    y_pred = classifier.predict(X_test_pca)

    accuracy = accuracy_score(y_test, y_pred)
    metrics = evaluate_model(y_test, y_pred)
    metrics["Accuracy"] = accuracy

    return metrics, pca

def fisherfaces_model(X_train, X_test, y_train, y_test):
    """Implements the Fisherfaces model using LDA."""
    lda = LDA()
    X_train_lda = lda.fit_transform(X_train, y_train)
    X_test_lda = lda.transform(X_test)

    classifier = SVC(kernel="linear", random_state=42)
    classifier.fit(X_train_lda, y_train)
    y_pred = classifier.predict(X_test_lda)

    accuracy = accuracy_score(y_test, y_pred)
    metrics = evaluate_model(y_test, y_pred)
    metrics["Accuracy"] = accuracy

    return metrics, lda

def cnn_model(X_train, X_test, y_train, y_test, input_shape):
    """Implements a CNN-based facial recognition model."""
    X_train_cnn = X_train.reshape(-1, *input_shape, 1)
    X_test_cnn = X_test.reshape(-1, *input_shape, 1)

    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(*input_shape, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(len(np.unique(y_train)), activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(X_train_cnn, y_train, epochs=10, validation_data=(X_test_cnn, y_test), batch_size=32, verbose=2)
    y_pred = model.predict(X_test_cnn).argmax(axis=1)

    accuracy = accuracy_score(y_test, y_pred)
    metrics = evaluate_model(y_test, y_pred)
    metrics["Accuracy"] = accuracy

    return metrics, model, history

# Path to the FERET dataset
base_dir = "colorferet"

# Load the dataset
images, labels, label_map = load_feret_data(base_dir)
print(f"Loaded {len(images)} images with {len(label_map)} unique labels.")

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Eigenfaces model
eigen_metrics, eigen_pca = eigenfaces_model(X_train, X_test, y_train, y_test)
print("Eigenfaces Metrics:", eigen_metrics)