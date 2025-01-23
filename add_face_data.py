from crop_face_data import crop_and_save_faces
import cv2
import os
import sys
import json 

from data_aug import augment_and_process
from labels import load_labels, save_labels
from train_model import labels_for_training_data, load_trained_model, train_classifier, update_classifier

LABELS_FILE = "labels.json"


def capture_images(person_name, datasets_folder='dataset_test', test_folder='test_data'):

    labels = load_labels(LABELS_FILE)
    
    folders = [item for item in os.listdir(datasets_folder) if os.path.isdir(os.path.join(datasets_folder, item))]
    id = str(len(folders))

    labels[id] = person_name
    save_labels(labels, LABELS_FILE)

    person_path = os.path.join(datasets_folder, id)
    os.makedirs(person_path, exist_ok=True)

    # Determine the starting index for new images
    existing_files = [f for f in os.listdir(person_path) if f.endswith('.jpg')]
    count = len(existing_files) + 1

    test_path = os.path.join(test_folder, id)
    os.makedirs(test_path, exist_ok=True)
    existing_fs = [f for f in os.listdir(test_path) if f.endswith('.jpg')]
    count_test = len(existing_fs) + 1
    
    cam = cv2.VideoCapture(1)  # Change the index if your primary webcam is not at index 0
    print("Press Enter to capture a photo, or ESC to quit.")

    while True:
        ret, frame = cam.read()
        if not ret:
            print("Failed to capture image. Please check your camera.")
            break

        cv2.imshow("Capture", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 13:  # Enter key
            file_name = os.path.join(person_path, f"{person_name}_{count}.jpg")
            cv2.imwrite(file_name, frame)
            print(f"Photo #{count} saved at {file_name}")
            count += 1

        if key == 27:  # ESC key
            print("Capture session ended.")
            print("Processing new data...")

            crop_and_save_faces("dataset_test", "dataset_crop_test", id)

            aug_input_dir = "dataset_crop_test"
            aug_output_dir = "new_augmented_dataset_crop_test"
            aug_target_folder = id
 
            target_folder_path = os.path.join(aug_input_dir, aug_target_folder)
            if not os.path.exists(target_folder_path):
                print(f"Target folder {aug_target_folder} does not exist. Exiting.")
            else:
                current_index = 1
                for filename in sorted(os.listdir(target_folder_path)):
                    image_path = os.path.join(target_folder_path, filename)
                    if os.path.isfile(image_path):
                        print(f"Processing image: {image_path}")
                        current_index = augment_and_process(
                             image_path=image_path,
                             output_dir= aug_output_dir,
                             folder= aug_target_folder,
                             current_index=current_index
            )
                        
            face_recognizer = load_trained_model('models/trained_on_test4.yml')
            folder = f"new_augmented_dataset_crop_test/{id}"

            faces, faceID = labels_for_training_data(folder)
            face_recognizer = update_classifier(face_recognizer, faces, faceID)
            
            face_recognizer.save('models/trained_on_test4.yml')

            print("successful")

            break
        
        if key == 32:
            file_name = os.path.join(test_path, f"{person_name}_{count_test}.jpg")
            cv2.imwrite(file_name, frame)
            print(f"Photo #{count_test} saved at {file_name}")
            count_test += 1

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python add_face_data.py <person_name>")
        sys.exit(1)

    person_name = sys.argv[1]
    datasets_folder = 'dataset_test'
    test_folder = 'test_data'

    if not os.path.exists(datasets_folder):
        os.makedirs(datasets_folder)
    if not os.path.exists(test_folder):
        os.makedirs(test_folder)

    capture_images(person_name, datasets_folder, test_folder)
