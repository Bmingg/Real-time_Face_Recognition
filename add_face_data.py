import cv2
import os
import sys

def capture_images(person_name, datasets_folder='dataset_test', test_folder='test_data'):
    # Create or access the folder for the person
    folders = [item for item in os.listdir(datasets_folder) if os.path.isdir(os.path.join(datasets_folder, item))]
    id = str(len(folders))
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

        # Save the frame as an image
        if key == 13:  # Enter key
            file_name = os.path.join(person_path, f"{person_name}_{count}.jpg")
            cv2.imwrite(file_name, frame)
            print(f"Photo #{count} saved at {file_name}")
            count += 1

        # Exit the loop if ESC is pressed
        if key == 27:  # ESC key
            print("Capture session ended.")
            break
        
        # Save the frame into the test data if Space is pressed
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
