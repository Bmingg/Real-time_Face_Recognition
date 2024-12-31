import cv2
import os
import sys

def capture_images(person_name, datasets_folder='datasets'):
    # Create or access the folder for the person
    person_path = os.path.join(datasets_folder, person_name)
    os.makedirs(person_path, exist_ok=True)

    # Determine the starting index for new images
    existing_files = [f for f in os.listdir(person_path) if f.endswith('.jpg')]
    count = len(existing_files) + 1

    cam = cv2.VideoCapture(0)  # Change the index if your primary webcam is not at index 0
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

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python add_face_data.py <person_name>")
        sys.exit(1)

    person_name = sys.argv[1]
    datasets_folder = 'dataset_test'  

    if not os.path.exists(datasets_folder):
        os.makedirs(datasets_folder)

    capture_images(person_name, datasets_folder)
