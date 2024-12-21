import os

# Folder where face data is stored
datasets = 'dataset_crop_test'

# Function to count the number of faces in a dataset
def count_faces_in_dataset(dataset_path):
    face_count = 0
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(".jpg"):  # Check if the file is a PNG image
                face_count += 1
    return face_count

# Function to list faces and their counts
def list_faces():
    print("List of Faces:")
    print("----------------")
    for subdir in os.listdir(datasets):
        person_path = os.path.join(datasets, subdir)
        if os.path.isdir(person_path):  # Check if the path corresponds to a directory
            face_count = count_faces_in_dataset(person_path)
            print(f"{subdir}: {face_count} faces")

# Entry point of the script
if __name__ == "__main__":
    list_faces()  # Call the list_faces function to display the list of faces and counts
