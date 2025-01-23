import json 
import os 

def load_labels(file_path):
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            return json.load(f)
    return {}

def save_labels(labels, file_path):
    with open(file_path, "w") as f:
        json.dump(labels, f, indent=4)

