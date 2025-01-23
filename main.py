import cv2
import os
import tkinter as tk
from tkinter import simpledialog
import subprocess

python = "python3"
capturing = False
recognizing = False

def add_faces():
    add_faces_button.config(state=tk.DISABLED)
    recognize_face_button.config(state=tk.DISABLED)
    list_faces_button.config(state=tk.DISABLED)

    person_name = simpledialog.askstring("Input", "Enter the person's name:")
    if not person_name:
        tk.messagebox.showerror("Error", "Please provide a valid name.")
        capturing = False
        return

    subprocess.run([python, "add_face_data.py", person_name])

    add_faces_button.config(state=tk.NORMAL)
    recognize_face_button.config(state=tk.NORMAL)
    list_faces_button.config(state=tk.NORMAL)

def recognize_face():
    add_faces_button.config(state=tk.DISABLED)
    recognize_face_button.config(state=tk.DISABLED)
    list_faces_button.config(state=tk.DISABLED)
    status_label.config(text="Recognizing Faces..., Press ESC to close")
    root.update()  

    subprocess.run([python, "face_recognize.py"])

    add_faces_button.config(state=tk.NORMAL)
    recognize_face_button.config(state=tk.NORMAL)
    list_faces_button.config(state=tk.NORMAL)
    status_label.config(text="")

# Function to list available faces
def list_available_faces():    
    available_faces = subprocess.run([python, "list_faces.py"], capture_output=True, text=True)
    result_window = tk.Toplevel(root)
    result_window.title("Available Face Data")

    # Display the available faces list with appropriate formatting
    result_label = tk.Label(result_window, text=available_faces.stdout, font=("Helvetica", 12))
    result_label.pack(padx=20, pady=20)

    # Add a close button to the result window
    close_button = tk.Button(result_window, text="Close", command=result_window.destroy, padx=20, pady=10)
    close_button.pack(pady=10)

# Create the main application window
root = tk.Tk()
root.title("Face Recognizer")

# Create UI elements: labels, buttons, and status label
title_label = tk.Label(root, text="Face Recognizer", font=("Helvetica", 16, "bold"))
title_label.pack(pady=20)

add_faces_button = tk.Button(root, text="Add Faces", command=add_faces, padx=20, pady=10)
add_faces_button.pack(pady=10)

recognize_face_button = tk.Button(root, text="Recognize Face", command=recognize_face, padx=20, pady=10)
recognize_face_button.pack(pady=10)

list_faces_button = tk.Button(root, text="List Available Faces", command=list_available_faces, padx=20, pady=10)
list_faces_button.pack(pady=10)

status_label = tk.Label(root, text="", font=("Helvetica", 12))
status_label.pack(pady=10)

root.mainloop()
