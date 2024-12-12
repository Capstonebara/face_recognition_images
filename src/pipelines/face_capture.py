import os
import cv2
import sys

# Add the 'src' directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from modules.camera import select_camera, open_camera
from modules.face_detection import process_image_for_face_capture
from modules.face_embedding import process_faces_for_embedding

def capture_and_process_faces():
    face_name = str(input("Enter name of Face:"))
    capture_images_folder = f'captured_images/{face_name}'
    face_images_folder = f'face_images/{face_name}'
    embedded_face_folder = f'embedded_face/{face_name}'

    # Step 1: Open the camera and capture images
    selected_camera = select_camera()
    if selected_camera is not None:
        open_camera(selected_camera, capture_images_folder)  # This function will save captured images to 'captured_images' folder
    
    # Step 2: Process the captured images for face detection and cropping
    process_image_for_face_capture(capture_images_folder, face_images_folder)

    #Step 3: Embededding the images and store to database
    process_faces_for_embedding(face_images_folder, embedded_face_folder)

if __name__ == "__main__":
    capture_and_process_faces()
