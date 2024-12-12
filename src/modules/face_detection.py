import os
import cv2
from retinaface import RetinaFace

def detect_faces(frame):
    """
    Detect faces in a given frame using RetinaFace.

    Args:
        frame (ndarray): The input image/frame.

    Returns:
        list: A list of bounding boxes [(x1, y1, x2, y2), ...].
    """
    detections = RetinaFace.detect_faces(frame)
    boxes = []
    if isinstance(detections, dict):
        for key in detections.keys():
            box = detections[key]['facial_area']
            boxes.append(box)
    return boxes

def crop_and_save_faces(frame, boxes, image_name, face_images_folder, mode='capture'):
    """
    Crop faces from the frame and save them to 'face_images' folder for capture mode, 
    or return cropped faces for recognition mode.

    Args:
        frame (ndarray): The input image/frame.
        boxes (list): List of bounding boxes [(x1, y1, x2, y2), ...].
        image_name (str): The name of the image for saving cropped faces.
        mode (str): 'capture' to save images, 'recognition' to return cropped faces.
    
    Returns:
        list: Cropped faces if mode is 'recognition', otherwise None.
    """
    os.makedirs(face_images_folder, exist_ok=True)

    cropped_faces = []
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        face = frame[y1:y2, x1:x2]
        
        if mode == 'capture':  # Save faces for face capture mode
            face_image_name = os.path.join(face_images_folder, f"{image_name}.jpg")
            cv2.imwrite(face_image_name, face)
            print(f"Saved{image_name} to {face_image_name}")
        elif mode == 'recognition':  # Return faces for face recognition mode
            cropped_faces.append(face)

    return cropped_faces

def process_image_for_face_capture(capture_images_folder, face_images_folder):
    """
    Process all images in the folder for face capture: detect, crop, and save faces.

    Args:
        images_folder (str): Path to the folder containing images to process.
    """
    # Get list of all images in the captured_images folder
    image_files = [f for f in os.listdir(capture_images_folder) if f.lower().endswith(('jpg', 'jpeg', 'png'))]

    if not image_files:
        print("No images found in the folder.")
        return

    for image_name in image_files:
        image_path = os.path.join(capture_images_folder, image_name)
        
        # Read the image
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Error loading image: {image_path}")
            continue

        print(f"Processing image for face capture: {image_path}")
        
        # Detect faces
        boxes = detect_faces(frame)

        # Crop and save faces for face capture
        if boxes:
            crop_and_save_faces(frame, boxes, os.path.splitext(image_name)[0],face_images_folder, mode='capture')
        else:
            print(f"No faces detected in {image_name}")

def process_image_for_face_recognition(image_path):
    """
    Process a single image for face recognition: detect, draw bounding box, and crop faces.

    Args:
        image_path (str): Path to the image to process.
    
    Returns:
        list: Cropped face images for recognition.
    """
    # Read the image
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error loading image: {image_path}")
        return []

    print(f"Processing image for face recognition: {image_path}")
    
    # Detect faces
    boxes = detect_faces(frame)

    # Draw bounding boxes around detected faces
    for (x1, y1, x2, y2) in boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Show the image with bounding boxes
    cv2.imshow("Detected Faces", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Crop and return faces for recognition
    if boxes:
        return crop_and_save_faces(frame, boxes, os.path.basename(image_path), mode='recognition')
    else:
        print(f"No faces detected in {image_path}")
        return []

def main(pipeline_type, image_path=None):
    """
    Main function to handle face capture or face recognition.

    Args:
        pipeline_type (str): 'capture' for face capture, 'recognition' for face recognition.
        image_path (str): Path to the image to process (used only for recognition).
    """
    if pipeline_type == 'capture':
        process_image_for_face_capture()  # Process all images in captured_images folder for face capture
    elif pipeline_type == 'recognition' and image_path:
        return process_image_for_face_recognition(image_path)  # Process a single image for face recognition
    else:
        print("Invalid pipeline type. Use 'capture' or 'recognition'.")

if __name__ == "__main__":
    # Example: Choose 'capture' or 'recognition' based on pipeline type
    pipeline_type = 'capture'  # Or 'recognition'
    image_path = 'captured_images/sample.jpg'  # Path to the image you want to process for recognition

    main(pipeline_type, image_path)
