import cv2
import os

def list_cameras():
    """
    Detect and list all available cameras on the system.
    """
    available_cameras = []
    for i in range(4):  # Check up to 4 camera indices
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cameras.append(i)
            cap.release()
    return available_cameras

def select_camera():
    """
    Display all available cameras for user selection.
    """
    cameras = list_cameras()
    if not cameras:
        print("No cameras detected!")
        return None
    
    print("Available cameras:")
    for idx, cam in enumerate(cameras):
        print(f"{idx}: Camera {cam}")

    while True:
        try:
            choice = int(input(f"Select a camera (0-{len(cameras) - 1}): "))
            if 0 <= choice < len(cameras):
                return cameras[choice]
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def open_camera(camera_index, image_folder):
    """
    Open the selected camera using cv2 and capture 66 frames when 's' is pressed.
    Automatically closes the window after capturing 66 images.
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Unable to open camera {camera_index}")
        return None

    print(f"Camera {camera_index} is now open. Press 's' to capture 10 pictures or 'q' to quit.")
    os.makedirs(image_folder, exist_ok=True)
    
    image_count = 0
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame. Exiting...")
            break

        cv2.imshow("Camera Feed", frame)

        # Capture all 66 frames when 's' is pressed
        if cv2.waitKey(1) & 0xFF == ord('s') and image_count == 0:
            print("Capturing 10 images...")
            for i in range(10):
                ret, frame = cap.read()
                if ret:
                    frames.append(frame)
                    image_path = os.path.join(image_folder, f'image_{i + 1}.jpg')
                    cv2.imwrite(image_path, frame)
                    print(f"Captured image {i + 1}")
                else:
                    print("Error capturing image.")
                    break
            print("Finished capturing 10 images.")
            image_count = 10  # Mark as finished capturing

            # Auto-close the camera feed window
            break  # Exit the while loop

        # Break if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()  # Close all OpenCV windows

if __name__ == "__main__":
    selected_camera = select_camera()
    if selected_camera is not None:
        open_camera(selected_camera)
