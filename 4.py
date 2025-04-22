# Import required libraries
from picamera2 import Picamera2
import cv2
import numpy as np
import face_recognition
import os
import time


# Load reference images and encode faces
def load_reference_image(path):
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Failed to load image: {path}")
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb)
    if not face_locations:
        raise ValueError("No face detected in reference image")
    return face_recognition.face_encodings(rgb, face_locations)[0]


# Initialize camera
def init_camera():
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={
            "size": (640, 480),  # Camera resolution
            "format": "BGR888"
        },
        controls={
            "FrameDurationLimits": (33333, 66666),  # 30fps
            "AwbMode": 0,  # Auto white balance
            "ExposureTime": 20000  # 20ms exposure
        }
    )
    picam2.configure(config)
    picam2.start()
    return picam2


# Save new face image and update known encodings
def save_new_face(frame, face_location, name):
    top, right, bottom, left = face_location
    face_image = frame[top:bottom, left:right]
    # Convert BGR to RGB
    face_image_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
    # Create a directory to save face images if it doesn't exist
    if not os.path.exists('faces'):
        os.makedirs('faces')
    image_path = os.path.join('faces', f'{name}.jpg')
    cv2.imwrite(image_path, face_image_rgb)
    new_encoding = load_reference_image(image_path)
    return new_encoding


# Load all known faces from the faces directory
def load_all_known_faces():
    known_encodings = []
    known_names = []
    if os.path.exists('faces'):
        for filename in os.listdir('faces'):
            if filename.endswith('.jpg'):
                name = os.path.splitext(filename)[0]
                image_path = os.path.join('faces', filename)
                try:
                    encoding = load_reference_image(image_path)
                    known_encodings.append(encoding)
                    known_names.append(name)
                except Exception as e:
                    print(f"Error loading {image_path}: {str(e)}")
    return known_encodings, known_names


# Main program
def main():
    # Load known faces
    try:
        known_encodings, known_names = load_all_known_faces()
    except Exception as e:
        print(f"Initialization failed: {str(e)}")
        return

    # Initialize camera
    camera = init_camera()

    # Display parameters
    SCALE_FACTOR = 0.5
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    THRESHOLD = 0.5

    # Timer settings
    TIMEOUT = 3  # Time in seconds to wait before asking to save
    unknown_start_time = None

    try:
        while True:
            # Capture frame
            frame = camera.capture_array()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Preprocessing
            small_frame = cv2.resize(
                frame_rgb,
                (0, 0),
                fx=SCALE_FACTOR,
                fy=SCALE_FACTOR,
                interpolation=cv2.INTER_AREA
            )

            # Face detection
            face_locations = face_recognition.face_locations(small_frame)
            face_encodings = face_recognition.face_encodings(small_frame, face_locations)

            # Recognition processing
            all_unknown = True
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                # Restore original coordinates
                top = int(top / SCALE_FACTOR)
                right = int(right / SCALE_FACTOR)
                bottom = int(bottom / SCALE_FACTOR)
                left = int(left / SCALE_FACTOR)

                # Calculate matching distance
                distances = face_recognition.face_distance(known_encodings, face_encoding)
                min_distance = np.min(distances)
                match_index = np.argmin(distances)

                # Determine identity
                name = "Unknown"
                color = (0, 0, 255)  # Red
                if min_distance <= THRESHOLD:
                    name = known_names[match_index]
                    color = (0, 255, 0)  # Green
                    all_unknown = False

                # Draw bounding box and label
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                text = f"{name} ({min_distance:.2f})"
                cv2.putText(frame, text, (left + 6, bottom - 6),
                            FONT, 0.5, color, 1)

            # Handle unknown faces with timer
            if all_unknown:
                if unknown_start_time is None:
                    unknown_start_time = time.time()
                elif time.time() - unknown_start_time >= TIMEOUT:
                    # Prompt user to save the new face
                    save_choice = input("Unknown face detected for a while. Do you want to save this face? (y/n): ")
                    if save_choice.lower() == 'y':
                        new_name = input("Enter the name for this person: ")
                        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                            top = int(top / SCALE_FACTOR)
                            right = int(right / SCALE_FACTOR)
                            bottom = int(bottom / SCALE_FACTOR)
                            left = int(left / SCALE_FACTOR)
                            new_encoding = save_new_face(frame, (top, right, bottom, left), new_name)
                            known_encodings.append(new_encoding)
                            known_names.append(new_name)
                            print(f"{new_name}'s face has been saved.")
                    unknown_start_time = None
            else:
                unknown_start_time = None

            # Display output
            cv2.imshow('Face Recognition', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            # Exit condition
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("User terminated program")
                break

    finally:
        # Cleanup resources
        camera.stop()
        camera.close()
        cv2.destroyAllWindows()
        print("System resources released")


if __name__ == "__main__":
    main()
