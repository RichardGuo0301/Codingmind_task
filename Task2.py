# Imports
import mediapipe as mp
from picamera2 import Picamera2
import time
import cv2
import numpy as np

# Initialize the pi camera
pi_camera = Picamera2()
# Convert the color mode to RGB and set a resolution
config = pi_camera.create_preview_configuration(main={"format": "RGB888", "size": (1280, 720)})
pi_camera.configure(config)

# Start the pi camera and give it a second to set up
pi_camera.start()
time.sleep(1)

mp_pose = mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def draw_pose(image, landmarks):
    '''
    Draw circles on the landmarks and lines connecting the landmarks 
    then return the image.
    '''
    # copy the image
    landmark_image = image.copy()
    
    # get the dimensions of the image
    height, width, _ = image.shape

    # Define a list of connections between landmarks based on the provided image
    connections = [
        (0, 1), (0, 4), (1, 2), (2, 3), (3, 7), (4, 5), (5, 6),
        (6, 8), (9, 10), (11, 12), (11, 23), (12, 24), (23, 24),
        (23, 25), (24, 26), (25, 27), (26, 28), (27, 29), (28, 30),
        (29, 31), (30, 32), (9, 13), (13, 15), (15, 17), (17, 19),
        (19, 21), (10, 14), (14, 16), (16, 18), (18, 20), (20, 22)
    ]

    # Draw connections
    for connection in connections:
        start_idx, end_idx = connection
        start_landmark = landmarks.landmark[start_idx]
        end_landmark = landmarks.landmark[end_idx]

        # Convert the landmark positions to image coordinates
        start_point = (int(start_landmark.x * width), int(start_landmark.y * height))
        end_point = (int(end_landmark.x * width), int(end_landmark.y * height))

        # Draw the line
        cv2.line(landmark_image, start_point, end_point, (0, 0, 255), 2)

    # Draw landmarks
    for landmark in landmarks.landmark:
        # Convert the landmark position to image coordinates
        x, y = int(landmark.x * width), int(landmark.y * height)

        # Draw the landmark
        cv2.circle(landmark_image, (x, y), 5, (0, 255, 0), -1)

    return landmark_image

def main():
    # Wait for the camera to start
    time.sleep(2)
    
    # Capture the image from the pi camera
    buffer = pi_camera.capture_array()
    image = cv2.cvtColor(buffer, cv2.COLOR_RGB2BGR)
    
    # Process the image and detect the pose
    with mp_pose as pose:
        # To improve performance, mark the image as not writeable to pass by reference.
        image.flags.writeable = False
        results = pose.process(image)
        
        # Draw the pose annotations on the image.
        if results.pose_landmarks:
            result_image = draw_pose(image, results.pose_landmarks)
            cv2.imwrite('/home/pi/RaspberryPiBoostrap/output.png', result_image)
            cv2.imshow('Pose', result_image)
            cv2.waitKey(0)  # Wait for a key press to exit
            cv2.destroyAllWindows()
        else:
            print('No Pose Detected')

    # Stop the camera after capturing the image
    pi_camera.stop()

if __name__ == "__main__":
    main()
    print('done')
