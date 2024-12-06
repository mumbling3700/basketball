import cv2
import numpy as np

# Initialize the VideoCapture object to use the default camera
cap = cv2.VideoCapture(0)
cap.set(3, 1920)
cap.set(4, 1080)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Capture video frame by frame
while True:
    ret, frame = cap.read()

    # If frame is read correctly, ret is True
    if not ret:
        print("Error: Can't receive frame (stream end?). Exiting ...")
        break

    # next draw a cross marker for calibration purpose
    x, y = 960, 340
    color = (0, 0, 255)
    markerType = cv2.MARKER_CROSS
    markerSize = 35
    thickness = 3
    cv2.drawMarker(frame, (x, y), color, markerType, markerSize, thickness)

    image = frame[100:600, 600:1320]
    # Display the resulting frame
    cv2.imshow('frame', image)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Release the VideoCapture object
cap.release()
cv2.destroyAllWindows()