from src import detect_faces, show_bboxes
import numpy as np
import cv2

# Start the video capture
cap = cv2.VideoCapture('/home/juan/Videos/AMOR.mp4')
 
 
while True:             
    # capture next frame
    ret, frame = cap.read()

    frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)

    bounding_boxes, landmarks = detect_faces(frame)

    frame = show_bboxes(frame, bounding_boxes, landmarks)

    # Display the image
    cv2.imshow('frame',frame)
 
    # Read keyboard and exit if ESC was pressed
    k = cv2.waitKey(10) & 0xFF
    if k ==27:
        break
 
# Release resources
cap.release()
cv2.destroyAllWindows()


