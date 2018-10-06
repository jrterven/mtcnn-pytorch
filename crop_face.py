from src import detect_faces, show_bboxes
import numpy as np
import cv2
import pysrt

# Load the subtitles
subs = pysrt.open('/home/juan/Videos/AMOR.es.srt')


# Start the video capture
cap = cv2.VideoCapture('/home/juan/Videos/AMOR.mp4')
 
cv2.namedWindow('Original')
cv2.namedWindow('Cropped')

num_subs = len(subs)
print('Num subtitles:', num_subs)
cv2.waitKey(0)

while True:             
    # capture next frame
    ret, frame = cap.read()

    frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
    bounding_boxes, landmarks = detect_faces(frame)
    
    if bounding_boxes.shape[0] == 1:
        bb = bounding_boxes[0]
        x1, y1, x2, y2 = int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])
    
        cropped = frame[y1:y2, x1:x2]
        cv2.imshow('Cropped', cropped)

    frame = show_bboxes(frame, bounding_boxes, landmarks)

    # Display the image
    cv2.imshow('Original',frame)
 
    # Read keyboard and exit if ESC was pressed
    k = cv2.waitKey(10) & 0xFF
    if k ==27:
        break
 
# Release resources
cap.release()
cv2.destroyAllWindows()


