from src import detect_faces, show_bboxes
import numpy as np
import cv2

img = cv2.imread('/home/juan/Pictures/office1.jpg',cv2.IMREAD_COLOR)

bounding_boxes, landmarks = detect_faces(img)
img = show_bboxes(img, bounding_boxes, landmarks)

cv2.imshow('Image', img)

cv2.waitKey(0)
