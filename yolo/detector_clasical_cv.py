import cv2
import numpy as np
from grip.detectorthree import GripPipeline

cap = cv2.VideoCapture(0)

j=0

detector = GripPipeline()

while j < 4:
    ret, frame = cap.read()

    #hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    detector.process(frame)

    cv2.imshow('im2', detector.mask_output)
    cv2.waitKey()
