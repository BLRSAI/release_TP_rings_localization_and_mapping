import torch
import cv2
from grip.detectorfour import GripPipeline

# Load yolov5s as a placeholder for object detection
# tune inference settings later
# force_reload ensures we always have the newest version
model = torch.hub.load('ultralytics/yolov5', 'custom', path='models/best (4).pt', force_reload=True)
#print(type(model))

#model.cuda()
model.cpu()

cap = cv2.VideoCapture(0)
detector = GripPipeline()

while(True):
    ret, frame = cap.read()

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
    #hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    detector.process(frame)

    frame = detector.mask_output

    #frame = cv2.resize(frame, (416, 416))
    pred = model(frame)

    for row in pred.pandas().xyxy[0].iterrows():
        #print(row[1]["ymin"])
        pt1 = (int(row[1]["xmin"]), int(row[1]["ymin"]))
        pt2 = (int(row[1]["xmax"]), int(row[1]["ymax"]))
        color = (255, 0, 0)
        thickness = 2
        frame = cv2.rectangle(frame, pt1, pt2, color, thickness)

    cv2.imshow('frame', frame)
    cv2.waitKey()
