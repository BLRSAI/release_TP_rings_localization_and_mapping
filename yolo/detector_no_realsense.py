import torch
import numpy as np
import cv2
import pyrealsense2 as rs

# Load yolov5s as a placeholder for object detection
# tune inference settings later
# force_reload ensures we always have the newest version
model = torch.hub.load('ultralytics/yolov5', 'custom', path='models/best.pt', force_reload=True)
#print(type(model))

#model.cuda()
model.cpu()

#cap = cv2.VideoCapture(0)
pipeline = rs.pipeline()

config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

if device_product_line == "L500":
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)

while(True):
    #ret, frame = cap.read()
    frames = pipeline.wait_for_frames()
    
    frame = np.asanyarray(frames.get_color_frame().get_data())

    #rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)

    #frame = cv2.resize(frame, (416, 416))
    pred = model(frame)

    #pred.print()
    #print(pred.pandas().xyxy[0])
    #print(type(pred.pandas().xyxy[0]))

    for row in pred.pandas().xyxy[0].iterrows():
        #print(row[1]["ymin"])
        pt1 = (int(row[1]["xmin"]), int(row[1]["ymin"]))
        pt2 = (int(row[1]["xmax"]), int(row[1]["ymax"]))
        color = (255, 0, 0)
        thickness = 2
        #print(pt1, pt2)
        frame = cv2.rectangle(frame, pt1, pt2, color, thickness)

    #for row in pred.pandas().xyxy:
     #   print("here" + row)
        #frame = cv2.rectangle(frame, (row[0], row[1]), (row[2], row[3]), (255, 0, 0), 2)

    cv2.imshow('frame', frame)
    cv2.waitKey()
