'''
This file is used to run a model on each frame of a video
as a means to test a model's detection capabilities in the
real world.

It takes the video input as the first argument, and video output
as the second argument...

python3 video_benchmark.py video_in/IMG_2203_2.MOV video_out/demo1.mkv
'''
import sys
import os
import torch
import cv2
from grip.detectorfour import GripPipeline

model = torch.hub.load('ultralytics/yolov5', 'custom', path='models/best.pt', force_reload=True)
model.cpu()
detector = GripPipeline()

import re
def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(data, key=alphanum_key)

def main():
    # Initializing necessary variables
    video_input = sys.argv[1]
    video_output = sys.argv[2]
    vidcap = cv2.VideoCapture(video_input)
    success, image = vidcap.read()
    wd = os.getcwd()
    temp_path = os.path.join(wd, "temp")
    count = 0

    while success:
        # Process a frame
        frame = image
        # Run the frame through the GRIP pipeline
        detector.process(frame)
        frame = detector.mask_output
        # Predict and draw bounding boxes from the 
        # predicitons made by the YOLO model
        pred = model(frame)
        for row in pred.pandas().xyxy[0].iterrows():
            pt1 = (int(row[1]["xmin"]), int(row[1]["ymin"]))
            pt2 = (int(row[1]["xmax"]), int(row[1]["ymax"]))
            color = (255, 0, 0)
            thickness = 2
            image = cv2.rectangle(image, pt1, pt2, color, thickness)
        # Frame process end
        # Save frame as JPEG file
        cv2.imwrite(os.path.join(temp_path, "frame%d.jpg" % count), image) 
        count = count + 1
        success, image = vidcap.read()

    # Join all of the processed frames into a video
    images = [img for img in os.listdir(temp_path + "/")]
    images = sorted_alphanumeric(images)[1:]
    frame = cv2.imread(os.path.join(temp_path, images[0]))
    height, width, layers = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_output, fourcc, 30, (width, height))
    for image in images:
        video.write(cv2.imread(os.path.join(temp_path, image)))
    cv2.destroyAllWindows()
    video.release()

    # Remove all processed frames
    dir = temp_path
    for f in os.listdir(dir):
        os.remove(os.path.join(dir, f))

if __name__ == "__main__":
    main()
