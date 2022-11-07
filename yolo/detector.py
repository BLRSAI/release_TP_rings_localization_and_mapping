# First import the library
import pyrealsense2 as rs

# https://github.com/ultralytics/yolov5/issues/36 Read this if not working
# https://towardsdatascience.com/how-to-train-a-custom-object-detection-model-with-yolo-v5-917e9ce13208
import torch

# Create a context object. This object owns the handles to all connected realsense devices
pipeline = rs.pipeline()
pipeline.start()

# Load yolov5s as a placeholder for object detection
# tune inference settings later
# force_reload ensures we always have the newest version
model = torch.hub.load('ultralytics/yolov5s', 'yolov5s', force_reload=True)

# Load model onto GPU for accelerated prediction
#model.cuda()

try:
    while True:
        # Create a pipeline object. This object configures the streaming camera and owns it's handle
        frames = pipeline.wait_for_frames()
        depth = frames.get_depth_frame()
        if not depth: continue

        # Print a simple text-based representation of the image, by breaking it into 10x20 pixel regions and approximating the coverage of pixels within one meter

        detection_results = model(frames)

        print(detection_results.pandas().xyxy[0])
        # Once purple rings are detected, gather the distances of everything.

        """
        coverage = [0]*64
        for y in xrange(480):
            for x in xrange(640):
                dist = depth.get_distance(x, y)
                if 0 < dist and dist < 1:
                    coverage[x/10] += 1

            if y%20 is 19:
                line = ""
                for c in coverage:
                    line += " .:nhBXWW"[c/25]
                coverage = [0]*64
                print(line)
        """

finally:
    pipeline.stop()
