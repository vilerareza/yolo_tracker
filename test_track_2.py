import cv2 as cv
import numpy as np
from ultralytics import YOLO


# Load an official or custom model
model = YOLO('yolov8n.pt')  # Load an official Detect model
#print (model.model.names)

# Model input size
input_size = (640, 640)

# For webcam
# cam = cv.VideoCapture(0)
# For video file
video_path = 'videos/classroom.mp4'
cap = cv.VideoCapture(video_path)

flip = False

# Run object tracker
results = model.track(video_path, imgsz=input_size, stream=True, persist=True)

for res in results:
    # Display the resulting frame
    frame = res.plot()
    frame = cv.resize(frame, (640, 480)).astype('uint8')
    cv.imshow('frame', frame)

    # the 'q' button is set as the
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cv.destroyAllWindows()