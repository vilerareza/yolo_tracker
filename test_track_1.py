import cv2 as cv
import numpy as np
from ultralytics import YOLO


# Load an official or custom model
model = YOLO('yolov8n.pt')  # Load an official Detect model
#model = YOLO('yolov8n-seg.pt')  # Load an official Segment model
#model = YOLO('yolov8n-pose.pt')  # Load an official Pose model
#model = YOLO('path/to/best.pt')  # Load a custom trained model

# Model input size
input_size = (640, 640)


# For webcam
# cam = cv.VideoCapture(0)
# For video file
# video_path = 'videos/classroom.mp4'
video_path = 'videos/terrace1-c0.avi'
cap = cv.VideoCapture(video_path)

flip = False


while(True):

    ret, frame = cap.read()

    # Flip
    if flip:
        frame = cv.rotate(frame, cv.ROTATE_180)

    ''' Preprocess '''
    
    # Resize the frame to match the model input size
    frame = cv.resize(frame, input_size).astype('uint8')

    # RGB to BGR
    # frame = frame[:,:,::-1]

    # Run object tracker
    results = model.track(frame, persist=True)
    frame_ = results[0].plot()

    # Display the resulting frame
    cv.imshow('frame', frame_)

    # the 'q' button is set as the
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
# Destroy all the windows
cv.destroyAllWindows()


# Perform tracking with the model
#results = model.track(source="https://youtu.be/LNwODJXcvt4", show=True)  # Tracking with default tracker
#results = model.track(source="https://youtu.be/LNwODJXcvt4", show=True, tracker="bytetrack.yaml")  # Tracking with ByteTrack tracker