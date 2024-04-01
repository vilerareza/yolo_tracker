import cv2 as cv
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors


# Load an official or custom model
model = YOLO('yolov8n.pt')  # Load an official Detect model
names = model.model.names

# Model input size
input_size = (640, 640)

# For video file
video_path_1 = 'videos/terrace1-c0.avi'
video_path_2 = 'videos/terrace1-c1.avi'
cap_1 = cv.VideoCapture(video_path_1)
cap_2 = cv.VideoCapture(video_path_2)


flip = False


while(True):

    ret, frame_1 = cap_1.read()
    ret, frame_2 = cap_2.read()

    # Flip
    if flip:
        frame_1 = cv.rotate(frame_1, cv.ROTATE_180)
        frame_2 = cv.rotate(frame_2, cv.ROTATE_180)

    ''' Preprocess '''
    
    # Resize the frame to match the model input size
    frame_1 = cv.resize(frame_1, input_size).astype('uint8')
    frame_2 = cv.resize(frame_2, input_size).astype('uint8')

    # RGB to BGR
    # frame = frame[:,:,::-1]

    # Run object tracker
    results_1 = model(frame_1)
    #results_1 = model.track(frame_1, persist=True)
    #frame_1 = results_1[0].plot()
    results_2 = model(frame_2)
    #results_2 = model.track(frame_2, persist=True)
    #frame_2 = results_2[0].plot()

    # Parsing results
    ## Frame 1
    boxes_1 = results_1[0].boxes.xyxy.cpu().tolist()
    clss_1 = results_1[0].boxes.cls.cpu().tolist()
    confs_1 = results_1[0].boxes.conf.float().cpu().tolist()

    ## Frame 2
    boxes_2 = results_2[0].boxes.xyxy.cpu().tolist()
    clss_2 = results_2[0].boxes.cls.cpu().tolist()
    confs_2 = results_2[0].boxes.conf.float().cpu().tolist()

    annotator_1 = Annotator(frame_1, line_width=2)
    annotator_2 = Annotator(frame_2, line_width=2)

    for box, cls, conf in zip(boxes_1, clss_1, confs_1):
        if cls==0 and conf > 0.8:
            annotator_1.box_label(box, color=colors(int(cls), True), label=names[int(cls)])

    for box, cls, conf in zip(boxes_2, clss_2, confs_2):
        if cls==0 and conf > 0.8:
            annotator_2.box_label(box, color=colors(int(cls), True), label=names[int(cls)])


    # Display the resulting frame
    cv.imshow('frame_1', frame_1)
    cv.imshow('frame_2', frame_2)

    # the 'q' button is set as the
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap_1.release()
cap_2.release()
# Destroy all the windows
cv.destroyAllWindows()