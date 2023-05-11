import cv2 as cv
from ultralytics import YOLO
import numpy as np
import time

from utils.rectify_image import rectify_frame
from utils.box_matching import parse, match_box

gravar = 0
# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# Open the video file
video_path = "video_416_p.avi"
cap_e = cv.VideoCapture(2)
if not cap_e.isOpened():
    print("Cannot open camera")
    exit()
cap_e.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
cap_e.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

cap_d= cv.VideoCapture(4) 
if not cap_d.isOpened():
    print("Cannot open camera")
    exit()
cap_d.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
cap_d.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

if gravar:
    fourcc = cv.VideoWriter_fourcc(*'MJPG')
    out = cv.VideoWriter('./note_416_41fps.avi', fourcc, 24.0, (416, 416))
    

# Loop through the video frames
while cap_e.isOpened():
    # Read a frame from the video
    ret_e, frame_e = cap_e.read()
    ret_d, frame_d = cap_d.read()

    if ret_e:
        # Run YOLOv8 inference on the frame
        r_e = rectify_frame(frame_e, 0)
        r_d = rectify_frame(frame_d, 1)

        results_e = model(r_e, imgsz=416, conf=0.3, verbose=False)
        results_d = model(r_d, imgsz=416, conf=0.3, verbose=False)

        res_e = results_e[0].boxes.cpu().numpy()
        res_d = results_d[0].boxes.cpu().numpy()

        match_e = parse(res_e.xywh)
        match_d = parse(res_d.xywh)
        
        pairs = match_box(match_e, match_d)

        # Visualize the results on the frame
        annotated_frame_e = results_e[0].plot()
        annotated_frame_d = results_d[0].plot()

    
        # Display the annotated frame
        cv.imshow("YOLOv8 e", annotated_frame_e)
        cv.imshow("YOLOv8 d", annotated_frame_d)

        if gravar:
            out.write(annotated_frame_e)
            out.write(annotated_frame_d)
        
        # Break the loop if 'q' is pressed
        if cv.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap_e.release()
# cap_d.release()
if gravar:
    out.release()
cv.destroyAllWindows()
