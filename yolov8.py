import cv2 as cv
from ultralytics import YOLO
import numpy as np
import time
import random

from utils.rectify_image import rectify_frame, CameraParams
from utils.box_matching import parse, match_box

gravar = 0
# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# Open the video file
video_path = "video_416_p.avi"
cap_e = cv.VideoCapture(0)
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
    
params = CameraParams()

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

        match_e = parse(res_e.xywhn)
        match_d = parse(res_d.xywhn)
        
        pairs = match_box(match_e, match_d)

        # Visualize the results on the frame
        annotated_frame_e = results_e[0].plot()
        annotated_frame_d = results_d[0].plot()

        i = 0
        p = pairs[i]
        cv.circle(annotated_frame_e,(int(res_e.xywh[p[0]][0]), int(res_e.xywh[p[0]][1])), 15, (0,0,255), 20)
        cv.circle(annotated_frame_d,(int(res_d.xywh[p[1]][0]), int(res_d.xywh[p[1]][1])), 15, (0,0,255), 20)
        cv.putText(annotated_frame_e,'dist'+str(i), (int(res_e.xyxy[p[0]][0]+5), int(res_e.xyxy[p[0]][3]-5)), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 5)
        cv.putText(annotated_frame_d,'dist'+str(i), (int(res_d.xyxy[p[1]][0]+5), int(res_d.xyxy[p[1]][3]-5)), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 5)
       
        # for i, p in enumerate(pairs):
        #     cv.putText(annotated_frame_e,'dist'+str(i), (int(res_e.xyxy[p[0]][0]+5), int(res_e.xyxy[p[0]][3]-5)), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 5)
        #     cv.putText(annotated_frame_d,'dist'+str(i), (int(res_d.xyxy[p[1]][0]+5), int(res_d.xyxy[p[1]][3]-5)), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 5)    
        # Display the annotated frame
        s = annotated_frame_e.shape
        crop_e = annotated_frame_e[100:s[0]-100, :, :]
        crop_d = annotated_frame_d[100:s[0]-100, :, :]
        sc = crop_e.shape
        im = np.zeros((sc[0]*2, sc[1], sc[2]),np.uint8)
        im[:sc[0],:,:] = crop_e
        im[sc[0]:,:,:] = crop_d
        cv.imshow("YOLOv8 e", annotated_frame_e)
        cv.imshow("YOLOv8 d", annotated_frame_d)
        cv.imshow('concat', im)

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
