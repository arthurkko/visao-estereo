import cv2 as cv
from ultralytics import YOLO
import numpy as np
import time

from utils.rectify_image import rectify_frame, crop_frame, CameraParams
from utils.box_matching import parse, match_box
from utils.world_to_camera import calculate_dist, calculate_disp
gravar = 0

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# Open the video file

# cap_e = cv.VideoCapture(4)
cap_e = cv.VideoCapture('/home/smir/Desktop/Visao Estereo/video/Esquerda/video05_15_18_29.avi')
if not cap_e.isOpened():
    print("Cannot open camera")
    exit()
cap_e.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
cap_e.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

# cap_d= cv.VideoCapture(2) 
cap_d= cv.VideoCapture('/home/smir/Desktop/Visao Estereo/video/Direita/video05_15_18_29.avi') 
if not cap_d.isOpened():
    print("Cannot open camera")
    exit()
cap_d.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
cap_d.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

if gravar:
    fourcc = cv.VideoWriter_fourcc(*'MJPG')
    out = cv.VideoWriter('./note_416_41fps.avi', fourcc, 24.0, (320, 320))
    
params = CameraParams()

t1 = time.time() 
t2 = time.time()
# Loop through the video frames
while cap_e.isOpened():
    # print('fps: ', 1/(t1-t2))
    t2 = t1
    t1 = time.time()

    # Read a frame from the video
    ret_e, frame_e = cap_e.read()
    ret_d, frame_d = cap_d.read()

    if ret_e:
        # Run YOLOv8 inference on the frame
        r_e = rectify_frame(frame_e, 0, params.mapx1, params.mapy1, params.mapx2, params.mapy2)
        r_d = rectify_frame(frame_d, 1, params.mapx1, params.mapy1, params.mapx2, params.mapy2)
        r_e = crop_frame(r_e, params.roi1, params.roi2)
        r_d = crop_frame(r_d, params.roi1, params.roi2)
        # r_e = frame_e
        # r_d = frame_d
        
        # Detecção de objetos
        results_e = model(r_e, imgsz=320, conf=0.35, verbose=False)
        results_d = model(r_d, imgsz=320, conf=0.35, verbose=False)

        res_e = results_e[0].boxes.cpu().numpy()
        res_d = results_d[0].boxes.cpu().numpy()

        # (x_centro, y_centro, largura, altura) de cada objeto
        match_e = parse(res_e.xywhn)
        match_d = parse(res_d.xywhn)
        
        pairs = match_box(match_e, match_d)

        # Visualize the results on the frame
        annotated_frame_e = results_e[0].plot(labels=False)
        annotated_frame_d = results_d[0].plot(labels=False)
 
        # Escolhe um objeto de referência
        # i = 0
        # p = pairs[i]

        # cv.circle(annotated_frame_e,(int(res_e.xywh[p[0]][0]), int(res_e.xywh[p[0]][1])), 10, (0,0,255), 15)
        # cv.circle(annotated_frame_d,(int(res_d.xywh[p[1]][0]), int(res_d.xywh[p[1]][1])), 10, (0,0,255), 15)
        # cv.putText(annotated_frame_e,'dist'+str(i), (int(res_e.xyxy[p[0]][0]+5), int(res_e.xyxy[p[0]][3]-5)), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 5)
        # cv.putText(annotated_frame_d,'dist'+str(i), (int(res_d.xyxy[p[1]][0]+5), int(res_d.xyxy[p[1]][3]-5)), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 5)
       
        
        for p in pairs:
            dist = calculate_dist(res_e.xywh, res_d.xywh, p, params)
            # disp = calculate_disp(res_e.xywh, res_d.xywh, p, params)
            disp = int(res_e.xywh[p[0]][0] - res_d.xywh[p[1]][0])
            
            # cv.circle(annotated_frame_e,(int(res_e.xywh[p[0]][0]), int(res_e.xywh[p[0]][1])), 10, (0,0,255), 15)
            # cv.circle(annotated_frame_d,(int(res_d.xywh[p[1]][0]), int(res_d.xywh[p[1]][1])), 10, (0,0,255), 15)
            cv.putText(annotated_frame_e, f'px:{int(res_e.xywh[p[0]][0])}',
                       (int(res_e.xyxy[p[0]][2]-60), int(res_e.xyxy[p[0]][1]-25)),
                       cv.FONT_HERSHEY_PLAIN, 1, (0,0,255), 2)
            cv.putText(annotated_frame_e, f'py:{int(res_e.xywh[p[0]][1])}',
                       (int(res_e.xyxy[p[0]][2]-60), int(res_e.xyxy[p[0]][1]-10)),
                       cv.FONT_HERSHEY_PLAIN, 1, (0,0,255), 2)
            cv.putText(annotated_frame_d, f'px:{int(res_d.xywh[p[1]][0])}',
                       (int(res_d.xyxy[p[1]][2]-60), int(res_d.xyxy[p[1]][1]-25)),
                       cv.FONT_HERSHEY_PLAIN, 1, (0,0,255), 2)
            cv.putText(annotated_frame_d, f'py:{int(res_d.xywh[p[1]][1])}',
                       (int(res_d.xyxy[p[1]][2]-60), int(res_d.xyxy[p[1]][1]-10)),
                       cv.FONT_HERSHEY_PLAIN, 1, (0,0,255), 2)
            
            cv.putText(annotated_frame_e, f'x:{str(dist[0])}',
                       (int(res_e.xyxy[p[0]][0]+5), int(res_e.xyxy[p[0]][3]-35)),
                       cv.FONT_HERSHEY_PLAIN, 1, (0,0,255), 2)
            cv.putText(annotated_frame_e, f'y:{str(dist[1])}',
                       (int(res_e.xyxy[p[0]][0]+5), int(res_e.xyxy[p[0]][3]-20)),
                       cv.FONT_HERSHEY_PLAIN, 1, (0,0,255), 2)
            cv.putText(annotated_frame_e, f'z:{str(dist[2])}',
                       (int(res_e.xyxy[p[0]][0]+5), int(res_e.xyxy[p[0]][3]-5)),
                       cv.FONT_HERSHEY_PLAIN, 1, (0,0,255), 2)
            cv.putText(annotated_frame_d, f'disp:{disp}',
                       (int(res_d.xyxy[p[1]][0]+5), int(res_d.xyxy[p[1]][3]-5)),
                       cv.FONT_HERSHEY_PLAIN, 1, (0,0,255), 2)
        

        # Markng important points 
        # cv.circle(frame_e, (int(params.u0e), int(params.v0e)), 5, (0,255,0), 10) # (cx, cy) left camera
        # cv.circle(frame_d, (int(params.u0d), int(params.v0d)), 5, (0,255,0), 10) # (cx, cy) right camera
        # cv.circle(annotated_frame_e, (int(params.u0e), int(params.v0e)), 5, (0,255,0), 10) # (cx, cy) left camera
        # cv.circle(annotated_frame_d, (int(params.u0d), int(params.v0d)), 5, (0,255,0), 10) # (cx, cy) right camera
        # cv.circle(annotated_frame_e, (546,242), 5, (0,255,0), 10) # (cx, cy) left camera
        # cv.circle(annotated_frame_d, (546,242), 5, (0,255,0), 10) # (cx, cy) right camera
        
        # Crop rectified images
        # annotated_frame_e = rectify_frame(annotated_frame_e, 0, params.mapx1, params.mapy1, params.mapx2, params.mapy2)
        # annotated_frame_d = rectify_frame(annotated_frame_d, 1, params.mapx1, params.mapy1, params.mapx2, params.mapy2)
        # annotated_frame_e = crop_frame(annotated_frame_e, params.roi1, params.roi2)
        # annotated_frame_d = crop_frame(annotated_frame_d, params.roi1, params.roi2)

        # Display the annotated frame
        s = annotated_frame_e.shape
        crop_e = annotated_frame_e[150:s[0], :, :]
        crop_d = annotated_frame_d[150:s[0], :, :]
        sc = crop_e.shape
        im = np.zeros((sc[0]*2, sc[1], sc[2]),np.uint8)
        im[:sc[0],:,:] = crop_e
        im[sc[0]:,:,:] = crop_d
        # cv.imshow("Original e", frame_e)
        # cv.imshow("Original d", frame_d)
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
            cv.imwrite("./ime.png", annotated_frame_e)
            cv.imwrite("./imd.png", annotated_frame_d)

        
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap_e.release()
cap_d.release()

cv.destroyAllWindows()
