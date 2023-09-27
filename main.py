import cv2 as cv
from ultralytics import YOLO
import numpy as np
import time 

from utils.rectify_image import rectify_frame, crop_frame, CameraParams
from utils.box_matching import parse, match_box
from utils.world_to_camera import calculate_dist, calculate_disp
from utils.annotate_image import annotate_position, annotate_dist, annotate_disp, display, save

# GRAVAR=0 para parar de gravar
GRAVAR = 0

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# Open the video file
# cap_e = cv.VideoCapture(2)
cap_e = cv.VideoCapture('/home/smir/Desktop/Visao Estereo/video/Esquerda/video05_15_18_29.avi')
if not cap_e.isOpened():
    print("Cannot open camera")
    exit()
cap_e.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
cap_e.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

# cap_d= cv.VideoCapture(4) 
cap_d= cv.VideoCapture('/home/smir/Desktop/Visao Estereo/video/Direita/video05_15_18_29.avi') 
if not cap_d.isOpened():
    print("Cannot open camera")
    exit()
cap_d.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
cap_d.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

if GRAVAR:
    fourcc = cv.VideoWriter_fourcc(*'MJPG')
    annotated_e = cv.VideoWriter('./videos para tcc/annotated_e.avi', fourcc, 10, (1104, 499))
    annotated_d = cv.VideoWriter('./videos para tcc/annotated_d.avi', fourcc, 10, (1104, 499))
    # original_e = cv.VideoWriter('./videos para tcc/original_e.avi', fourcc, 10, (1280, 720))
    # original_d = cv.VideoWriter('./videos para tcc/original_d.avi', fourcc, 10, (1280, 720))
    # rectified_e = cv.VideoWriter('./videos para tcc/rectified_e.avi', fourcc, 10, (1104, 499))
    # rectified_d = cv.VideoWriter('./videos para tcc/rectified_d.avi', fourcc, 10, (1104, 499))
    
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
        # Processo de retifiicação dos frames
        r_e = rectify_frame(frame_e, 0, params.mapx1, params.mapy1, params.mapx2, params.mapy2)
        r_d = rectify_frame(frame_d, 1, params.mapx1, params.mapy1, params.mapx2, params.mapy2)
        r_e = crop_frame(r_e, params.roi1, params.roi2)
        r_d = crop_frame(r_d, params.roi1, params.roi2)
   
        # Detecção de objetos
        results_e = model.track(r_e, imgsz=320, conf=0.35, verbose=False, persist=True)
        results_d = model.track(r_d, imgsz=320, conf=0.35, verbose=False, persist=True)

        # Retorna os resultados em formato numpy
<<<<<<< HEAD
        # res_e = results_e[0].boxes.cpu().numpy()
        # res_d = results_d[0].boxes.cpu().numpy()
=======
        if results_e[0].boxes.id is not None:
            res_e = results_e[0].boxes.cpu().numpy()
            res_d = results_d[0].boxes.cpu().numpy()
     
        # (x_centro, y_centro, largura, altura) de cada objeto normalizado
        xywhn_e = res_e.xywhn
        xywhn_d = res_d.xywhn
>>>>>>> track

        # (x_centro, y_centro, largura, altura) de cada objeto
        xywh_e = res_e.xywh
        xywh_d = res_d.xywh
        
        # xy superior esquerda e xy inferior direita de cada objeto
        xyxy_e = res_e.xyxy
        xyxy_d = res_d.xyxy

        # Cria um vetor de parâmetros único para cada objeto
        match_e = parse(xywhn_e)
        match_d = parse(xywhn_d)
        
        # Relaciona objetos das imagens direita e esquerda [(obj1_e, obj1_d), ...]
        pairs = match_box(match_e, match_d)

        # Anota os resultados nos frames
        annotated_frame_e = results_e[0].plot(labels=False)
        annotated_frame_d = results_d[0].plot(labels=False)
    
        for p in pairs:
            # Calcula a distância dos objetos entre as cameras
            dist = calculate_dist(xywh_e[p[0]], xywh_d[p[1]], params)

            # Calcula a disparidade dos objetos entre as cameras
            disp = calculate_disp(xywh_e[p[0]], xywh_d[p[1]], params)
            
            # write the x and y position of objects in pixel
            annotate_position(
                annotated_frame_e, annotated_frame_d,
                xywh_e[p[0]], xywh_d[p[1]],
                xyxy_e[p[0]], xyxy_d[p[1]]
                )
            
            # write the xyz position of object from the right camera
            annotate_dist(annotated_frame_e, xyxy_e[p[0]], dist)
            
            # write the disparity of objects between cameras
            annotate_disp(annotated_frame_d, xyxy_d[p[1]], disp)
        
        if GRAVAR:
            annotated_e.write(annotated_frame_e)
            annotated_d.write(annotated_frame_d)
            # original_e.write(frame_e)
            # original_d.write(frame_d)
            # rectified_e.write(r_e)
            # rectified_d.write(r_d)
                
        # Display the annotated frame
        display(annotated_frame_e, annotated_frame_d)
        
        
        # Break the loop if 'q' is pressed
        if cv.waitKey(1) & 0xFF == ord("q"):
      
            save(frame_e, frame_d, r_e, r_d, annotated_frame_e, annotated_frame_d)

            break

    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap_e.release()
cap_d.release()

if GRAVAR:
    annotated_e.release()
    annotated_d.release()
    # original_e.release()
    # original_d.release()
    # rectified_e.release()
    # rectified_d.release()

cv.destroyAllWindows()
