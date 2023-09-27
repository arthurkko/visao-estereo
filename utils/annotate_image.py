import numpy as np
import cv2 as cv

def annotate_position(frame_e, frame_d, xywh_e, xywh_d, xyxy_e, xyxy_d):
    cv.putText(frame_e, f'px:{int(xywh_e[0])}',
                (int(xyxy_e[2]-60), int(xyxy_e[1]-25)),
                cv.FONT_HERSHEY_PLAIN, 1, (0,0,255), 2)
    cv.putText(frame_e, f'py:{int(xywh_e[1])}',
                (int(xyxy_e[2]-60), int(xyxy_e[1]-10)),
                cv.FONT_HERSHEY_PLAIN, 1, (0,0,255), 2)
    cv.putText(frame_d, f'px:{int(xywh_d[0])}',
                (int(xyxy_d[2]-60), int(xyxy_d[1]-25)),
                cv.FONT_HERSHEY_PLAIN, 1, (0,0,255), 2)
    cv.putText(frame_d, f'py:{int(xywh_d[1])}',
                (int(xyxy_d[2]-60), int(xyxy_d[1]-10)),
                cv.FONT_HERSHEY_PLAIN, 1, (0,0,255), 2)
    
def annotate_dist(frame_e, xyxy_e, dist, color=(0,0,255)):
    cv.putText(frame_e, f'x:{str(dist[0])}',
                (int(xyxy_e[0]+5), int(xyxy_e[3]-35)),
                cv.FONT_HERSHEY_PLAIN, 1, color, 2)
    cv.putText(frame_e, f'y:{str(dist[1])}',
                (int(xyxy_e[0]+5), int(xyxy_e[3]-20)),
                cv.FONT_HERSHEY_PLAIN, 1, color, 2)
    cv.putText(frame_e, f'z:{str(dist[2])}',
                (int(xyxy_e[0]+5), int(xyxy_e[3]-5)),
                cv.FONT_HERSHEY_PLAIN, 1, color, 2)
    
def annotate_disp(frame_d, xyxy_d, disp):
    cv.putText(frame_d, f'disp:{disp}',
                (int(xyxy_d[0]+5), int(xyxy_d[3]-5)),
                cv.FONT_HERSHEY_PLAIN, 1, (0,0,255), 2)
    
def display(frame_e, frame_d):
    s = frame_e.shape
    crop_e = frame_e[150:s[0], :, :]
    crop_d = frame_d[150:s[0], :, :]
    sc = crop_e.shape
    im = np.zeros((sc[0]*2, sc[1], sc[2]),np.uint8)
    im[:sc[0],:,:] = crop_e
    im[sc[0]:,:,:] = crop_d
    # cv.imshow("Original e", frame_e)
    # cv.imshow("Original d", frame_d)
    cv.imshow("YOLOv8 e", frame_e)
    cv.imshow("YOLOv8 d", frame_d)
    cv.imshow('concat', im)

def save(original_e, original_d, rectified_e, rectified_d, annotated_e, annotated_d):
    cv.imwrite("./imagens para tcc/original_e.png", original_e)
    cv.imwrite("./imagens para tcc/original_d.png", original_d)
    cv.imwrite("./imagens para tcc/retificada_e.png", rectified_e)
    cv.imwrite("./imagens para tcc/retificada_d.png", rectified_d)
    cv.imwrite("./imagens para tcc/anotada_e.png", annotated_e)
    cv.imwrite("./imagens para tcc/anotada_d.png", annotated_d)