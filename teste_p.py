import cv2 as cv
import numpy as np
#import serial
import time
from datetime import datetime
#from serial.tools.list_ports import comports
#from beeprint import pp
import cv2
#from ultralytics import YOLO


if __name__ == "__main__":

    v = cv.VideoCapture('/home/smir/Desktop/Visao Estereo/yolov7/runs/detect/exp/2.avi')
    print(v.get(cv.CAP_PROP_FPS))
    
    while True:
        # Capture frame-by-frame
        img = np.zeros((498, 1115, 3))

        # Display the resulting frame
        cv.imshow('Camera Esquerda', img)
        

        if cv.waitKey(1) == ord('q'):
            break

    # When everything done, release the capture

    cv.destroyAllWindows()

