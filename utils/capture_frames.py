"""
Código para gravar e fotografar a saída das câmeras do sistema estéreo.

Para ativar a função coloque 1 na variável, caso contrário, 0.
"""

import numpy as np
import cv2 as cv
import time
from datetime import datetime

GRAVAR = 0
FOTO = 0

t1 = time.time()
t2 = time.time()
currentTime = datetime.now().strftime("%m_%d_%H_%M")
i = 1

cap_e = cv.VideoCapture(4)
if not cap_e.isOpened():
    print("Cannot open camera")
    exit()
cap_e.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
cap_e.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

cap_d= cv.VideoCapture(0) 
if not cap_d.isOpened():
    print("Cannot open camera")
    exit()
cap_d.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
cap_d.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

# Define the codec and create VideoWriter object
if GRAVAR:
    fourcc = cv.VideoWriter_fourcc(*'MJPG')
    out_e = cv.VideoWriter('./video/Esquerda/video'+currentTime+'.avi', fourcc, 24.0, (1280, 720))
    out_d = cv.VideoWriter('./video/Direita/video'+currentTime+'.avi', fourcc, 24.0, (1280, 720))

while True:
    # Capture frame-by-frame
    ret_e, frame_e = cap_e.read()
    ret_d, frame_d = cap_d.read()


    # if frame is read correctly ret is True
    if not ret_e:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    if not ret_d:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    
    # write the flipped frame
    if GRAVAR:
        out_e.write(frame_e)
        out_d.write(frame_d)

    # Display the resulting frame
    cv.imshow('Camera Esquerda', frame_e)
    cv.imshow('Camera Direita', frame_d)
    
    t2 = time.time()
    if (t2-t1>=3) and FOTO:
        t1 = time.time()
        frame_name = 'frame' + str(i) + '.jpg'
        i += 1

        #save frame
        cv.imwrite("./imagem/Esquerda/"+frame_name, frame_e, [cv.IMWRITE_JPEG_QUALITY, 100])
        cv.imwrite("./imagem/Direita/"+frame_name, frame_d, [cv.IMWRITE_JPEG_QUALITY, 100])

        cv.destroyWindow('Camera Esquerda')
        cv.destroyWindow('Camera Direita')

    if cv.waitKey(1) == ord('q'):
        break

# When everything done, release the capture
cap_e.release()
cap_d.release()
if gravar:
    out_d.release()
    out_e.release()
cv.destroyAllWindows()