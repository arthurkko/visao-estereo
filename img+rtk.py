import cv2 as cv
import serial
import time
from pynmeagps import NMEAReader
from beeprint import pp

delta = {}
i = 1
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

serial_port = '/dev/ttyACM0'
ser = serial.Serial(serial_port, 57600)

with open("./RTK/NMEAOutputs.txt", "a") as outputs:
    while True:
        # Capture frame-by-frame
        t1 = time.time()
        ret_e, frame_e = cap_e.read()
        ret_d, frame_d = cap_d.read()
        msg = NMEAReader(ser) 
        t2 = time.time()


        # if frame is read correctly ret is True
        if not ret_e:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        if not ret_d:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        
        (_, parsed_data) = msg.read()
        pdict = parsed_data.__dict__

        if pdict.get('lat') and pdict.get('lon'):
            delta['frame'+str(i)] = t2-t1
            
            if i!=1:
                f = 1/(t1-t_ant)
                t_ant = t1
            else:
                t_ant = 0
            
            lat = pdict['lat']
            lon = pdict['lon']
            cv.imshow('Camera Esquerda', frame_e)
            cv.imshow('Camera Direita', frame_d)
            path_e = './imagem/SLAM/frame_e'+str(i)+'.jpg'
            path_d = './imagem/SLAM/frame_d'+str(i)+'.jpg'
            cv.imwrite(path_e, frame_e, [cv.IMWRITE_JPEG_QUALITY, 100])
            cv.imwrite(path_d, frame_d, [cv.IMWRITE_JPEG_QUALITY, 100])
            print(path_e,',',path_d,',',lat,',',lon, file=outputs)
            i+=1

        if cv.waitKey(1)==ord('q'):
            break

cv.destroyWindow('Camera Esquerda')
cv.destroyWindow('Camera Direita')
cap_e.release()
cap_d.release()
cv.destroyAllWindows()

pp(delta)
print(str(f)+'Hz')