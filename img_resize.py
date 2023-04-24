import cv2 as cv
import numpy as np

# imagem=0 / video=1
media = 1
gravar = 1

if media:
    video = cv.VideoCapture("./video/Esquerda/video_e_0401.avi")
else:
    frame = cv.imread("./imagem/SLAM/frame_d63.jpg")

if gravar:
    fourcc = cv.VideoWriter_fourcc(*'MJPG')
    out = cv.VideoWriter('./video_640_p.avi', fourcc, 24.0, (640, 640))

while 1:
    if media:
        ret, frame = video.read()
        if not ret:
            break
    
    frame = frame[:,280:1000,:]
    new = cv.resize(frame, (640, 640), 0, 0, cv.INTER_CUBIC)

    cv.imshow('old', frame)
    cv.imshow('new', new)

    if gravar:
        out.write(new)

    if cv.waitKey(1)==ord('q'):
        break

cv.destroyWindow('old')
cv.destroyWindow('new')
if media:
    video.release()
if gravar:
    out.release()
cv.destroyAllWindows()