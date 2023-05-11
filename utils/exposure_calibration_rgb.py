import cv2 as cv
import numpy as np

def mean_exposure(path):
    #img = cv.imread(path).astype(int)
    img = path.astype(int)
    mean = np.mean(img)
    
    return mean

def calibrate_exposure(path, ref_mean, img_mean):
    #img = cv.imread(path).astype(int)
    img = path.astype(float)

    new_img = np.zeros((720, 1280,3))
    mask = np.zeros((720, 1280,3))

    img = img*ref_mean/img_mean
    
    if ref_mean > img_mean:
        mask = img < 255
        img = mask*img + (~mask)*255
    else:
        mask = img > 0
        img = mask*img

    new_img = img.astype("uint8")

    return new_img


gravar = 0
video = cv.VideoCapture("./video/video_d_noite2.avi")

if gravar:
    fourcc = cv.VideoWriter_fourcc(*'MJPG')
    out = cv.VideoWriter('./video_rgb.avi', fourcc, 24.0, (1280, 720))

ref_mean = [70.]

while 1:
    ret, frame = video.read()
    if not ret:
        break

    img_mean = mean_exposure(frame)
    #print("img_mean: ", img_mean)

    new_img = calibrate_exposure(frame, ref_mean, img_mean)
    new_mean = mean_exposure(new_img)
    #print("new_mean: ", new_mean)

    if gravar:
        out.write(new_img)
        
    cv.imshow("frame", new_img)

    if cv.waitKey(1) == ord("q"):
        break

if gravar:
    out.release()
cv.destroyAllWindows()