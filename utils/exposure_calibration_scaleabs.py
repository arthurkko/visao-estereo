import cv2 as cv
import numpy as np

def basicLinearTransform(img):
    brightness = np.mean(img)
    minimum_brightness = 120
    ratio = brightness / minimum_brightness
    if ratio >= 1:
        #rint("Image already bright enough")
        return img

    # Otherwise, adjust brightness to get the target brightness
    return cv.convertScaleAbs(img, alpha = 1 / ratio, beta = 0)

def gammaCorrection(img):
    gamma = 0.92
    ## [changing-contrast-brightness-gamma-correction]
    lookUpTable = np.empty((1,256), np.uint8)
    for i in range(256):
        lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)

    return cv.LUT(img, lookUpTable)
    ## [changing-contrast-brightness-gamma-correction]


if __name__=="__main__":

    gravar = 0
    video = cv.VideoCapture("./video_original.avi")

    if gravar:
        fourcc = cv.VideoWriter_fourcc(*'MJPG')
        out = cv.VideoWriter('./video_abs.avi', fourcc, 24.0, (1280, 720))

    while 1:
        ret, frame = video.read()
        if not ret:
            break

        trans1 = basicLinearTransform(frame)
        trans2 = gammaCorrection(trans1)
        print(np.mean(trans2))
        if gravar:
            out.write(trans2)

        cv.imshow("farme", trans2)
        
        if cv.waitKey(1) == ord("q"):
            break

    if gravar:
        out.release()
    cv.destroyAllWindows()