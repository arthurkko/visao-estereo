import cv2 as cv
import numpy as np

def mean_exposure(path):
    #img = cv.imread(path).astype(int)
    img = path
    img = cv.cvtColor(img, cv.COLOR_RGB2HLS).astype(int)
    mean = np.mean(img[:,:,1])

    return mean

def calibrate_exposure(path, ref_mean, img_mean):
    #img = cv.imread(path).astype(int)
    img = path
    new_img = np.zeros(img.shape)
    mask = np.zeros(img.shape)

    img = cv.cvtColor(img, cv.COLOR_RGB2HLS).astype(float)

    img[:,:,1] = img[:,:,1]*(ref_mean/img_mean)
    
    if ref_mean > img_mean:
        mask = img[:,:,1] < 255
        img[:,:,1] = mask*img[:,:,1] + (~mask)*255
    else:
        mask = img[:,:,1] > 0
        img[:,:,1] = mask*img[:,:,1]

    new_img = img.astype("uint8")
    new_img = cv.cvtColor(new_img, cv.COLOR_HLS2RGB)

    return new_img

if __name__=="__main__":
    gravar = 0
    video = cv.VideoCapture("./video/video_d_noite2.avi")

    if gravar:
        fourcc = cv.VideoWriter_fourcc(*'MJPG')
        out = cv.VideoWriter('./video_hls.avi', fourcc, 24.0, (1280, 720))

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