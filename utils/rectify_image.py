import cv2 as cv 
import numpy as np
import matplotlib as plt

##################################################### 5 GRUAS #####################################################
# cameraMatrix1 = np.array([[812.4393, -0.6777, 668.0206],
#                           [0, 812.9133, 373.3885],
#                           [0, 0, 1]])
# distCoeffs1 = np.array([-0.3875, 0.1561, 4.2380e-04, -8.4097e-04])
# cameraMatrix2 = np.array([[801.7487, 0.1210, 622.6017],
#                           [0, 802.4932, 416.6464],
#                           [0, 0, 1]])
# distCoeffs2 = np.array([-0.3976, 0.1652, 0.0020, -6.5162e-04])
imageSize = (1280, 720)
R = np.array([[1, -8.5426e-04, 0.0086],
              [8.1895e-04, 1, 0.0041],
              [-0.0086, -0.0041, 1]])
T = np.array([-354.0020, -1.1359, 1.7502])

##################################################### 15 GRUAS #####################################################
cameraMatrix1 = np.array([[828.1032, 0.8006, 676.2680],
                          [0, 828.9722, 373.6433],
                          [0, 0, 1]])
distCoeffs1 = np.array([-0.4025, 0.1630, 5.6058e-04, -6.5175e-04])
cameraMatrix2 = np.array([[806.7682, 1.1066, 632.5296],
                          [0, 807.3399, 421.4820],
                          [0, 0, 1]])
distCoeffs2 = np.array([-0.4131, 0.1823, 0.0011, -8.2555e-04])
# imageSize = (1280, 720)
# R = np.array([[1, -4.8655e-04, 0.0072],
#               [4.4810e-04, 1, 0.0053],
#               [-0.0072, -0.0053, 1]])
# T = np.array([-353.8582, -1.1266, -0.4277])

#perform the rectification
R1, R2, P1, P2, Q, roi1, roi2 = cv.stereoRectify(cameraMatrix1, distCoeffs1,
                                                 cameraMatrix2, distCoeffs2,
                                                 imageSize,
                                                 R, T, alpha=1.0)
mapx1, mapy1 = cv.initUndistortRectifyMap(cameraMatrix1, distCoeffs1, R1, P1,
                                          imageSize,
                                          cv.CV_32FC2)
mapx2, mapy2 = cv.initUndistortRectifyMap(cameraMatrix2, distCoeffs2, R2, P2,
                                          imageSize,
                                          cv.CV_32FC2)

path1 = '/home/smir/Desktop/imagens/Direita/frame16.jpg'
path2 = '/home/smir/Desktop/imagens/Esquerda/frame16.jpg'
# img1 = cv.imread(path1)
# img2 = cv.imread(path2)

def rectify_frame(img, index):
    try:
        if index==0:
            img_rect = cv.remap(img, mapx1, mapy1, cv.INTER_LINEAR)
        else:
            img_rect = cv.remap(img, mapx2, mapy2, cv.INTER_LINEAR)

        rect = (roi2[0], roi1[1], roi1[2]+55, roi1[3]+88)
        img_rect = img_rect[rect[1]:rect[3], rect[0]:rect[2], :]

    except:
        img_rect = np.zeros((498, 1115, 3))

    # draw the images side by side
    # total_size = (max(img_rect1.shape[0], img_rect2.shape[0]),
    #                 img_rect1.shape[1] + img_rect2.shape[1], 3)
    # img = np.zeros(total_size, dtype=np.uint8)
    # img[:img_rect1.shape[0], :img_rect1.shape[1]] = img_rect1
    # img[:img_rect2.shape[0], img_rect1.shape[1]:] = img_rect2

    # draw horizontal lines every 25 px accross the side by side image
    # for i in range(20, img.shape[0], 30):
    #     cv.line(img, (0, i), (img.shape[1], i), (255, 0, 0))

    # cv.rectangle(img_rect1,rect[:2], rect[2:],(0,255,0))
    # cv.rectangle(img_rect2,rect[:2], rect[2:],(255,0,0))

    # cv.imshow('img', img)
    
    # cv.imshow('img2', img_rect2)
    return img_rect


if __name__ == "__main__":

    cap = cv.VideoCapture("/home/smir/Desktop/Visao Estereo/video/Esquerda/video04_29_13_06.avi")
    while 1:
        _, f = cap.read()
        r = rectify_frame(f)

        cv.imshow("frame", f)
        cv.imshow("rect", r)

        if r.shape!=(498, 1115, 3):
            print(f.shape)
            print(r.shape)

        if cv.waitKey(1)==ord('q'):
            break

cv.destroyAllWindows()