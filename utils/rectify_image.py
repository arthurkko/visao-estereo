import cv2 as cv 
import numpy as np
import matplotlib as plt
from numpy.linalg import inv

class CameraParams():
    def __init__(self):

        ##################################################### 5 GRUAS #####################################################
        cameraMatrix1 = np.array([[812.4393, -0.6777, 668.0206],
                                  [0, 812.9133, 373.3885],
                                  [0, 0, 1]])
        distCoeffs1 = np.array([-0.3875, 0.1561, 4.2380e-04, -8.4097e-04])
        cameraMatrix2 = np.array([[801.7487, 0.1210, 622.6017],
                                  [0, 802.4932, 416.6464],
                                  [0, 0, 1]])
        distCoeffs2 = np.array([-0.3976, 0.1652, 0.0020, -6.5162e-04])

        R = np.array([[1, -8.5426e-04, 0.0086],
                    [8.1895e-04, 1, 0.0041],
                    [-0.0086, -0.0041, 1]])
        T = np.array([-354.0020, -1.1359, 1.7502])
        ##################################################### 15 GRUAS #####################################################
        # cameraMatrix1 = np.array([[828.1032, 0.8006, 676.2680],
        #                         [0, 828.9722, 373.6433],
        #                         [0, 0, 1]])
        # distCoeffs1 = np.array([-0.4025, 0.1630, 5.6058e-04, -6.5175e-04])
        # cameraMatrix2 = np.array([[806.7682, 1.1066, 632.5296],
        #                         [0, 807.3399, 421.4820],
        #                         [0, 0, 1]])
        # distCoeffs2 = np.array([-0.4131, 0.1823, 0.0011, -8.2555e-04])
        # R = np.array([[1, -4.8655e-04, 0.0072],
        #               [4.4810e-04, 1, 0.0053],
        #               [-0.0072, -0.0053, 1]])
        # T = np.array([-353.8582, -1.1266, -0.4277])

        imageSize = (1280, 720)

        #perform the rectification
        R1, R2, P1, P2, Q, roi1, roi2 = cv.stereoRectify(cameraMatrix1, distCoeffs1,
                                                        cameraMatrix2, distCoeffs2,
                                                        imageSize,
                                                        R, T, alpha=1.0)
        self.mapx1, self.mapy1 = cv.initUndistortRectifyMap(cameraMatrix1, distCoeffs1, R1, P1,
                                                imageSize,
                                                cv.CV_32FC2)
        self.mapx2, self.mapy2 = cv.initUndistortRectifyMap(cameraMatrix2, distCoeffs2, R2, P2,
                                                imageSize,
                                                cv.CV_32FC2)
        
        self.T = T
        self.D1 = distCoeffs1
        self.D2 = distCoeffs2

        self.fxe = cameraMatrix1[0][0]
        self.fye = cameraMatrix1[1][1]
        self.u0e = cameraMatrix1[0][2]
        self.v0e = cameraMatrix1[1][2]
        self.se = cameraMatrix1[0][1]

        self.fxd = cameraMatrix2[0][0]
        self.fyd = cameraMatrix2[1][1]
        self.u0d = cameraMatrix2[0][2]
        self.v0d = cameraMatrix2[1][2]
        self.sd = cameraMatrix2[0][1]

        self.dx = T[0]
        self.dy = T[1]

        self.roi1 = roi1
        self.roi2 = roi2
            
        Me = np.c_[cameraMatrix1, np.zeros((3,1))]
        Md = np.c_[cameraMatrix2, np.zeros((3,1))]

        H = np.r_[np.c_[np.around(R), T], [np.array([0,0,0,1])]]
     
        self.M = np.dot(Me,H)
       
def rectify_frame(img, index, mapx1, mapy1, mapx2, mapy2):
    try:
        if index==0:
            img_rect = cv.remap(img, mapx1, mapy1, cv.INTER_LINEAR)
        else:
            img_rect = cv.remap(img, mapx2, mapy2, cv.INTER_LINEAR)
  
    except:
        img_rect = np.zeros((498, 1092, 3))

    return img_rect

def crop_frame(img, roi1, roi2):
    rect = (roi2[0], roi1[1], roi1[2]+44, roi2[3]+73)
    # rect = (roi2[0]+10, roi1[1], roi2[2]+57, roi2[3]+73)

    img_crop = img[rect[1]:rect[3], rect[0]:rect[2], :]

    return img_crop

if __name__ == "__main__":
    p = CameraParams()


    cap1 = cv.VideoCapture('/home/smir/Desktop/Visao Estereo/video/Esquerda/video05_15_18_29.avi')
    cap2 = cv.VideoCapture('/home/smir/Desktop/Visao Estereo/video/Direita/video05_15_18_29.avi')
    while 1:
        _, f1 = cap1.read()
        _, f2 = cap2.read()
        img_rect1 = rectify_frame(f1, 0, p.mapx1, p.mapy1, p.mapx2, p.mapy2)
        img_rect2 = rectify_frame(f2, 1, p.mapx1, p.mapy1, p.mapx2, p.mapy2)
        
        # img_rect1 = crop_frame(img_rect1, p.roi1, p.roi2)
        # img_rect2 = crop_frame(img_rect2, p.roi1, p.roi2)
        # print(img_rect2.shape)

        # draw the images side by side
        total_size = (max(img_rect1.shape[0], img_rect2.shape[0]),
                        img_rect1.shape[1] + img_rect2.shape[1], 3)
        img = np.zeros(total_size, dtype=np.uint8)
        img[:img_rect1.shape[0], :img_rect1.shape[1]] = img_rect1
        img[:img_rect2.shape[0], img_rect1.shape[1]:] = img_rect2

        # draw horizontal lines every 25 px accross the side by side image
        for i in range(20, img.shape[0], 30):
            cv.line(img, (0, i), (img.shape[1], i), (255, 0, 0))

        print('roi1: ', p.roi1)
        print('roi2: ', p.roi2)
        print('')

        rect1 = (p.roi1[0], p.roi1[1], p.roi1[2]+44, p.roi2[3]+73)
        rect2 = (p.roi2[0], p.roi1[1], p.roi2[2]+83, p.roi2[3]+73)
        
        cv.rectangle(img_rect1,rect1[:2], rect1[2:],(0,255,0))
        cv.rectangle(img_rect2,rect1[:2], rect1[2:],(255,0,0))

        cv.imshow('img', img)
        cv.imshow('img1', img_rect1)
        cv.imshow('img2', img_rect2)

        if cv.waitKey(1)==ord('q'):
            break

cv.destroyAllWindows()