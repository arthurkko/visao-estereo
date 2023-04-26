import cv2 as cv 
import numpy as np
import matplotlib as plt
import time

cameraMatrix1 = np.array([[828.1032, 0.8006, 676.2680],
                          [0, 828.9722, 373.6433],
                          [0, 0, 1]])
distCoeffs1 = np.array([-0.4025, 0.1630, 5.6058e-04, -6.5175e-04])
cameraMatrix2 = np.array([[806.7682, 1.1066, 632.5296],
                          [0, 807.3399, 421.4820],
                          [0, 0, 1]])
distCoeffs2 = np.array([-0.4131, 0.1823, 0.0011, -8.2555e-04])
imageSize = (1280, 720)
R = np.array([[1, -4.8655e-04, 0.0072],
              [4.4810e-04, 1, 0.0053],
              [-0.0072, -0.0053, 1]])
T = np.array([-353.8582, -1.1266, -0.4277])


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
path1 = '/home/smir/Desktop/Visao Estereo/imagem/Direita/frame13.jpg'
path2 = '/home/smir/Desktop/Visao Estereo/imagem/Esquerda/frame13.jpg'
img1 = cv.imread(path1)
img2 = cv.imread(path2)
t1 = time.time()
img_rect1 = cv.remap(img1, mapx1, mapy1, cv.INTER_LINEAR)
t2 = time.time()
img_rect2 = cv.remap(img2, mapx2, mapy2, cv.INTER_LINEAR)
print("delta: ", t2-t1)
img_rect1 = img_rect1[roi1[1]:roi2[3]+73, roi2[0]+10:roi2[2]+57, :]
img_rect2 = img_rect2[roi1[1]:roi2[3]+73, roi2[0]+10:roi2[2]+57, :]

# draw the images side by side
total_size = (max(img_rect1.shape[0], img_rect2.shape[0]),
                img_rect1.shape[1] + img_rect2.shape[1], 3)
img = np.zeros(total_size, dtype=np.uint8)
img[:img_rect1.shape[0], :img_rect1.shape[1]] = img_rect1
img[:img_rect2.shape[0], img_rect1.shape[1]:] = img_rect2

# draw horizontal lines every 25 px accross the side by side image
for i in range(20, img.shape[0], 20):
    cv.line(img, (0, i), (img.shape[1], i), (255, 0, 0))

cv.imshow('img', img)
cv.imshow('img1', img_rect1)
cv.imshow('img2', img_rect2)

while 1:
    if cv.waitKey(1)==ord('q'):
        break

