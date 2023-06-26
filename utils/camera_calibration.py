import numpy as np
import cv2 as cv
import glob

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 60, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*8,3), np.float32)
objp[:,:2] = np.mgrid[0:8,0:6].T.reshape(-1,2)
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

# 0-direita / 1-esquerda
esquerda_ou_direita = 0
images_d = glob.glob('./imagem/Direita/*.jpg')
images_e = glob.glob('./imagem/Esquerda/*.jpg')
images = images_e if esquerda_ou_direita else images_d
count = 0

result = 'result_pr.jpg' if esquerda_ou_direita else 'result_pb.jpg'
for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
  
    # Find the chess board corners
    chessboard_flags = cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_FAST_CHECK + cv.CALIB_CB_NORMALIZE_IMAGE
    ret, corners = cv.findChessboardCorners(gray, (8,6), None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)
        # Draw and display the corners
        cv.drawChessboardCorners(img, (8,6), corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(500)
        count += 1
cv.destroyAllWindows()
print('Total de imagens: ', count)

# Calibration
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# Undistortion
img_d = cv.imread('./imagem/Direita/frame2.jpg')
img_e = cv.imread('./imagem/Esquerda/frame2.jpg')
img = img_e if esquerda_ou_direita else img_d
h,  w = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

print('img resolution: ', (h,w))

## Método1
# undistort
dst = cv.undistort(img, mtx, dist, None, newcameramtx)

# crop the image
x, y, w, h = roi
#x, y, w, h = (150, 33, 1000, 649)
dst = dst[y:y+h, x:x+w]
h,  w = dst.shape[:2]
#cv.imwrite('./results/'+result, dst)

## Visualização
print('x: ', x)
print('w: ', w)
print('y: ', y)
print('h: ', h)
print('dst resolution: ', dst.shape[:2])
cv.imshow('dst', dst)
while 1:
    if cv.waitKey(1)==ord('q'):
        break

# Re-projection Error 
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
    mean_error += error
print( "total error: {}".format(mean_error/len(objpoints)) )

# Salva a matrix da camera e a matrix de distorções
# if esquerda_ou_direita:
#     np.savetxt('./matrizes/cam_pr_mtx.out', mtx)
#     np.savetxt('./matrizes/cam_pr_dsit.out', dist)   
# else:
#     np.savetxt('./matrizes/cam_pb_mtx.out', mtx)
#     np.savetxt('./matrizes/cam_pb_dsit.out', dist)

print('mtx: ', mtx)
print('dist: ', dist)
print('r: ', rvecs)
print('t: ', tvecs)
