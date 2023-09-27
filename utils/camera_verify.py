"""
Código básico para verificar a posição da câmera direita e esquerda
"""

import cv2 as cv

cap = cv.VideoCapture(2)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)

while 1:
    ret, f = cap.read()

    cv.imshow("im", f)
    if cv.waitKey(1)==ord('q'):
        break

cap.release()
cv.destroyAllWindows()