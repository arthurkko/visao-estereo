import numpy as np
from numpy.linalg import pinv
from math import sqrt

def calculate_dist(xywh_e, xywh_d, camera_params):

    # Fator de Correção
    FC = 1.42

    z = 815*(-camera_params.T[0])/((xywh_e[0]-xywh_d[0])*FC)
    x = -z*(552-xywh_d[0])*FC/camera_params.fxd
    y = z*(xywh_d[1]-250)*FC/camera_params.fyd
    dist = np.array([int(x), int(y), int(z)])

    return dist

def calculate_disp(xywh_e, xywh_d, camera_params):

    disp = int(xywh_e[0] - xywh_d[0])

    return disp