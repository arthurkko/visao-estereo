
import numpy as np
from numpy.linalg import pinv
from math import sqrt

def calculate_disp_dist(xywh_e, xywh_d, camera_params):
    
    # Cálculo de disparidade
    disp = xywh_e[0] - xywh_d[0]

    # Fator de Correção
    FC = 1.42

    # Cálculo de distância
    z = 815*(-camera_params.T[0])/(disp*FC)
    x = -z*(552-xywh_d[0])*FC/camera_params.fxd
    y = z*(xywh_d[1]-250)*FC/camera_params.fyd

    dist = np.array([int(x), int(y), int(z)])/1000
    disp = int(disp)

    return disp, dist