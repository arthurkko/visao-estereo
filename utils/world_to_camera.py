import numpy as np
from numpy.linalg import pinv
from math import sqrt

def calculate_dist(boxes1, boxes2, match, camera_params):
    # Outro método de cálculo de distância
    # teta0 = 2.0365
    # B = 0.354
    # x0 = annotated_frame_d.shape[1]
    # x1 = res_e.xywh[p[0]][0] - x0/2
    # x2 = res_d.xywh[p[1]][0] - x0/2
    # D = B*x0/(teta0*(x1-x2))
    # print(D)

    # Método Cabral de cálculo de distância
    # A = np.array([
    #     [params.fxe, 0, -(res_e.xywh[p[0]][0]-params.u0e)],
    #     [params.fxd, 0, -(res_d.xywh[p[1]][0]-params.u0d)],
    #     [0, params.fye, -(res_e.xywh[p[0]][1]-params.v0e)],
    #     [0, params.fyd, -(res_d.xywh[p[1]][1]-params.v0d)]])
    
    # A = np.array([
    #     [camera_params.fxe, 0, -boxes1[match[0]][0]],
    #     [camera_params.fxd, 0, -boxes2[match[1]][0]],
    #     [0, camera_params.fye, -boxes1[match[0]][1]],camera_params.fxd
    #     [0, camera_params.fyd, -boxes2[match[1]][1]]])
    
    # A = np.array([
    #     [camera_params.M[0][0], camera_params.M[0][1], camera_params.M[0][2]-boxes1[match[0]][0]],
    #     [camera_params.M[1][0], camera_params.M[1][1], camera_params.M[1][2]-boxes1[match[0]][1]],
    #     [camera_params.fxd, camera_params.sd, camera_params.u0d-boxes2[match[1]][0]],
    #     [0, camera_params.fyd, camera_params.v0d-boxes2[match[1]][1]]
    # ])
    # A1 = pinv(A)
    # # b = np.array([0, -camera_params.fxd*camera_params.dx, 0, -camera_params.fyd*camera_params.dy])
    # b = np.array([
    #     camera_params.M[2][3]*boxes1[match[0]][0]-camera_params.M[0][3], 
    #     camera_params.M[2][3]*boxes1[match[0]][1]-camera_params.M[1][3], 
    #     0, 0
    # ])
    # dist = np.zeros(3)
    # dist = np.trunc(np.dot(A1,b))

    FC = 1.42

    z = 815*(-camera_params.T[0])/((boxes1[match[0]][0]-boxes2[match[1]][0])*FC)
    x = -z*(552-boxes2[match[1]][0])*FC/camera_params.fxd
    y = z*(boxes2[match[1]][1]-250)*FC/camera_params.fyd
    dist = np.array([int(x), int(y), int(z)])

    return dist

def calculate_disp(boxes1, boxes2, match, camera_params):
    
    # re = (boxes1[match[0]][0]-552)**2+(boxes1[match[0]][1]-249)**2
    # xde = 552 + (boxes1[match[0]][0]-552)*(1+(camera_params.D1[0]/820**2)*re+(camera_params.D1[1]/820**4)*re**2)

    # rd = (boxes2[match[1]][0]-552)**2+(boxes2[match[1]][1]-249)**2
    # xdd = 552 + (boxes2[match[1]][0]-552)*(1+(camera_params.D2[0]/820**2)*rd+(camera_params.D2[1]/820**4)*rd**2)

    # disp = xde - xdd
    # if (boxes1[match[0]][0]>400) and (boxes1[match[0]][0]<420):
    #     print((1+camera_params.D1[0]*re+camera_params.D1[1]*re**2))
    #     print('xde: ', xde)
    #     print('xdd: ', xdd)
    #     print('disp: ', disp)
    #     print('')

    disp = int(boxes1[match[0]][0] - boxes2[match[1]][0])

    return disp