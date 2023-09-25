# import numpy as np
import cupy as cp

def parse(boxes):
    
    detect = cp.array([[0,0,0,0,0]])
    
    for i in range(len(boxes)):
        box = boxes[i]
        
        area = box[2]*box[3]
        cx = box[0]
        cy = box[1]
        h = box[3]
        w = box[2]

        d = cp.array([[area, cx, cy, h, w]])
        detect = cp.append(detect, d, axis=0)
    
    detect = detect[1:]
   
    return detect

def match_box(boxes1, boxes2):
    pairs = []

    n1 = boxes1.shape[0]
    n2 = boxes2.shape[0]
    
    flag = n1<=n2
    # print('')

    for i1, b in enumerate(boxes1 if flag else boxes2):
        dif = (boxes1 if not flag else boxes2) - b
        sum_abs = cp.sum(cp.abs(dif), axis=1)
        i2 = cp.argmin(sum_abs)
        if flag:
            pairs.append((i1,i2))
        else:
            pairs.append((i2,i1))

    return pairs


if __name__=="__main__":
    print('oi')