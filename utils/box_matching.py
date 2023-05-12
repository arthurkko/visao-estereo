import numpy as np

def parse(boxes):
    detect = np.array([[0,0,0,0,0]])

    for i in range(len(boxes)):
        box = boxes[i]

        area = box[2]*box[3]
        cx = box[0]
        cy = box[1]
        h = box[3]
        w = box[2]

        d = np.array([[area, cx, cy, h, w]])
        detect = np.append(detect, d, axis=0)
    
    detect = np.delete(detect, 0, axis=0)

    return detect

def match_box(boxes1, boxes2):
    pairs = []

    n1 = boxes1.shape[0]
    n2 = boxes2.shape[0]
    
    flag = n1<=n2
    # print('')

    for i1, b in enumerate(boxes1 if flag else boxes2):
        dif = (boxes1 if not flag else boxes2) - b
        sum_abs = np.sum(np.abs(dif), axis=1)
        i2 = np.argmin(sum_abs)
        if flag:
            pairs.append((i1,i2))
        else:
            pairs.append((i2,i1))
        # print(f"box{pairs[-1][0]} of boxes1 is the same of box{pairs[-1][1]} of boxes2")

        if i1==1 and False:
            print(boxes1)
            print(boxes1.shape)
            print(boxes2)
            print(boxes2.shape)
            print('boxes1' if flag else 'boxes2')
            print('b')
            print(b)
            print('dif')
            print(dif)
            print('sum_abs')
            print(sum_abs)
            print(f"box{pairs[i1][0]} of boxes1 is the same of box{pairs[i1][1]} of boxes2")
    
    return pairs


if __name__=="__main__":
    print('oi')