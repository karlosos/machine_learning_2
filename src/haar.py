import numpy as np
import cv2

DEF_HEIGHT = 480

def resize_image(i):
    width = int(np.round(i.shape[1] * DEF_HEIGHT / (1.0 * i.shape[0])))
    i = cv2.resize(i, (width, DEF_HEIGHT))
    return i

def gray_image(i):
    i = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
    return i

if __name__ == '__main__':
    path = "../data/"
    i0 = cv2.imread(path + "000000.jpg")
    i1 = resize_image(i0)
    i = gray_image(i1)
    cv2.imshow("test image", i)
    cv2.waitKey(0)
