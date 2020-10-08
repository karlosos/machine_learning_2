import numpy as np
import cv2

if __name__ == "__main__":
    path = "../data/"
    i = cv2.imread(path + "000000.jpg")
    print(i)

    cv2.imshow("image", i)
    cv2.waitKey(0)
