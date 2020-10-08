import numpy as np
import cv2

DEF_HEIGHT = 480
 
# each template defined via a list of white rectangles located within a unit square
# each rectangle given as a quadruple (j, k), (h, w)
HAAR_TEMPLATES = [
    np.array([0.0, 0.0, 1.0, 0.5]), # left-right edge
    np.array([0.0, 0.0, 0.5, 1.0]), # up-down edge
    np.array([0.0, 0.25, 1.0, 0.5]), # left-middle-right edge
    np.array([0.25, 0, 0.5, 1]), # left-middle-right edge
    np.array([[0.0, 0.0, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5]]) # diagonal
]

def resize_image(i):
    width = int(np.round(i.shape[1] * DEF_HEIGHT / (1.0 * i.shape[0])))
    i = cv2.resize(i, (width, DEF_HEIGHT))
    return i

def gray_image(i):
    i = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
    return i

def haar_features_indexes(s, p):
    """
    Generate set of indexes: (t, s_j, s_k, p_j, p_k)

    s - scale
    p - pivots
    """
    indexes = []
    for t in range(len(HAAR_TEMPLATES)):
        for s_j in range(s):
            for s_k in range(s):
                for p_j in range(-p+1, p):
                    for p_k in range(-p+1, p):
                        indexes.append([t, s_j, s_k, p_j, p_k])
    return indexes

def integral_image(i):
    h, w = i.shape
    ii = np.zeros(i.shape, dtype="int32")
    row_ii = np.zeros((w), dtype="int32")
    for j in range(h):
        for k in range(w):
            row_ii[k] = i[j, k]
            if k > 0:
                row_ii[k] += row_ii[k - 1]
            ii[j, k] = row_ii[k]
            if j > 0:
                ii[j, k] += ii[j-1, k]
    return ii


if __name__ == '__main__':
    path = "../data/"
    i0 = cv2.imread(path + "000000.jpg")
    i1 = resize_image(i0)
    i = gray_image(i1)
    cv2.imshow("test image", i)
    cv2.waitKey(0)

    hfs_indexes = haar_features_indexes(4, 5)
    print(len(hfs_indexes))

    ii = integral_image(i)
    print(i[:4, :4])
    print(ii[:4, :4])