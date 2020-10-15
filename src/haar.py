import numpy as np
import cv2

DEF_HEIGHT = 480

# each template defined via a list of white rectangles
# located within a unit square
# each rectangle given as a quadruple (j, k), (h, w)
HAAR_TEMPLATES = [
    np.array([[0.0, 0.0, 1.0, 0.5]]),  # left-right edge
    np.array([[0.0, 0.0, 0.5, 1.0]]),  # up-down edge
    np.array([[0.0, 0.25, 1.0, 0.5]]),  # left-middle-right edge
    np.array([[0.25, 0, 0.5, 1]]),  # left-middle-right edge
    np.array([[0.0, 0.0, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5]]),  # diagonal
]

F_MIN = 0.2
F_MAX = 0.5


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

    s - number of scales
    p - number pivots, latice of pivots
    """
    indexes = []
    for t in range(len(HAAR_TEMPLATES)):
        for s_j in range(s):
            for s_k in range(s):
                for p_j in range(-p + 1, p):
                    for p_k in range(-p + 1, p):
                        indexes.append([t, s_j, s_k, p_j, p_k])
    return indexes


def integral_image(i):
    # h, w = i.shape
    # ii = np.zeros(i.shape, dtype="int32")
    # row_ii = np.zeros((w), dtype="int32")
    # for j in range(h):
    #     for k in range(w):
    #         row_ii[k] = i[j, k]
    #         if k > 0:
    #             row_ii[k] += row_ii[k - 1]
    #         ii[j, k] = row_ii[k]
    #         if j > 0:
    #             ii[j, k] += ii[j-1, k]
    ii = np.apply_over_axes(np.cumsum, i, axes=[0, 1])
    ii = ii.astype("int32")
    return ii


def delta(ii, j1, k1, j2, k2):
    """
    Calculate cumsum over rectangle
    j1 - start position row
    k1 - start position column
    j2 - end position row
    k2 - end position column
    """
    d = ii[j2, k2]
    if j1 > 0:
        d -= ii[j1 - 1, k2]
    if k1 > 0:
        d -= ii[j2, k1 - 1]
    if j1 > 0 and k1 > 0:
        d += ii[j1 - 1, k1 - 1]
    return d


def haar_features_coordinates(hfs_indexes, s, p):
    """
    Generates coordinates of features within unit square
    """
    F_SIZE_JUMP = (F_MAX - F_MIN) / (s - 1) if s > 1 else F_MAX
    jump_denominator = 2 * p - 2

    hfs_coords = []
    for t, s_j, s_k, p_j, p_k in hfs_indexes:
        f_h = F_MIN + s_j * F_SIZE_JUMP
        f_w = F_MIN + s_k * F_SIZE_JUMP
        jump_j = (1.0 - f_h) / jump_denominator
        jump_k = (1.0 - f_w) / jump_denominator
        offset_j = 0.5 + p_j * jump_j - 0.5 * f_h
        offset_k = 0.5 + p_k * jump_k - 0.5 * f_w
        hf_coords = [np.array([offset_j, offset_k, f_h, f_w])]
        for white in HAAR_TEMPLATES[t]:
            white_j = offset_j + white[0] * f_h
            white_k = offset_k + white[1] * f_w
            white_h = white[2] * f_h
            white_w = white[3] * f_w
            hf_coords.append(np.array([white_j, white_k, white_h, white_w]))
        hfs_coords.append(np.array(hf_coords))
    return np.array(hfs_coords, dtype='object')


def haar_feature(ii, hf_coords_window, j0, k0):
    """
    Calculate difference between area under squares
    """
    j, k, h, w = hf_coords_window[0]
    j1 = int(j0 + j)
    k1 = int(k0 + k)
    j2 = int(j1 + h - 1)
    k2 = int(k1 + w - 1)
    sum_all = delta(ii, j1, k1, j2, k2)
    area_all = h * w
    sum_white = 0
    area_white = 0
    for j, k, h, w in hf_coords_window[1:]:
        j1 = int(j0 + j)
        k1 = int(k0 + k)
        j2 = int(j1 + h - 1)
        k2 = int(k1 + w - 1)
        sum_white += delta(ii, j1, k1, j2, k2)
        area_white += h * w
    sum_black = sum_all - sum_white
    area_black = area_all - area_white
    return int(sum_white / area_white - sum_black / area_black)


def draw_haar_feature_at(i, hf_coords, j0, k0):
    j, k, h, w = hf_coords[0]
    j1 = round(j0 + j)
    j2 = round(j1 + h - 1)
    k1 = round(k0 + k)
    k2 = round(k1 + w - 1)

    i_copy = i.copy()
    cv2.rectangle(i_copy, (k1, j1), (k2, j2), (0, 0, 0), cv2.FILLED)
    for c in hf_coords[1:]:
        j, k, h, w = c
        j1 = round(j0 + j)
        j2 = round(j1 + h - 1)
        k1 = round(k0 + k)
        k2 = round(k1 + w - 1)
        cv2.rectangle(i_copy, (k1, j1), (k2, j2), (255, 255, 255), cv2.FILLED)

    cv2.addWeighted(i_copy, 0.6, i, 0.4, 0.0, i_copy)
    return i_copy


if __name__ == "__main__":
    path = "./data/"
    i0 = cv2.imread(path + "000000.jpg")
    i1 = resize_image(i0)
    i = gray_image(i1)
    # cv2.imshow("test image", i)
    ii = integral_image(i)

    s = 2
    p = 2

    hfs_indexes = haar_features_indexes(s, p)
    print(len(hfs_indexes))

    hfs_coords = haar_features_coordinates(hfs_indexes, s=s, p=p)
    h = 68
    w = 68
    hfs_coords_window = w * hfs_coords

    # features demonstration on image and example window
    j0 = 160
    k0 = 280
    p1 = (k0, j0)
    p2 = (k0 + w - 1, j0 + h - 1)
    for hf_coords_window, index in zip(hfs_coords_window, hfs_indexes):
        i2 = draw_haar_feature_at(i1, hf_coords_window, j0, k0)
        cv2.rectangle(i2, p1, p2, (0, 0, 255), 1)
        feature = haar_feature(ii, hf_coords_window, j0, k0)
        print(f'{index}: {feature}')
        cv2.imshow("FEATURE DEMO", i2)
        cv2.waitKey(0)

    cv2.waitKey(0)
