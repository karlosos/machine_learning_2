import numpy as np
import cv2

from haar_students_new import gray_image, integral_image, \
    haar_features_indexes, haar_features_coordinates, unpickle_all
from realboostbins import RealBoostBins  # path for Karol
# from src.realboostbins import RealBoostBins  # path for Wojtas

def fddb_data(path_fddb_root, clf, verbose=False):
    """
    Read all data folds
    :param path_fddb_root: directory with FDDB-folds, 2002 and 2003 folders
    :param clf: classifier object for detection
    :param verbose: if True then it shows images with bounding boxes. Default is False.
    :return: TODO: consider what it should return. Maybe accuracy and other metrics or real and detected bounding boxes
                   for each image?
    """
    fold_paths_train = [
        "FDDB-folds/FDDB-fold-01-ellipseList.txt",
        # "FDDB-folds/FDDB-fold-02-ellipseList.txt",
        # "FDDB-folds/FDDB-fold-03-ellipseList.txt",
        # "FDDB-folds/FDDB-fold-04-ellipseList.txt",
        # "FDDB-folds/FDDB-fold-05-ellipseList.txt",
        # "FDDB-folds/FDDB-fold-06-ellipseList.txt",
        # "FDDB-folds/FDDB-fold-07-ellipseList.txt",
        # "FDDB-folds/FDDB-fold-08-ellipseList.txt",
        # "FDDB-folds/FDDB-fold-09-ellipseList.txt",
        # "FDDB-folds/FDDB-fold-10-ellipseList.txt",
    ]
    for index, fold_path in enumerate(fold_paths_train):
        print("PROCESSING TRAIN FOLD " + str(index + 1) + "/" + str(len(fold_paths_train)) + "...")
        fddb_read_single_fold(path_fddb_root, fold_path, verbose=verbose)


def fddb_read_single_fold(path_root, path_fold_relative, verbose=False):
    """
    Read single fold file and iterate through all images in this fold

    :param path_root: directory with FDDB-folds, 2002 and 2003 folders
    :param path_fold_relative: directory with FDD-fold text files
    :param verbose: if True then it shows images with bounding boxes. Default is False.
    :return: TODO: consider what it should return. Maybe accuracy and other metrics or real and detected bounding boxes
                   for each image?
    """
    f = open(path_root + path_fold_relative, "r")
    line = f.readline().strip()
    n_img = 0
    n_faces = 0
    counter = 0

    """
    For each image store a tiple (real_coords, detected_coords).
    Where real_coords is a list of ground truth coords read from fdbb files
    detected_coords is a list of detected coords with detection function value
    """
    combined_coords = []
    while line is not "":
        file_name = path_root + line + ".jpg"
        log_line = str(counter) + ": [" + file_name + "]"
        print(log_line)
        counter += 1

        i0 = cv2.imread(file_name)
        i = gray_image(i0)
        ii = integral_image(i)
        n_img += 1
        n_img_faces = int(f.readline())
        img_faces_coords = []
        for z in range(n_img_faces):
            r_major, r_minor, angle, center_x, center_y, dummy_one = list(map(float, f.readline().strip().split()))
            w = int(1.5 * r_major)
            j0 = int(center_y - w / 2)
            k0 = int(center_x - w / 2)
            img_face_coords = np.array([j0, k0, j0 + w - 1, k0 + w - 1])
            if j0 < 0 or k0 < 0 or j0 + w - 1 >= i.shape[0] or k0 + w - 1 >= i.shape[1]:
                if verbose:
                    print("WINDOW " + str(img_face_coords) + " OUT OF BOUNDS. [IGNORED]")
                continue
            if w / ii.shape[0] < 0.05:
                if verbose:
                    print("WINDOW " + str(img_face_coords) + " TOO SMALL. [IGNORED]")
                continue
            n_faces += 1
            img_faces_coords.append(img_face_coords)
            if verbose:
                p1 = (k0, j0)
                p2 = (k0 + w - 1, j0 + w - 1)
                cv2.rectangle(i0, p1, p2, (0, 0, 255), 1)
                cv2.imshow("FDDB", i0)
        # TODO: run detect with classifier and compare with img_faces_coors

        if verbose:
            cv2.imshow("FDDB", i0)
            cv2.waitKey(0)
        # TODO: change None to detected_coords
        combined_coords.append((img_faces_coords, None))
        line = f.readline().strip()
    print("IMAGES IN THIS FOLD: " + str(n_img) + ".")
    print("ACCEPTED FACES IN THIS FOLD: " + str(n_faces) + ".")
    print(f"Length of combined_coords: {len(combined_coords)}")
    print(f"First element of combined_coords: {combined_coords[1]}")
    f.close()


def load_classifier():
    path_data_root = "../data/"  # path for Karl
    path_clfs_root = "../clfs/"  # path for Karl
    s = 6
    p = 4
    hfs_indexes = haar_features_indexes(s, p)
    n = len(hfs_indexes)
    print("NO. OF HAAR-LIKE FEATURES: " + str(n))
    data_description = "n_" + str(n) + "_s_" + str(s) + "_p_" + str(p)
    T = 128  # 128 słabych klasyfikatorów
    B = 8  # maximum depth
    clf_description = data_description + "_T_" + str(T) + "_B_" + str(B)
    path_clf = path_clfs_root + "fddb_real_" + clf_description + ".pkl"
    clf = unpickle_all(path_clf)[0]
    return clf


if __name__ == '__main__':
    clf = load_classifier()
    fddb_data("c:/Dev/machine_learning_2/data/", clf=clf, verbose=False)
