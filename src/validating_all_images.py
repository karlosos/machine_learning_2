import numpy as np
import cv2

from haar_students_new import gray_image, integral_image, \
    haar_features_indexes, haar_features_coordinates, unpickle_all, detect, draw_bounding_boxes
from realboostbins import RealBoostBins  # path for Karol
# from src.realboostbins import RealBoostBins  # path for Wojtas

def fddb_data(path_fddb_root, show_images=False, verbose=False):
    """
    Read all data folds
    :param path_fddb_root: directory with FDDB-folds, 2002 and 2003 folders
    :param show_images: if True then it shows images with bounding boxes. Default is False.
    :return: combined_coords: tuple for each image with (name of file, ground truth bounding boxes,
    detected bounding boxes with decision values (limited to those which decision value > 0))
    """
    clf, hfs_coords_subset, fi, n = load_classifier()

    fold_paths_train = [
        # "FDDB-folds/FDDB-fold-01-ellipseList.txt",
        # "FDDB-folds/FDDB-fold-02-ellipseList.txt",
        # "FDDB-folds/FDDB-fold-03-ellipseList.txt",
        # "FDDB-folds/FDDB-fold-04-ellipseList.txt",
        # "FDDB-folds/FDDB-fold-05-ellipseList.txt",
        # "FDDB-folds/FDDB-fold-06-ellipseList.txt",
        # "FDDB-folds/FDDB-fold-07-ellipseList.txt",
        # "FDDB-folds/FDDB-fold-08-ellipseList.txt",
        # "FDDB-folds/FDDB-fold-09-ellipseList.txt",
        "FDDB-folds/FDDB-fold-10-ellipseList.txt",
    ]

    combined_coords = []

    for index, fold_path in enumerate(fold_paths_train):
        if verbose:
            print("PROCESSING TRAIN FOLD " + str(index + 1) + "/" + str(len(fold_paths_train)) + "...")
        single_fold_coords = fddb_read_single_fold(path_fddb_root, fold_path, clf, hfs_coords_subset, fi, n, show_images=show_images)
        combined_coords = combined_coords + single_fold_coords
        if verbose:
            print("")

    if verbose:
        print(f'FINISHED GETTING COORDS FROM ALL FOLDS')
        print("")
    return combined_coords


def fddb_read_single_fold(path_root, path_fold_relative, clf, hfs_coords_subset, fi, n, verbose=False, show_images=False):
    """
    Read single fold file and iterate through all images in this fold

    :param path_root: directory with FDDB-folds, 2002 and 2003 folders
    :param path_fold_relative: directory with FDD-fold text files
    :param clf: classifier used for detection
    :param hfs_coords_subset: subset of haar features coords
    :param fi: features indexes selected by classifier
    :param n: number of all feature indexes
    :param show_images: if True then it shows images with bounding boxes. Default is False.
    :return: combined_coords: tuple for each image with (name of file, ground truth bounding boxes,
    detected bounding boxes with decision values (limited to those which decision value > 0))
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
        if verbose:
            log_line = str(counter) + ": [" + file_name + "]"
            print(log_line)
        counter += 1

        i0 = cv2.imread(file_name)
        i = gray_image(i0)
        ii = integral_image(i)
        n_img += 1
        n_img_faces = int(f.readline())
        img_faces_coords = []
        # Read ground truth bounding boxes from file
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
            if show_images:
                p1 = (k0, j0)
                p2 = (k0 + w - 1, j0 + w - 1)
                cv2.rectangle(i0, p1, p2, (0, 0, 255), 1)
                cv2.imshow("FDDB", i0)
        # Run detection for image
        detections = detect(i, clf, hfs_coords_subset, n, fi, clf_threshold=3, ii=ii)
        if show_images:
            i0 = draw_bounding_boxes(i0, detections, color=(54, 193, 56), thickness=1)
            cv2.imshow("FDDB", i0)
            cv2.waitKey(0)
        combined_coords.append((file_name, img_faces_coords, detections))
        line = f.readline().strip()
        if counter > 10:
            break
    if verbose:
        print("IMAGES IN THIS FOLD: " + str(n_img) + ".")
        print("ACCEPTED FACES IN THIS FOLD: " + str(n_faces) + ".")
        print(f"Length of combined_coords: {len(combined_coords)}")
        print(f"First element of combined_coords: {combined_coords[0]}")
    f.close()
    return combined_coords


def load_classifier(verbose=False):
    path_data_root = "../data/"  # path for Karl
    path_clfs_root = "../clfs/"  # path for Karl
    s = 6
    p = 4
    hfs_indexes = haar_features_indexes(s, p)
    hfs_coords = haar_features_coordinates(hfs_indexes, s, p)
    n = len(hfs_indexes)
    if verbose:
        print("NO. OF HAAR-LIKE FEATURES: " + str(n))
    data_description = "n_" + str(n) + "_s_" + str(s) + "_p_" + str(p)
    T = 128  # 128 słabych klasyfikatorów
    B = 8  # maximum depth
    clf_description = data_description + "_T_" + str(T) + "_B_" + str(B)
    path_clf = path_clfs_root + "fddb_real_" + clf_description + ".pkl"
    clf = unpickle_all(path_clf)[0]
    fi = np.unique(clf.features_).astype("int32")

    hfs_coords_subset = hfs_coords[fi]
    return clf, hfs_coords_subset, fi, n


if __name__ == '__main__':
    combined_coords = fddb_data("c:/Dev/machine_learning_2/data/", show_images=False)
    print(f'Combined coords length: {len(combined_coords)}')
    print(f'First coords: {combined_coords[0]}')
    print(f'Last coords: {combined_coords[-1]}')
