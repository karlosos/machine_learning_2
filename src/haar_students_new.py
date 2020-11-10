import numpy as np
import cv2
import time
import pickle
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from numba import jit
from realboostbins import RealBoostBins

DEF_HEIGHT = 480

DETECTION_SCALES = 4
DETECTION_W_MIN = 64
DETECTION_WINDOW_GROWTH = 1.2
DETECTION_WINDOW_JUMP = 0.1

# each template defined via a numpy array of white rectangles (written as rows) located within a unit square,
# each rectangle given as a quadruple (j, k), (h, w)
HAAR_TEMPLATES = [
        np.array([[0.0, 0.0, 1.0, 0.5]]), # left-right edge
        np.array([[0.0, 0.0, 0.5, 1.0]]), # top-down edge
        np.array([[0.0, 0.25, 1.0, 0.5]]), # left-middle-right edge
        np.array([[0.25, 0.0, 0.5, 1.0]]), # top-middle-down edge
        np.array([[0.0, 0.0, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5]]) # diagonal
    ]

HAAR_FEATURE_SIZE_RELATIVE_MIN = 0.25
HAAR_FEATURE_SIZE_RELATIVE_MAX = 0.5

def resize_image(i):
    width = int(np.round(i.shape[1] * DEF_HEIGHT / (1.0 * i.shape[0])))
    i = cv2.resize(i, (width, DEF_HEIGHT))
    return i

def gray_image(i):
    i = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
    return i

def haar_features_indexes(s, p):    
    indexes = []    
    p_range = range(-(p - 1), p)
    for t in range(len(HAAR_TEMPLATES)):
        for s_j in range(s):
            for s_k in range(s):
                for p_j in p_range:
                    for p_k in p_range:
                        indexes.append([t, s_j, s_k, p_j, p_k])                     
    return np.array(indexes)

def adaboost_features_indexes(n, clf):
    flags = np.zeros(n)
    for e in clf.estimators_:
        flags[np.where(e.feature_importances_ > 0)[0]] = 1
    return np.where(flags == 1)[0]

def integral_image(i):
#     h, w = i.shape
#     ii = np.zeros(i.shape, dtype=int)
#     ii_row = np.zeros(w)
#     for j in range(h):        
#         for k in range(w):
#             ii_row[k] = i[j, k]
#             if k > 0:
#                 ii_row[k] += ii_row[k - 1]
#             ii[j, k] = ii_row[k]
#             if j > 0:
#                 ii[j, k] += ii[j - 1, k]
#     return ii 
    ii = np.apply_over_axes(np.cumsum, i, axes=[0, 1])
    ii = ii.astype("int32") # without this line uint32 was assumed and underflow errors were alarmed by delta(...) function
    return ii

@jit(nopython=True)
def delta(ii, j1, k1, j2, k2):
    d = ii[j2, k2]
    if j1 > 0:
        d -= ii[j1 - 1, k2]
    if k1 > 0:
        d -= ii[j2, k1 - 1]
    if k1 > 0 and j1 > 0:
        d += ii[j1 - 1, k1 - 1]
    return d

def haar_features_coordinates(hfs_indexes, s, p):
    f_size_relative_jump = (HAAR_FEATURE_SIZE_RELATIVE_MAX - HAAR_FEATURE_SIZE_RELATIVE_MIN) / (s - 1) if s > 1 else 0.0
    grid_jump_denominator = 1.0 * (2 * p - 2)    
    c_j = 0.5
    c_k = 0.5
    hfs_coords = []
    for index in hfs_indexes:
        t, s_j, s_k, p_j, p_k = index
        f_h = HAAR_FEATURE_SIZE_RELATIVE_MIN + s_j * f_size_relative_jump
        f_w = HAAR_FEATURE_SIZE_RELATIVE_MIN + s_k * f_size_relative_jump
        p_jump_j = (1.0 - f_h) / grid_jump_denominator 
        p_jump_k = (1.0 - f_w) / grid_jump_denominator
        offset_j = c_j + p_j * p_jump_j - 0.5 * f_h
        offset_k = c_k + p_k * p_jump_k - 0.5 * f_w
        hf_coords = [np.array([offset_j, offset_k, f_h, f_w])]
        for white in HAAR_TEMPLATES[t]:
            white_r_j, white_r_k, white_r_h, white_r_w = white
            white_j = offset_j + white_r_j * f_h
            white_k = offset_k + white_r_k * f_w 
            white_h = white_r_h * f_h
            white_w = white_r_w * f_w
            hf_coords.append(np.array([white_j, white_k, white_h, white_w]))        
        hfs_coords.append(np.array(hf_coords)) 
    return np.array(hfs_coords)    

# Comment: all int(...) operations removed, assuming they were performed earlier outside i.e. hf_coords_window already as ints
@jit(nopython=True)  
def haar_feature(ii, hf_coords_window, j0, k0):
    j, k, h, w = hf_coords_window[0]
    j1 = j0 + j
    j2 = j1 + h - 1
    k1 = k0 + k
    k2 = k1 + w - 1 
    sum_all = delta(ii, j1, k1, j2, k2)
    area_all = h * w
    sum_white = 0
    area_white = 0        
    for c in hf_coords_window[1:]:
        j, k, h, w = c
        j1 = j0 + j
        j2 = j1 + h - 1
        k1 = k0 + k
        k2 = k1 + w - 1
        sum_white += delta(ii, j1, k1, j2, k2)
        area_white += h * w
    sum_black = sum_all - sum_white
    area_black = area_all - area_white    
    return int(sum_white / area_white - sum_black / area_black)
    
def haar_features(ii, hfs_coords_window_subset, j0, k0, n, fi=None):
    if fi is None:
        fi = np.arange(n)
    features = np.zeros(n, "int32")
    for i, hf_coords_window in enumerate(hfs_coords_window_subset):
        features[fi[i]] = haar_feature(ii, hf_coords_window, j0, k0)
    return features            

def haar_features_2(ii, hfs_coords_window_subset, j0, k0, n, fi, features):
    for i, hf_coords_window in enumerate(hfs_coords_window_subset):
        features[fi[i]] = haar_feature(ii, hf_coords_window, j0, k0)
    return features            
    
def draw_haar_feature_at(i, hf_coords, j0, k0):    
    j, k, h, w = hf_coords[0]
    j1 = int(j0 + j) 
    j2 = int(j1 + h - 1)
    k1 = int(k0 + k)
    k2 = int(k1 + w - 1)
    
    i_copy = i.copy()
    cv2.rectangle(i_copy, (k1, j1), (k2, j2), (0, 0, 0), cv2.FILLED)
    for c in hf_coords[1:]:
        j, k, h, w = c
        j1 = int(j0 + j) 
        j2 = int(j1 + h - 1)
        k1 = int(k0 + k)
        k2 = int(k1 + w - 1)        
        cv2.rectangle(i_copy, (k1, j1), (k2, j2), (255, 255, 255), cv2.FILLED)
    cv2.addWeighted(i_copy, 0.6, i, 0.4, 0.0, i_copy)
    return i_copy

def iou(coords_1, coords_2):
    j11, k11, j12, k12 = coords_1
    j21, k21, j22, k22 = coords_2    
    dj = np.min([j12, j22]) - np.max([j21, j11]) + 1 
    if dj <= 0: 
        return 0.0
    dk = np.min([k12, k22]) - np.max([k21, k11]) + 1
    if dk <= 0: 
        return 0.0
    i = dj * dk
    u = (j12 - j11 + 1) * (k12 - k11 + 1) + (j22 - j21 + 1) * (k22 - k21 + 1) - i
    return i / u 

def fddb_read_single_fold(path_root, path_fold_relative, n_negs_per_img, hfs_coords, n, verbose=False, fold_title=""):
    np.random.seed(1)    
    
    # settings for sampling negatives
    w_relative_min = 0.1
    w_relative_max = 0.35
    w_relative_spread = w_relative_max - w_relative_min
    neg_max_iou = 0.5
    
    X_list = []
    y_list = []
    
    f = open(path_root + path_fold_relative, "r")
    line = f.readline().strip()
    n_img = 0
    n_faces = 0
    counter = 0
    while line is not "":
        file_name = path_root + line + ".jpg"
        log_line =  str(counter) + ": [" + file_name + "]"
        if fold_title is not "":
            log_line += " [" + fold_title + "]" 
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
            if (w / ii.shape[0] < 0.05): # min relative size of positive window (smaller may lead to division by zero when white regions in haar features have no area)
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
                cv2.waitKey(0)            
            hfs_coords_window = w * hfs_coords
            hfs_coords_window = np.array(list(map(lambda npa: npa.astype("int32") , hfs_coords_window)))            
            feats = haar_features(ii, hfs_coords_window, j0, k0, n)
            if verbose:
                print("POSITIVE WINDOW " + str(img_face_coords) + " ACCEPTED. FEATURES: " + str(feats) + ".") 
            X_list.append(feats)
            y_list.append(1)
        if verbose:      
            cv2.imshow("FDDB", i0)
            cv2.waitKey(0)
        for z in range(n_negs_per_img):
            while True:
                w = int((np.random.random() * w_relative_spread + w_relative_min) * i.shape[0])
                j0 = int(np.random.random() * (i.shape[0] - w + 1))
                k0 = int(np.random.random() * (i.shape[1] - w + 1))                 
                patch = np.array([j0, k0, j0 + w - 1, k0 + w - 1])
                ious = list(map(lambda ifc : iou(patch, ifc), img_faces_coords))
                max_iou = max(ious) if len(ious) > 0 else 0.0
                if max_iou < neg_max_iou:
                    hfs_coords_window = w * hfs_coords
                    hfs_coords_window = np.array(list(map(lambda npa: npa.astype("int32") , hfs_coords_window)))
                    feats = haar_features(ii, hfs_coords_window, j0, k0, n)
                    X_list.append(feats)
                    y_list.append(-1)                    
                    if verbose:
                        print("NEGATIVE WINDOW " + str(patch) + " ACCEPTED. FEATURES: " + str(feats) + ".")
                        p1 = (k0, j0)
                        p2 = (k0 + w - 1, j0 + w - 1)            
                        cv2.rectangle(i0, p1, p2, (0, 255, 0), 1)
                    break
                else:                    
                    if verbose:
                        print("NEGATIVE WINDOW " + str(patch) + " IGNORED. [MAX IOU: " + str(max_iou) + "]")
                        p1 = (k0, j0)
                        p2 = (k0 + w - 1, j0 + w - 1)
                        cv2.rectangle(i0, p1, p2, (255, 255, 0), 1)
        if verbose: 
            cv2.imshow("FDDB", i0)
            cv2.waitKey(0)
        line = f.readline().strip()
    print("IMAGES IN THIS FOLD: " + str(n_img) + ".")
    print("ACCEPTED FACES IN THIS FOLD: " + str(n_faces) + ".")
    f.close()
    X = np.stack(X_list)
    y = np.stack(y_list)
    return X, y

def fddb_data(path_fddb_root, hfs_coords, n_negs_per_img, n):
    n_negs_per_img = n_negs_per_img
       
    fold_paths_train = [
        "FDDB-folds/FDDB-fold-01-ellipseList.txt",
        "FDDB-folds/FDDB-fold-02-ellipseList.txt",
        "FDDB-folds/FDDB-fold-03-ellipseList.txt",
        "FDDB-folds/FDDB-fold-04-ellipseList.txt",
        "FDDB-folds/FDDB-fold-05-ellipseList.txt",
        "FDDB-folds/FDDB-fold-06-ellipseList.txt",
        "FDDB-folds/FDDB-fold-07-ellipseList.txt",
        "FDDB-folds/FDDB-fold-08-ellipseList.txt",
        "FDDB-folds/FDDB-fold-09-ellipseList.txt", 
        ] 
    X_train = None 
    y_train = None
    for index, fold_path in enumerate(fold_paths_train):
        print("PROCESSING TRAIN FOLD " + str(index + 1) + "/" + str(len(fold_paths_train)) + "...")
        t1 = time.time()
        X, y = fddb_read_single_fold(path_fddb_root, fold_path, n_negs_per_img, hfs_coords, n, verbose=False, fold_title=fold_path)
        t2 = time.time()
        print("PROCESSING TRAIN FOLD " + str(index + 1) + "/" + str(len(fold_paths_train)) + " DONE IN " + str(t2 - t1) + " s.")
        print("---")
        if X_train is None:
            X_train = X
            y_train = y
        else:
            X_train = np.r_[X_train, X]
            y_train = np.r_[y_train, y]    
    fold_paths_test = [
        "FDDB-folds/FDDB-fold-10-ellipseList.txt",
        ]     
    X_test = None
    y_test = None
    for index, fold_path in enumerate(fold_paths_test):
        print("PROCESSING TEST FOLD " + str(index + 1) + "/" + str(len(fold_paths_test)) + "...")
        t1 = time.time()
        X, y = fddb_read_single_fold(path_fddb_root, fold_path, n_negs_per_img, hfs_coords, n, fold_title=fold_path)
        t2 = time.time()
        print("PROCESSING TEST FOLD " + str(index + 1) + "/" + str(len(fold_paths_test)) + " DONE IN " + str(t2 - t1) + " s.")
        print("---")
        if X_test is None:
            X_test = X
            y_test = y
        else:
            X_test = np.r_[X_test, X]
            y_test = np.r_[y_test, y]   
    print("TRAIN DATA SHAPE: " + str(X_train.shape))
    print("TEST DATA SHAPE: " + str(X_test.shape)) 
    return X_train, y_train, X_test, y_test

def pickle_all(fname, some_list):
    print("PICKLE...")
    t1 = time.time()
    f = open(fname, "wb+")
    pickle.dump(some_list, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()
    t2 = time.time()
    print("PICKLE DONE. [TIME: " + str(t2 - t1) + " s.]")

def unpickle_all(fname):
    print("UNPICKLE...")
    t1 = time.time()    
    f = open(fname, "rb")
    some_list = pickle.load(f)
    f.close()
    t2 = time.time()
    print("UNPICKLE DONE. [TIME: " + str(t2 - t1) + " s.]")
    return some_list 

def detect(i, clf, hfs_coords_subset, n, fi, clf_threshold=0):
    i_resized = resize_image(i)
    i_gray = gray_image(i_resized)
    ii = integral_image(i_gray)
    features = np.zeros(n)

    print(f"IMAGE SHAPE: {i_gray.shape}")

    n_windows_max = 0
    for s in range (DETECTION_SCALES):
        w = int(np.round(DETECTION_W_MIN * DETECTION_WINDOW_GROWTH**s))
        dj = int(np.round(w * DETECTION_WINDOW_JUMP))
        dk = dj
        j_start = int(((ii.shape[0] - w) % dj) / 2)
        k_start = int(((ii.shape[1] - w) % dk) / 2)
        for j in range(j_start, ii.shape[0] - w + 1, dj):
            for k in range(k_start, ii.shape[1] - w + 1, dk):
                n_windows_max += 1
    print(f"WINDOWS TO BE CHECKED: {n_windows_max}")

    progress_step = int(np.round(0.01 * n_windows_max))

    t1 = time.time()
    n_windows = 0
    for s in range (DETECTION_SCALES):
        w = int(np.round(DETECTION_W_MIN * DETECTION_WINDOW_GROWTH**s))
        dj = int(np.round(w * DETECTION_WINDOW_JUMP))
        dk = dj
        print("!S = " + str(s) + ", W = " + str(w) + " DJ = " + str(dj) + ", DK = " + str(dk) + "...")
        j_start = int(((ii.shape[0] - w) % dj) / 2)
        k_start = int(((ii.shape[1] - w) % dk) / 2)
        hfs_coords_window_subset = w * hfs_coords_subset
        hfs_coords_window_subset = np.array(list(map(lambda npa: npa.astype("int32"), hfs_coords_window_subset)))
        print(f"Loop from {j_start} to {ii.shape[0] - w + 1} step: {dj}")
        for j in range(j_start, ii.shape[0] - w + 1, dj):
            for k in range(k_start, ii.shape[1] - w + 1, dk):
                # t1_extraction = time.time()
                # features = haar_features(ii, hfs_coords_window_subset, j, k, n, fi)
                features = haar_features_2(ii, hfs_coords_window_subset, j, k, n, fi, features)
                # t2_extraction = time.time()
                # t1_decision = time.time()
                decision = clf.decision_function(np.array([features]))[0]
                # t2_decision = time.time()
                # print(f"EXTRACTION: {t2_extraction-t1_extraction} s., DECISION: {t2_decision - t1_decision}")
                if decision > clf_threshold:
                    print("!DETECTION AT " + str((j, k)))
                    cv2.rectangle(i_resized, (k, j), (k + w - 1, j + w - 1), (0, 0, 255), 1)
                n_windows += 1
                # if n_windows % progress_step == 0:
                #     print(f"PROGRESS: {np.round(n_windows / n_windows_max, 2)}")
    t2 = time.time()
    total_time = t2 - t1
    print(f"TOTAL TIME: {total_time} s.")
    time_per_window = total_time / n_windows
    print(f"TIME PER WINDOW: {time_per_window} s.")
    return i_resized

if __name__ == "__main__":
    print("STARTING...")     
    path_data_root = "../data/"
    path_clfs_root = "../clfs/"    
    s = 6
    p = 4
    hfs_indexes = haar_features_indexes(s, p)
    hfs_coords = haar_features_coordinates(hfs_indexes, s, p)
    n = len(hfs_indexes)
    print("NO. OF HAAR-LIKE FEATURES: " + str(n))
    data_description = "n_" + str(n) + "_s_" + str(s) + "_p_" + str(p)
    path_data = path_data_root + "fddb_" + data_description + ".pkl"
    
    
    # features demonstration on image and example window
#     i0 = cv2.imread(path_data_root + "000000.jpg")
#     i1 = resize_image(i0)
#     i = gray_image(i1)
#     print(i.shape)             
#     ii = integral_image(i)            
#     h = 69
#     w = 69    
#     hfs_coords_window = w * hfs_coords
#     hfs_coords_window = np.array(list(map(lambda npa: npa.astype("int32") , hfs_coords_window)))
#     j0 = 160
#     k0 = 280
#     p1 = (k0, j0)
#     p2 = (k0 + w - 1, j0 + h - 1)             
#     for hf_coords_window, index in zip(hfs_coords_window, hfs_indexes):
#         i2 = draw_haar_feature_at(i1, hf_coords_window, j0, k0)
#         cv2.rectangle(i2, p1, p2, (0, 0, 255), 1)        
#         cv2.imshow("FEATURE DEMO", i2)
#         feature = haar_feature(ii, hf_coords_window, j0, k0)        
#         print("FEATURE INDEX: " + str(index) + ", VALUE: " + str(feature) + ".")         
#         cv2.waitKey(0)
    
    # print("PREPARING DATA...");
    # t1 = time.time()
    # X_train, y_train, X_test, y_test = fddb_data("c:/Dev/machine_learning_2/data/large/", hfs_coords, 10, n)
    # t2 = time.time()
    # print("PREPARING DATA DONE IN " + str(t2 - t1) + " s.")
    # pickle_all(path_data, [X_train, y_train, X_test, y_test])   
  
    X_train, y_train, X_test, y_test = unpickle_all(path_data)
    train_index_pos = np.where(y_train == 1)[0]
    train_index_neg = np.where(y_train == 1)[0]
    print("X_TRAIN: " + str(X_train.shape) + " [POSITIVES: " + str(train_index_pos.size) + "]")
    test_index_pos = np.where(y_test == 1)[0]
    test_index_neg = np.where(y_test == -1)[0]
    print("X_TEST: " + str(X_test.shape) + " [POSITIVES: " + str(test_index_pos.size) + "]")
         
    # T = 128 # 128 słabych klasyfikatorów
    # MD = 1 # maximum depth
    # clf_description = data_description + "_T_" + str(T) + "_MD_" + str(MD)
    # path_clf = path_clfs_root + "fddb_ada_" + clf_description + ".pkl"
    # clf = AdaBoostClassifier(n_estimators=T, random_state=0, base_estimator=DecisionTreeClassifier(max_depth=MD))
    # print("LEARNING...");   
    # t1 = time.time()
    # clf.fit(X_train, y_train)
    # t2 = time.time()
    # print("LEARNING DONE IN " + str(t2 - t1) + " s.")
    # pickle_all(path_clf, [clf])
    # clf = unpickle_all(path_clf)[0]
    # fi = adaboost_features_indexes(n, clf)
    # print("SELECTED FEATURES: " + str(len(fi)))

    T = 128  # 128 słabych klasyfikatorów
    B = 8  # maximum depth
    clf_description = data_description + "_T_" + str(T) + "_B_" + str(B)
    path_clf = path_clfs_root + "fddb_real_" + clf_description + ".pkl"
    # clf = RealBoostBins(T, B)
    # print("LEARNING...")
    # t1 = time.time()
    # clf.fit(X_train, y_train)
    # t2 = time.time()
    # print(f"LEARNING DONE IN {t2 - t1} s.")
    # pickle_all(path_clf, [clf])
    clf = unpickle_all(path_clf)[0]
    fi = np.unique(clf.features_).astype("int32")
    print(f"SELECTED FEATURES {len(fi)}")

    # print("ACCURACY MEASURING...");
    # t1 = time.time()
    # print("TRAIN ACC: " + str(clf.score(X_train, y_train)))
    # y_test_df = clf.decision_function(X_test)
    # y_test_pred = (y_test_df > 0.0) * 2 - 1
    # acc = np.sum(y_test == y_test_pred) / len(y_test)
    # print("TEST ACC: " + str(acc))    
    # sens = np.sum(y_test[test_index_pos] == y_test_pred[test_index_pos]) / len(test_index_pos)
    # far = 1.0 - np.sum(y_test[test_index_neg] == y_test_pred[test_index_neg]) / len(test_index_neg)
    # print("TEST SENSITIVITY: " + str(sens))
    # print("TEST FAR: " + str(far))
    # t2 = time.time()
    # print("ACCURACY MEASUREING DONE IN " + str(t2 - t1) + " s.")

    i = cv2.imread(path_data_root + "000001.jpg")
    hfs_coords_subset = hfs_coords[fi]
    i_out = detect(i, clf, hfs_coords_subset, n, fi, clf_threshold=2.5)
    cv2.imshow("DETECTION OUTCOME", i_out)
    cv2.waitKey(0)
    
    print("DONE.")    

    # TODO: wariant równoległy detect 11:41
