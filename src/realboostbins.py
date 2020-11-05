from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
import time

class RealBoostBins(BaseEstimator, ClassifierMixin):
    """Działa tylko dla problemów binarnych"""
    OUTLIER_RATIO = 0.01

    def __init__(self, T=128, B=8, ):
        self.T_ = T  # liczba rund
        self.B_ = B  # liczba koszykow
        self.features_ = -1 * np.ones(self.T_, "int32")  # indeksy wybranych cech
        self.logits_ = np.zeros((self.T_, self.B_))
        self.class_labels_ = None
        self.mins_ = None  # minima oprócz outlierów o rozmiarze ile cech
        self.maxes_ = None

    def fit(self, X, y):
        m, n = X.shape
        self.class_labels_ = np.unique(y)
        indexes_neg = np.where(y == self.class_labels_[0])[0]  # pierwsza unikalna klasa to negatyw
        indexes_pos = np.where(y != self.class_labels_[0])[0]  # wszystkie oprócz pierwszej unikalnej klasy to pozytyw
        yy = -1 * np.ones(m, "int32")  # przygotowanie -1 i +1
        yy[indexes_pos] = 1

        print("SORTING FEATURES TO FIND RANGES")
        t1 = time.time()
        self.mins_ = np.zeros(n)
        self.maxes_ = np.zeros(n)
        i_min = int(np.ceil(self.OUTLIER_RATIO * m))
        i_max = int(np.floor((1.0 - self.OUTLIER_RATIO) * m))
        for j in range(n):  # iteracja po kolumnach
            feature = X[:, j].copy()
            feature.sort()
            self.mins_[j] = feature[i_min]
            self.maxes_[j] = feature[i_max]
        t2 = time.time()
        print(f"SORTING FEATURES TO FIND RANGED DONE. [TIME: {t2-t1} S]")
