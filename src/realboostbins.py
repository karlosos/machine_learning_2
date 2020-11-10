from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
import time

class RealBoostBins(BaseEstimator, ClassifierMixin):
    """Działa tylko dla problemów binarnych"""
    OUTLIER_RATIO = 0.01
    LOGIT_MAX = 2

    def __init__(self, T=128, B=8, ):
        self.T_ = T  # liczba rund
        self.B_ = B  # liczba koszykow
        self.features_ = -1 * np.ones(self.T_, "int32")  # indeksy wybranych cech
        self.logits_ = np.zeros((self.T_, self.B_))
        self.class_labels_ = None
        self.mins_ = None  # minima oprócz outlierów o rozmiarze ile cech
        self.maxes_ = None

    def calculate_logits(self, W_neg, W_pos):
        B = len(W_neg)
        logits = np.zeros(B)
        for b in range(B):
            if W_pos[b] > 0.0 and W_neg[b] == 0:
                logits[b] = self.LOGIT_MAX
            elif W_neg[b] > 0.0 and W_pos[b] == 0.0:
                logits[b] = -self.LOGIT_MAX
            elif W_neg[b] > 0.0 and W_pos[b] > 0.0:
                logits[b] = 0.5 * np.log(W_pos[b] / W_neg[b])
        return logits

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

        print("BINNING ... ")
        t1 = time.time()
        X_binned = np.floor((X - self.mins_) / (self.maxes_ - self.mins_) * self.B_).astype("int32")
        X_binned = np.clip(X_binned, 0, self.B_ - 1)
        t2 = time.time()
        print(f"BINNING DONE. [TIME: {t2-t1} S]")

        print("INDEXER")
        t1 = time.time()
        indexer_neg = np.empty((n, self.B_), "object")
        indexer_pos = np.empty((n, self.B_), "object")
        for j in range(n):
            for b in range(self.B_):
                indexes_j_in_b = np.where(X_binned[:, j] == b)[0]
                indexer_neg[j, b] = np.intersect1d(indexes_j_in_b, indexes_neg)
                indexer_pos[j, b] = np.intersect1d(indexes_j_in_b, indexes_pos)
        t2 = time.time()
        print(f"INDEXER DONE. [TIME: {t2-t1} S]")

        print("BOOSTING...")
        w = np.ones(m) / m
        for t in range(self.T_):
            print(f"ROUND {t+1}/{self.T_}")
            j_best = -1
            err_exp_best = np.inf
            logits_best = None
            for j in range(n):
                W_neg = np.zeros(self.B_)
                W_pos = np.zeros(self.B_)
                for b in range(self.B_):
                    W_neg[b] = w[indexer_neg[j, b]].sum()
                    W_pos[b] = w[indexer_pos[j, b]].sum()
                logits = self.calculate_logits(W_neg, W_pos)
                err_exp = np.sum(w * np.exp(-yy * logits[X_binned[:, j]]))  # w[i] * np.exp(-yy[i] * f[i])
                if err_exp < err_exp_best:
                    err_exp_best = err_exp
                    j_best = j
                    logits_best = logits
            print(f"BEST FEATURE: {j_best}, ERR EXP BEST + {err_exp_best}")
            print(f"BEST LOGITS: {np.round(logits_best, 2)}")
            self.features_[t] = j_best
            self.logits_[t, :] = logits_best
            w = w * np.exp(-logits_best[X_binned[:, j_best]] * yy) / err_exp_best   # rewazenie, Z = err_exp_best

    def decision_function(self, X):
        m = X.shape[0]
        X_slice = X[:, self.features_]  # wycięcie do wybranych kolumn z boostingu
        mins = self.mins_[:, self.features_]
        maxes = self.maxes_[:, self.features_]
        X_binned = np.floor((X_slice - mins) / (maxes - mins) * self.B_).astype("int32")
        X_binned = np.clip(X_binned, 0, self.B_ - 1)
        F = np.zeros(m)
        for i in range(m):
            F[i] = self.logits_[np.arange(self.T_), X_binned[i, :]].sum()
        return F

    def predict(self, X):
        F = self.decision_function(X)
        return self.class_labels_[F > 0.0]

    # TODO:
    #  - [x] reważenie wag
    #  - [x] przyspieszenie intersection
    #  - [ ] predict
    #  - [ ] wygenerowac mniejszy zbior, kilkaset cech
    #  - [ ] zrównoleglić detect
    #  - [x] inna cecha może wyjście 12:24? BEST FEATURE: 6716 - jest dobrze
