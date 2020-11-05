from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np

class RealBoostBins(BaseEstimator, ClassifierMixin):
    """Działa tylko dla problemów binarnych"""
    def __init__(self, T=128, B=8, ):
        self.T_ = T  # liczba rund
        self.B_ = B  # liczba koszykow
        self.features_ = -1 * np.ones(self.T_, "int32")  # indeksy wybranych cech
        self.logits_ = np.zeros((self.T_, self.B_))
        self.class_labels_ = None

    def fit(self, X, y):
        m, n = X.shape
        self.class_labels_ = np.unique(y)
        indexes_neg = np.where(y == self.class_labels_[0])[0]  # pierwsza unikalna klasa to negatyw
        indexes_pos = np.where(y != self.class_labels_[0])[0]  # wszystkie oprócz pierwszej unikalnej klasy to pozytyw
        yy = -1 * np.ones(m, "int32")  # przygotowanie -1 i +1
        yy[indexes_pos] = 1
