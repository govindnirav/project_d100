import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class StandardScaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):  # y = None is for compatibility with sklearn API
        self.mean_ = np.mean(X, axis=0)
        stdev = np.std(X, axis=0)
        self.stdev_ = np.where(stdev == 0, 1, stdev)
        return self

    def transform(self, X):
        check_is_fitted(self)
        X_scaled = (X - self.mean_) / self.stdev_

        return X_scaled
