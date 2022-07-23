from typing import Type
import numpy as np
from scipy.stats import boxcox

from .distributions import Distribution, NormalDistribution
from .interpolations import LinearInterpolation, Interpolation

def nan_boxcox(x):
    nans = np.nan * np.ones(x.size)
    idx = np.where(np.logical_not(np.isnan(x)))[0]
    bc, lmbda = boxcox(x[idx])
    nans[idx] = bc
    return nans, lmbda

def nan_boxcox_with_lambda(x, lmbda):
    nans = np.nan * np.ones(x.size)
    idx = np.where(np.logical_not(np.isnan(x)))[0]
    bc = boxcox(x[idx], lmbda)
    nans[idx] = bc
    return nans

boxcox_vectorize = np.vectorize(nan_boxcox, signature="(m) -> (m), ()")
boxcox_lambda_vectorize = np.vectorize(nan_boxcox_with_lambda, signature="(m), () -> (m)")

class AveragePercent():
    def __init__(self, interpolation: Type[Interpolation] = LinearInterpolation()):
        self.interpolation = interpolation

    def percent(self, X_T):
        pass

    def fit(self, X: np.ndarray):
        self.X = X
        self.Y = self.percent(self.X.T)
    
    def transform(self, X: np.ndarray):
        pass

    def predict(self, X: np.ndarray):
        pass

    def fit_predict(self, X):
        self.fit(X)
        return self.predict(X)

class ParametricAveragePercent():
    def __init__(self, distribution: Type[Distribution] = NormalDistribution(0, 1)):
        self.distribution = distribution
    
    def _fit(self, X: np.ndarray):
        self.mu = np.nanmean(X.T, axis=1)
        self.std = np.nanstd(X.T, axis=1, ddof=1)
    
    def fit(self, X):
        return self._fit(X)
    
    @staticmethod
    def _standardize_multivariate(X: np.ndarray, mu: np.ndarray, sigma: np.ndarray):
        return (X - mu) / sigma

    def standardize(self, X: np.ndarray):
        return self._standardize_multivariate(X, self.mu, self.std)
    
    def predict(self, X: np.ndarray):
        return np.nanmean(self.distribution.percent(self.standardize(X)), axis=1)
    
    def fit_predict(self, X):
        self.fit(X)
        return self.predict(X)

class BoxCoxParametricAveragePercent(ParametricAveragePercent):
    def __init__(self):
        super().__init__()
    
    @staticmethod
    @np.vectorize
    def boxcox_coef_pos(coefs):
        if coefs < 0:
            return -1
        return 1

    def fit(self, X):
        bc, lmbda = boxcox_vectorize(X.T)
        self.boxcox_lambda = lmbda
        self._fit(bc.T)
    
    def standardize(self, X: np.ndarray):
        boxcox_x = boxcox_lambda_vectorize(X.T, self.boxcox_lambda)
        return self._standardize_multivariate(boxcox_x.T, self.mu, self.std) * self.boxcox_coef_pos(self.boxcox_lambda)

