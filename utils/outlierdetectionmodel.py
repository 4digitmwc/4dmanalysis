from typing import Type
import numpy as np
from scipy.stats import boxcox

from .distributions import Distribution, NormalDistribution

class WeightedPercent():
    def __init__(self, interpolation='linear'):
        self.interpolation = interpolation
    
    def fit(self, X):
        pass
    
    def predict(self, X):
        pass

    def fit_predict(self, X):
        self.fit(X)
        return self.predict(X)

class ParametricWeightedPercent():
    def __init__(self, distribution: Type[Distribution] = NormalDistribution(0, 1)):
        self.distribution = distribution

    def fit(self, X):
        pass
    
    def predict(self, X):
        pass

    def fit_predict(self, X):
        self.fit(X)
        return self.predict(X)