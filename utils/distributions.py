from abc import ABC, abstractmethod
from scipy.stats import norm

class Distribution(ABC):
    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def percent(self, val):
        pass

    @abstractmethod
    def inverse(self, p):
        pass

class NormalDistribution(Distribution):
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma
    
    def percent(self, val):
        return norm.cdf((val - self.mu) / self.sigma)
    
    def inverse(self, p):
        return self.mu + norm.ppf(p) * self.sigma
