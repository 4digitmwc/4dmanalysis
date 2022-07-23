from typing import Callable
import numpy as np

class Interpolation():
    def __init__(self, interpolation_function: Callable):
        assert (interpolation_function(0) == 0 and interpolation_function(1) == 1)
        self.interpolation_function = interpolation_function
    
    def fit(self, X: np.ndarray, Y: np.ndarray):
        asort = X.argsort()
        self.X = X[asort]
        self.Y = Y[asort]
    
    def predict(self, X):
        @np.vectorize
        def interpolate(x):
            uniq_x, uniq_y_idx = np.unique(self.X, return_index=True)
            left = np.where(uniq_x <= x)[0]
            right = np.where(uniq_x > x)[0]
            if left.size == 0:
                left_idx = 0
                right_idx = 1
            elif right.size == 0:
                left_idx = uniq_x.size - 2
                right_idx = uniq_x.size - 1
            else:
                left_idx = left[-1]
                right_idx = right[0]
            
            
            t = (x - uniq_x[left_idx]) / (uniq_x[right_idx] - uniq_x[left_idx])
            return self.Y[uniq_y_idx][left_idx] * (1 - self.interpolation_function(t)) + self.Y[uniq_y_idx][right_idx] * t
        
        return interpolate(X)
    
    def fit_predict(self, X, Y):
        self.fit(X, Y)
        return self.predict(X)
                
    
class LinearInterpolation(Interpolation):
    def __init__(self):
        super().__init__(lambda x: x)
    
class PolynomialInterpolation(Interpolation):
    def __init__(self, degree):
        super().__init__(lambda x: x ** degree)
