import numpy as np

def euclidean(A: np.ndarray, B: np.ndarray):
    return np.sqrt(np.sum((A - B) ** 2))

def pearson(A: np.ndarray, B: np.ndarray):
    return np.corrcoef(A, B)[0, -1]

def elastic_euclidean(lmbda1: float, lmbda2: float, lmbda3: float):
    def _euclidean(A, B):
        return euclidean(A, B) * lmbda1 + np.sum(A ** 2) * lmbda2 + np.sum(np.abs(A)) * lmbda3
    
    return _euclidean