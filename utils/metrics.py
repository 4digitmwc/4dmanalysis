import numpy as np

def euclidean(A: np.ndarray, B: np.ndarray):
    return np.sqrt(np.sum((B - A) ** 2, axis=1))

def nan_euclidean(A: np.ndarray, B: np.ndarray):
    dif = (B - A) ** 2
    dif = np.nan_to_num(dif)
    return np.sqrt(np.sum(dif, axis=1))

def pearson(A: np.ndarray, B: np.ndarray):
    return np.corrcoef(A, B)[0, -1]

def elastic_euclidean(lmbda1: float, lmbda2: float, lmbda3: float):
    def _euclidean(A, B):
        return euclidean(A, B) * lmbda1 + np.sum(A ** 2) * lmbda2 + np.sum(np.abs(A)) * lmbda3
    
    return _euclidean

def elastic_nan_euclidean(lmbda1: float, lmbda2: float, lmbda3: float):
    def _euclidean(A, B):
        return nan_euclidean(A, B) * lmbda1 + np.sum(np.nan_to_num(A) ** 2) * lmbda2 + np.sum(np.abs(np.nan_to_num(A))) * lmbda3
    
    return _euclidean
